"""
deep_learning/dl_trainer.py
-----------------------------
Dedicated PyTorch training pipeline for deep learning models.

Issue #2  : ⚠ EXPERIMENTAL — deep learning support is experimental.
            Use tabular models (FLAML/sklearn) for production use cases.
Issue #12 : Model compatibility validation — warns when DL model selected
            for incompatible dataset type.
Issue #14 : GPU detection, mini-batch training, sequence windowing,
            early stopping, and loss/metric tracking.

Supported models
----------------
  LSTM  — time_series, sequential tabular
  GRU   — time_series, sequential tabular
  MLP   — tabular (classification / regression)

Install: pip install -r requirements-deep.txt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# -- Compatibility validation (Issue #12) --------------------------------------

# Maps model -> compatible task types
DL_COMPATIBILITY: Dict[str, List[str]] = {
    "LSTM":       ["time_series", "classification", "regression"],
    "GRU":        ["time_series", "classification", "regression"],
    "MLP":        ["classification", "regression"],
    "CNN1D":      ["time_series", "classification"],
    # BERT/DistilBERT: NLP AND text-based classification
    # (sentiment analysis, review classification, spam, intent detection
    #  are all "classification" tasks but operate on sentence/text meaning)
    "BERT":       ["nlp", "classification"],
    "DistilBERT": ["nlp", "classification"],
    "RoBERTa":    ["nlp", "classification"],
    "ResNet18":   ["computer_vision"],
    "ResNet50":   ["computer_vision"],
    "EfficientNet": ["computer_vision"],
    "ViT":        ["computer_vision"],
    "MobileNet":  ["computer_vision"],
}


def validate_dl_compatibility(model_name: str, task_type: str) -> Tuple[bool, str]:
    """
    Issue #12: Validates that model is compatible with the task/dataset type.

    Returns (is_compatible, warning_message).
    """
    compatible_tasks = DL_COMPATIBILITY.get(model_name)
    if compatible_tasks is None:
        return True, ""  # Unknown model — let it try
    if task_type not in compatible_tasks:
        msg = (
            f"⚠ [DL Compatibility] '{model_name}' is designed for "
            f"{compatible_tasks} tasks, but dataset is '{task_type}'. "
            f"Results may be poor. Consider using: "
            f"{[m for m, t in DL_COMPATIBILITY.items() if task_type in t]}"
        )
        return False, msg
    return True, ""


# -- Result dataclass ----------------------------------------------------------

@dataclass
class DLTrainingResult:
    model_name:    str
    task_type:     str
    epochs_run:    int
    train_losses:  List[float] = field(default_factory=list)
    val_losses:    List[float] = field(default_factory=list)
    metrics:       Dict[str, float] = field(default_factory=dict)
    best_epoch:    int = 0
    device_used:   str = "cpu"
    is_experimental: bool = True   # Issue #2


# -- Trainer -------------------------------------------------------------------

class DeepLearningTrainer:
    """
    ⚠ EXPERIMENTAL — Issue #2

    PyTorch-based trainer for LSTM, GRU, and MLP models.
    Handles GPU detection, mini-batch iteration, early stopping,
    and sequence windowing for time-series data.

    Usage
    -----
    trainer = DeepLearningTrainer(model_name="LSTM", task_type="time_series")
    result  = trainer.train(X_train, X_test, y_train, y_test)
    """

    def __init__(
        self,
        model_name: str = "MLP",
        task_type: str  = "classification",
        epochs: int     = 50,
        batch_size: int = 64,
        lr: float       = 1e-3,
        patience: int   = 10,           # early stopping
        window_size: int = 30,          # for LSTM/GRU sequence windowing
        hidden_size: int = 64,
        num_layers: int  = 2,
        dropout: float   = 0.2,
    ):
        self.model_name  = model_name
        self.task_type   = task_type
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.patience    = patience
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.model       = None

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> DLTrainingResult:
        """
        Full training loop with GPU support, batching, and early stopping.

        Parameters
        ----------
        X_train, X_test : Feature DataFrames (already preprocessed)
        y_train, y_test : Target Series

        Returns
        -------
        DLTrainingResult with metrics and training history.
        """
        # Issue #2: Experimental warning
        logger.warning(
            "⚠ [DeepLearningTrainer] EXPERIMENTAL — "
            "Deep learning support is under active development. "
            "Use tabular models for production pipelines."
        )

        # Issue #12: Compatibility check
        is_compat, compat_msg = validate_dl_compatibility(self.model_name, self.task_type)
        if not is_compat:
            logger.warning(compat_msg)

        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam
            from deep_learning.dl_utils import detect_device, create_sequences, make_dataloaders
            from deep_learning.dl_models import LSTMModel, GRUModel, MLPModel
        except ImportError as e:
            logger.error(
                f"PyTorch not installed: {e}. "
                "Run: pip install -r requirements-deep.txt"
            )
            return DLTrainingResult(
                model_name=self.model_name,
                task_type=self.task_type,
                epochs_run=0,
                metrics={"error": "PyTorch not installed"},
            )

        from config.settings import RANDOM_SEED
        torch.manual_seed(RANDOM_SEED)

        device = detect_device()

        # ── FIX: Safe numeric conversion — prevents "could not convert string
        # to float" crash when X_train has text/object columns (e.g. NLP tasks
        # where BERT/DistilBERT was selected but preprocessing left raw strings).
        #
        # For BERT/DistilBERT: HuggingFace models need raw text, not float arrays.
        # Since we don't have a full tokenizer pipeline here, we:
        #   1. Try to detect and handle text columns specially
        #   2. Fall back to numeric-only features if strings can't be converted
        #   3. Raise a clear error so main.py's except block catches it and
        #      falls through to AutoML (which handles text via TF-IDF)
        #
        # For LSTM/GRU/MLP: convert non-numeric columns via label encoding
        # so they can be used as float32 features.
        X_train_safe, X_test_safe = self._safe_to_float(X_train, X_test)

        X_tr = X_train_safe.values.astype(np.float32)
        X_te = X_test_safe.values.astype(np.float32)
        y_tr = y_train.values.astype(np.float32)
        y_te = y_test.values.astype(np.float32)

        n_features  = X_tr.shape[1]
        is_sequence = self.model_name in ("LSTM", "GRU")

        # Issue #14: Sequence windowing for LSTM/GRU
        if is_sequence:
            logger.info(
                f"[DL] Applying sequence windowing  window_size={self.window_size}"
            )
            X_tr, y_tr = create_sequences(X_tr, y_tr, self.window_size)
            X_te, y_te = create_sequences(X_te, y_te, self.window_size)
            logger.info(
                f"[DL] Sequences created — "
                f"X_train={X_tr.shape}  X_test={X_te.shape}"
            )

        train_loader, test_loader = make_dataloaders(
            X_tr, y_tr, X_te, y_te,
            batch_size=self.batch_size,
            device=device,
        )

        # Build model
        output_size = int(y_train.nunique()) if self.task_type == "classification" else 1
        if output_size == 2:
            output_size = 1  # binary -> sigmoid

        if self.model_name == "LSTM":
            self.model = LSTMModel(n_features, self.hidden_size, self.num_layers, output_size, self.dropout)
        elif self.model_name == "GRU":
            self.model = GRUModel(n_features, self.hidden_size, self.num_layers, output_size, self.dropout)
        else:
            self.model = MLPModel(n_features, output_size=output_size, dropout=self.dropout)

        self.model = self.model.to(device)

        # Loss function
        if self.task_type == "classification" and output_size == 1:
            criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,   # L2 regularization — prevents weight explosion / memorization
        )

        # Cosine annealing LR scheduler — reduces LR smoothly to help generalization
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        # Training loop with early stopping
        best_val_loss  = float("inf")
        patience_count = 0
        best_epoch     = 0
        train_losses, val_losses = [], []

        logger.info(
            f"[DL] Training {self.model_name}  "
            f"epochs={self.epochs}  lr={self.lr}  device={device}"
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                pred  = self.model(Xb).squeeze(-1)
                loss  = criterion(pred, yb)
                loss.backward()
                # Gradient clipping — prevents gradient explosion (memorization)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            # Step the LR scheduler after each epoch
            scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    pred = self.model(Xb).squeeze(-1)
                    val_loss += criterion(pred, yb).item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                logger.info(
                    f"[DL] Epoch {epoch}/{self.epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss  = val_loss
                best_epoch     = epoch
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(f"[DL] Early stopping at epoch {epoch}.")
                    break

        # Final metrics
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for Xb, _ in test_loader:
                p = self.model(Xb).squeeze(-1).cpu().numpy()
                all_preds.append(p)
        preds = np.concatenate(all_preds)

        from model_engine.evaluator import ModelEvaluator
        # Adjust y_te length to match preds (windowing may reduce length)
        y_eval = y_te[:len(preds)]
        if self.task_type == "classification":
            preds_class = (preds > 0.5).astype(int) if output_size == 1 else preds.argmax(axis=-1)
            metrics = ModelEvaluator()._compute_metrics(
                pd.Series(y_eval.astype(int)), preds_class, "classification"
            )
        else:
            metrics = ModelEvaluator()._compute_metrics(
                pd.Series(y_eval), preds, "regression"
            )

        logger.success(
            f"[DL] Training complete — best_epoch={best_epoch}  metrics={metrics}"
        )

        return DLTrainingResult(
            model_name=self.model_name,
            task_type=self.task_type,
            epochs_run=len(train_losses),
            train_losses=train_losses,
            val_losses=val_losses,
            metrics=metrics,
            best_epoch=best_epoch,
            device_used=str(device),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run inference on new data."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        import torch
        from deep_learning.dl_utils import detect_device
        device = detect_device()
        self.model.eval()
        X_safe, _ = self._safe_to_float(X, X)
        with torch.no_grad():
            t = torch.tensor(X_safe.values.astype(np.float32)).to(device)
            return self.model(t).squeeze(-1).cpu().numpy()

    # ── Helper: safe numeric conversion ──────────────────────────────────────

    def _safe_to_float(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ):
        """
        FIX: Convert all columns to numeric (float32-compatible) safely.

        Problem: "could not convert string to float" crashes happen when:
          - BERT/DistilBERT is selected for NLP but X_train has raw text strings
          - Any DL model receives mixed-type DataFrames

        Strategy:
          1. Already-numeric columns → keep as-is
          2. Object/string columns that look numeric → coerce with pd.to_numeric
          3. Object/string columns with few unique values → LabelEncode to int
          4. Object/string columns with many unique values (text) → drop with
             a warning (BERT/DistilBERT need a tokenizer, not float arrays;
             the outer except block in main.py will fall through to AutoML)
          5. After all transformations, fill any remaining NaN with 0

        Returns:
          (X_train_numeric, X_test_numeric) — pure float-compatible DataFrames
        """
        X_tr = X_train.copy()
        X_te = X_test.copy()

        for col in X_tr.columns:
            # Already numeric — nothing to do
            if pd.api.types.is_numeric_dtype(X_tr[col]):
                continue

            # Boolean → int
            if pd.api.types.is_bool_dtype(X_tr[col]):
                X_tr[col] = X_tr[col].astype(int)
                X_te[col] = X_te[col].astype(int) if col in X_te else 0
                continue

            # Try coercing to numeric (e.g. "3.14", "42")
            coerced_tr = pd.to_numeric(X_tr[col], errors="coerce")
            if coerced_tr.notna().mean() > 0.8:
                X_tr[col] = coerced_tr.fillna(0)
                X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(0) if col in X_te else 0
                continue

            # Categorical / low-cardinality string → LabelEncode
            n_unique = int(X_tr[col].nunique())
            if n_unique <= 200:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Fit on combined train+test values to avoid unseen-label errors
                all_vals = pd.concat([
                    X_tr[col].astype(str),
                    X_te[col].astype(str) if col in X_te.columns else pd.Series(dtype=str),
                ]).fillna("__nan__")
                le.fit(all_vals)
                X_tr[col] = le.transform(X_tr[col].astype(str).fillna("__nan__"))
                if col in X_te.columns:
                    X_te[col] = le.transform(
                        X_te[col].astype(str).fillna("__nan__").apply(
                            lambda v: v if v in le.classes_ else le.classes_[0]
                        )
                    )
                logger.debug(
                    f"[DL] LabelEncoded '{col}' ({n_unique} unique values)"
                )
                continue

            # High-cardinality text column — drop it
            # BERT/DistilBERT need a HuggingFace tokenizer pipeline, not floats.
            # Dropping here causes the DL path to fail gracefully, and main.py's
            # except block falls through to AutoML with TF-IDF preprocessing.
            logger.warning(
                f"[DL] Dropping high-cardinality text column '{col}' "
                f"({n_unique} unique values). "
                f"BERT/DistilBERT require a tokenizer — falling back to AutoML."
            )
            X_tr = X_tr.drop(columns=[col])
            if col in X_te.columns:
                X_te = X_te.drop(columns=[col])

        # If all columns were dropped, raise so AutoML fallback triggers
        if X_tr.shape[1] == 0:
            raise ValueError(
                "[DL] No numeric features remain after safe conversion. "
                "BERT/DistilBERT require a full tokenizer pipeline. "
                "Falling back to AutoML with TF-IDF preprocessing."
            )

        # Final NaN fill and conversion check
        X_tr = X_tr.fillna(0)
        X_te = X_te.fillna(0)

        # Verify all remaining columns are actually numeric
        non_numeric = [
            c for c in X_tr.columns
            if not pd.api.types.is_numeric_dtype(X_tr[c])
        ]
        if non_numeric:
            logger.warning(
                f"[DL] Dropping remaining non-numeric columns: {non_numeric}"
            )
            X_tr = X_tr.drop(columns=non_numeric)
            X_te = X_te.drop(columns=[c for c in non_numeric if c in X_te.columns])

        logger.info(
            f"[DL] Safe conversion done: {X_tr.shape[1]} numeric features "
            f"(from {X_train.shape[1]} original)"
        )
        return X_tr, X_te
