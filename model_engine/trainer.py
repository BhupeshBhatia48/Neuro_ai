"""
model_engine/trainer.py
-------------------------
Trains multiple models in manual mode and returns all fitted estimators.

BUG FIX: Previous implementation trained models sequentially but
model_selector only recommended one model per run in many cases.

FIX: Manual mode now ALWAYS trains ALL recommended models, not just
the first one. This gives the evaluator real comparison data.

Also fixed: No second train/test split here — uses the pre-split data
from PreprocessingPipeline to avoid data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from config.settings import DEFAULT_RANDOM_STATE
from model_engine.candidate_models import get_model, available_models

# Compatibility matrix: task -> models that are inappropriate
INCOMPATIBLE = {
    "time_series":     ["LogisticRegression", "SVM", "SVR", "LinearRegression", "Ridge"],
    "nlp":             ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
                        "LogisticRegression", "SVM", "SVR", "LinearRegression", "Ridge"],
    "computer_vision": ["RandomForest", "XGBoost", "LightGBM", "CatBoost",
                        "LogisticRegression", "SVM", "SVR", "LinearRegression", "Ridge"],
}

DEEP_LEARNING_MODELS = {
    "LSTM", "GRU", "MLP", "BERT", "DistilBERT", "RoBERTa",
    "ResNet18", "ResNet50", "EfficientNet", "EfficientNetB7",
    "ViT", "MobileNet", "ARIMA", "Prophet",
}


class ModelTrainer:
    """
    Trains all selected models and returns them with the test split.
    FIX: Trains ALL models in the list, not just the top one.
    """

    def train_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_names: List[str],
        task_type: str,
    ) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series]:
        """
        Trains every model in model_names on X_train/y_train.
        Returns ALL trained models so evaluator can compare them.

        Parameters
        ----------
        X_train, X_test, y_train, y_test : Pre-split from PreprocessingPipeline
        model_names : All models to train (not just top-1)
        task_type   : classification | regression | etc.
        """
        logger.info(
            f"[Trainer] Training {len(model_names)} models | "
            f"train={len(X_train):,} rows | test={len(X_test):,} rows"
        )

        trained: Dict[str, object] = {}
        available = available_models(task_type)

        for name in model_names:
            # Compatibility check
            warning = self._check_compatibility(name, task_type)
            if warning:
                logger.warning(warning)

            # Deep learning models go to DeepLearningTrainer
            if name in DEEP_LEARNING_MODELS:
                logger.warning(
                    f"'{name}' requires DeepLearningTrainer (PyTorch). "
                    "Skipping in tabular trainer."
                )
                continue

            if name not in available:
                logger.warning(
                    f"'{name}' not in standard registry for '{task_type}'. "
                    "Attempting anyway..."
                )

            try:
                model = get_model(name, task_type)
                logger.info(f"  Training {name}...")
                model.fit(X_train, y_train)
                trained[name] = model
                logger.success(f"  OK {name} trained.")
            except Exception as exc:
                logger.error(f"  FAIL {name}: {exc}")

        if not trained:
            logger.error(
                "[Trainer] No models trained successfully. "
                "Check model names and task type."
            )
        else:
            logger.success(
                f"[Trainer] Trained {len(trained)}/{len(model_names)} models: "
                f"{list(trained.keys())}"
            )

        return trained, X_test, y_test

    @staticmethod
    def _check_compatibility(model_name: str, task_type: str) -> str | None:
        incompatible_list = INCOMPATIBLE.get(task_type, [])
        if model_name in incompatible_list:
            return (
                f"Compatibility warning: '{model_name}' is not designed for "
                f"'{task_type}' tasks."
            )
        return None
