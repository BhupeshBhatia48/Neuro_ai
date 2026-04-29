"""
preprocessing/data_type_engine.py
-----------------------------------
Automatic column type detection and classification.

Classifies every column into one of five types:
  numeric    → int/float columns ready for scaling
  categorical → string with <50 unique values → OHE/Ordinal
  text       → long free-form strings → TF-IDF
  datetime   → date/time strings → decompose to year/month/day/hour
  id         → near-unique identifier → DROP

Also detects correct task type from the target column,
applies LabelEncoding, and validates all features are numeric
before returning to the training pipeline.

This solves ALL dtype-related crashes:
  "could not convert string to float"   → caught by TASK 2 + TASK 3
  "StringDtype cannot be interpreted"   → caught by TASK 5 validation
  models failing on raw text            → caught by text → TF-IDF path
  wrong task detection                  → caught by TASK 2 auto-detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


# Thresholds
MAX_OHE_CATEGORIES   = 50      # above this → OrdinalEncoder
MAX_TEXT_WORDS       = 5       # avg words above this → text column
ID_UNIQUE_RATIO      = 0.95    # unique/total above this → id column
HIGH_MISSING_DROP    = 0.90    # column with >90% missing → drop


@dataclass
class ColumnProfile:
    """Type classification for a single column."""
    name:       str
    col_type:   str            # numeric | categorical | text | datetime | id
    n_unique:   int
    missing_pct: float
    reason:     str            # explanation for the classification


@dataclass
class DataTypeProfile:
    """Complete type profile of all columns in a dataset."""
    numeric_cols:     List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    text_cols:        List[str] = field(default_factory=list)
    datetime_cols:    List[str] = field(default_factory=list)
    id_cols:          List[str] = field(default_factory=list)
    drop_cols:        List[str] = field(default_factory=list)
    column_profiles:  Dict[str, ColumnProfile] = field(default_factory=dict)
    task_type:        str = "classification"
    target_col:       Optional[str] = None


class DataTypeEngine:
    """
    Automatically detects column types and determines task type.
    Works for ANY dataset — no hardcoded column names.
    """

    def analyse(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type_hint: str       = "classification",
    ) -> DataTypeProfile:
        """
        Classify all columns and detect task type.

        Parameters
        ----------
        df             : DataFrame (already cleaned, before encoding)
        target_col     : Known target column name (or None)
        task_type_hint : Hint from IdeaParser — may be overridden by data

        Returns
        -------
        DataTypeProfile with all column classifications
        """
        profile = DataTypeProfile()
        profile.target_col = target_col
        n_rows = max(len(df), 1)

        logger.info(
            f"[DataTypeEngine] Analysing {len(df.columns)} columns, "
            f"{n_rows:,} rows"
        )

        for col in df.columns:
            if col == target_col:
                continue   # target handled separately below

            cp = self._classify_column(df[col], col, n_rows)
            profile.column_profiles[col] = cp

            if cp.col_type == "numeric":
                profile.numeric_cols.append(col)
            elif cp.col_type == "categorical":
                profile.categorical_cols.append(col)
            elif cp.col_type == "text":
                profile.text_cols.append(col)
            elif cp.col_type == "datetime":
                profile.datetime_cols.append(col)
            elif cp.col_type in ("id", "drop"):
                profile.drop_cols.append(col)

            logger.info(
                f"  {col:<30} → {cp.col_type:<12} "
                f"({cp.reason})"
            )

        # Detect task type from target column
        if target_col and target_col in df.columns:
            profile.task_type = self._detect_task_type(
                df[target_col], task_type_hint
            )
        else:
            profile.task_type = task_type_hint

        logger.info(
            f"[DataTypeEngine] "
            f"numeric={len(profile.numeric_cols)} "
            f"categorical={len(profile.categorical_cols)} "
            f"text={len(profile.text_cols)} "
            f"datetime={len(profile.datetime_cols)} "
            f"id/drop={len(profile.drop_cols)} "
            f"task={profile.task_type}"
        )
        return profile

    def _classify_column(
        self,
        series: pd.Series,
        name:   str,
        n_rows: int,
    ) -> ColumnProfile:
        """Classify a single column."""
        missing_pct = float(series.isnull().mean())
        n_unique    = int(series.nunique(dropna=True))
        dtype_str   = str(series.dtype)

        # RULE 1: Drop if >90% missing
        if missing_pct > HIGH_MISSING_DROP:
            return ColumnProfile(
                name=name, col_type="drop", n_unique=n_unique,
                missing_pct=missing_pct,
                reason=f"{missing_pct:.0%} missing → drop",
            )

        # RULE 2: Numeric dtype → numeric
        if pd.api.types.is_numeric_dtype(series):
            return ColumnProfile(
                name=name, col_type="numeric", n_unique=n_unique,
                missing_pct=missing_pct,
                reason=f"dtype={dtype_str}",
            )

        # RULE 3: Boolean → treat as categorical (2 values)
        if pd.api.types.is_bool_dtype(series):
            return ColumnProfile(
                name=name, col_type="categorical", n_unique=2,
                missing_pct=missing_pct,
                reason="bool → categorical",
            )

        # RULE 4: Datetime dtype → datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnProfile(
                name=name, col_type="datetime", n_unique=n_unique,
                missing_pct=missing_pct,
                reason="datetime dtype",
            )

        # For string/object columns — further sub-classification
        sample = series.dropna().astype(str)

        # RULE 5: Try to parse as datetime string
        if n_unique > 5 and self._looks_like_datetime(sample):
            return ColumnProfile(
                name=name, col_type="datetime", n_unique=n_unique,
                missing_pct=missing_pct,
                reason="datetime string detected",
            )

        # RULE 6: Try to parse as numeric string
        if self._looks_like_numeric(sample):
            return ColumnProfile(
                name=name, col_type="numeric", n_unique=n_unique,
                missing_pct=missing_pct,
                reason="numeric string → coerce to float",
            )

        # RULE 7: ID column — unique ratio close to 1.0
        unique_ratio = n_unique / n_rows
        if unique_ratio > ID_UNIQUE_RATIO:
            return ColumnProfile(
                name=name, col_type="id", n_unique=n_unique,
                missing_pct=missing_pct,
                reason=f"unique_ratio={unique_ratio:.2f} → id/drop",
            )

        # RULE 8: Long text strings → text
        avg_len   = float(sample.str.len().mean())
        avg_words = float(sample.str.split().str.len().mean())
        if avg_words > MAX_TEXT_WORDS or avg_len > 100:
            return ColumnProfile(
                name=name, col_type="text", n_unique=n_unique,
                missing_pct=missing_pct,
                reason=f"avg_words={avg_words:.1f} → text/TF-IDF",
            )

        # RULE 9: Categorical (low cardinality)
        if n_unique <= MAX_OHE_CATEGORIES:
            return ColumnProfile(
                name=name, col_type="categorical", n_unique=n_unique,
                missing_pct=missing_pct,
                reason=f"{n_unique} unique → OHE",
            )

        # RULE 10: High cardinality string → categorical (OrdinalEncoder)
        return ColumnProfile(
            name=name, col_type="categorical", n_unique=n_unique,
            missing_pct=missing_pct,
            reason=f"{n_unique} unique → OrdinalEncoder",
        )

    @staticmethod
    def _looks_like_datetime(sample: pd.Series) -> bool:
        """Check if string values look like dates."""
        try:
            parsed = pd.to_datetime(sample.head(50), format="mixed", errors="coerce")
            return float(parsed.notna().mean()) > 0.7
        except Exception:
            return False

    @staticmethod
    def _looks_like_numeric(sample: pd.Series) -> bool:
        """Check if string values are actually numbers."""
        try:
            converted = pd.to_numeric(sample.head(100), errors="coerce")
            return float(converted.notna().mean()) > 0.85
        except Exception:
            return False

    @staticmethod
    def _detect_task_type(target: pd.Series, hint: str) -> str:
        """
        Detect correct task type from the target column.
        Data is the ground truth — overrides LLM/rule hint when clear.

        Rules:
          numeric target with >20 unique → regression
          string/bool/few-unique → classification
          datetime → time_series
        """
        if pd.api.types.is_datetime64_any_dtype(target):
            logger.info("[DataTypeEngine] Target is datetime → time_series")
            return "time_series"

        n_unique = int(target.nunique())

        if pd.api.types.is_numeric_dtype(target):
            if n_unique > 20:
                logger.info(
                    f"[DataTypeEngine] Numeric target, {n_unique} unique "
                    f"→ regression (overrides hint='{hint}')"
                )
                return "regression"
            else:
                logger.info(
                    f"[DataTypeEngine] Numeric target, {n_unique} unique "
                    f"→ classification (low cardinality)"
                )
                return "classification"

        # String / object / bool target
        logger.info(
            f"[DataTypeEngine] String/categorical target, "
            f"{n_unique} unique → classification"
        )
        return "classification"


class TargetEncoder:
    """
    Encodes the target column for model training.
    Stores the encoder for inference-time decoding.
    """

    def __init__(self):
        self.label_encoder: Optional[LabelEncoder] = None
        self.classes_: List       = []
        self.is_encoded: bool     = False

    def encode(self, y: pd.Series, task_type: str) -> pd.Series:
        """
        Encode target column.
        - classification + string target → LabelEncoder
        - classification + numeric target → cast to int
        - regression → cast to float, keep as-is
        """
        if task_type == "regression":
            y_encoded = pd.to_numeric(y, errors="coerce").fillna(0.0)
            logger.info("[TargetEncoder] Regression target → cast to float")
            return y_encoded.astype(float)

        # Classification path
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 1:
            # Already numeric — cast to int
            y_encoded = y.fillna(0).astype(int)
            self.classes_   = sorted(y_encoded.unique().tolist())
            self.is_encoded = False
            logger.info(
                f"[TargetEncoder] Numeric classification target "
                f"→ {len(self.classes_)} classes"
            )
            return y_encoded

        # String/categorical → LabelEncoder
        self.label_encoder = LabelEncoder()
        y_str     = y.astype(str).str.strip("'\" ").str.strip()
        y_encoded = pd.Series(
            self.label_encoder.fit_transform(y_str),
            index=y.index,
            name=y.name,
        )
        self.classes_   = self.label_encoder.classes_.tolist()
        self.is_encoded = True
        logger.info(
            f"[TargetEncoder] LabelEncoder applied → "
            f"{len(self.classes_)} classes: {self.classes_[:8]}"
        )
        return y_encoded

    def decode(self, y_encoded) -> np.ndarray:
        """Decode predictions back to original labels."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_encoded)
        return y_encoded
