"""
preprocessing/encoding.py
---------------------------
Standalone CategoricalEncoder for use outside of the sklearn Pipeline —
e.g. when you need to encode a DataFrame directly before passing it to
a custom model, the AutoML controller, or an NLP/vision pipeline that
doesn't go through pipeline_builder.py.

Note: pipeline_builder.py has its OWN internal OneHotEncoder/LabelEncoder
via ColumnTransformer. Use THIS module when you want to encode a DataFrame
independently (e.g. before passing to FLAML, a Transformer model, etc.).
Do NOT double-encode by running both.
"""

from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from loguru import logger


# Columns with > this many unique values get label-encoded instead of OHE
OHE_CARDINALITY_LIMIT = 15


class CategoricalEncoder:
    """
    Encodes categorical columns.
    Low-cardinality columns -> OneHotEncoding
    High-cardinality columns -> LabelEncoding
    Binary columns -> LabelEncoding (already 0/1)
    """

    def __init__(self):
        self._label_encoders: dict = {}

    def fit_transform(
        self, df: pd.DataFrame, target_column: str | None = None
    ) -> pd.DataFrame:
        df = df.copy()
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
            if c != target_column
        ]

        ohe_cols: List[str] = []
        le_cols:  List[str] = []

        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= 2 or n_unique > OHE_CARDINALITY_LIMIT:
                le_cols.append(col)
            else:
                ohe_cols.append(col)

        # One-Hot Encoding
        if ohe_cols:
            df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
            logger.info(f"OHE applied to: {ohe_cols}")

        # Label Encoding
        for col in le_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le

        if le_cols:
            logger.info(f"LabelEncoding applied to: {le_cols}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to new data (inference)."""
        df = df.copy()
        for col, le in self._label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        return df
