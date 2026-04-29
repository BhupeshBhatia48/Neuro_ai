"""
data_analysis/dataset_analyzer.py
------------------------------------
Produces a DatasetProfile — a rich summary of the dataset
consumed by the model recommendation engine and preprocessing pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.constants import SMALL_DATASET_THRESHOLD, MEDIUM_DATASET_THRESHOLD


@dataclass
class DatasetProfile:
    num_rows: int
    num_features: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]           # long free-text fields
    missing_value_ratio: float        # 0-1
    duplicate_row_ratio: float        # 0-1
    target_column: Optional[str]
    task_type: Optional[str]          # refined after analysis
    size_bucket: str                  # "small" | "medium" | "large"
    column_stats: dict = field(default_factory=dict)  # per-column summary


class DatasetAnalyzer:
    """
    Analyses a DataFrame and returns a DatasetProfile.
    """

    TEXT_COLUMN_MEAN_WORDS = 5   # avg words > this -> treat as text

    def analyze(self, df: pd.DataFrame) -> DatasetProfile:
        logger.info("Analysing dataset …")

        numeric_cols    = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        datetime_cols = []
        text_cols     = []

        for col in categorical_cols[:]:
            # Detect datetime — infer_datetime_format removed in pandas 2.2,
            # use format="mixed" for flexible parsing
            try:
                parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                    continue
            except Exception:
                pass

            # Detect free-text (high avg word count)
            sample = df[col].dropna().astype(str).head(200)
            avg_words = sample.str.split().str.len().mean()
            if avg_words and avg_words > self.TEXT_COLUMN_MEAN_WORDS:
                text_cols.append(col)
                categorical_cols.remove(col)

        missing_ratio    = df.isnull().mean().mean()
        duplicate_ratio  = df.duplicated().sum() / max(len(df), 1)

        if len(df) < SMALL_DATASET_THRESHOLD:
            size_bucket = "small"
        elif len(df) < MEDIUM_DATASET_THRESHOLD:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        # Basic per-column stats
        col_stats = {}
        for col in df.columns:
            col_stats[col] = {
                "dtype":    str(df[col].dtype),
                "nunique":  int(df[col].nunique()),
                "missing%": round(float(df[col].isnull().mean()) * 100, 2),
            }

        profile = DatasetProfile(
            num_rows=len(df),
            num_features=len(df.columns),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
            missing_value_ratio=round(float(missing_ratio), 4),
            duplicate_row_ratio=round(float(duplicate_ratio), 4),
            target_column=None,     # filled by TargetDetector
            task_type=None,         # refined by TaskClassifier
            size_bucket=size_bucket,
            column_stats=col_stats,
        )

        logger.success(
            f"Profile -> rows={profile.num_rows:,}  features={profile.num_features}  "
            f"size={profile.size_bucket}  missing={profile.missing_value_ratio:.1%}"
        )
        return profile
