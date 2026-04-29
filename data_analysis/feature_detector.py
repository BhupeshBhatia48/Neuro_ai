"""
data_analysis/feature_detector.py
------------------------------------
Identifies feature columns and flags special properties
(high-cardinality, constant, redundant, etc.).
"""

from typing import Dict, List

import pandas as pd
from loguru import logger

from data_analysis.dataset_analyzer import DatasetProfile


class FeatureDetector:
    """
    Enriches a DatasetProfile with feature-level quality flags.
    Returns a dict: column_name -> list of flags.
    """

    HIGH_CARDINALITY_THRESHOLD = 0.95   # nunique / nrows

    def detect(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> Dict[str, List[str]]:
        """
        Returns feature_flags: { col_name: [flag1, flag2, ...] }
        Possible flags: "constant", "id_like", "high_cardinality", "skewed"
        """
        feature_flags: Dict[str, List[str]] = {}
        n = len(df)

        for col in df.columns:
            flags: List[str] = []
            series = df[col]

            # Constant column
            if series.nunique() <= 1:
                flags.append("constant")

            # ID-like column (all unique values + int/string)
            elif series.nunique() / n > self.HIGH_CARDINALITY_THRESHOLD:
                flags.append("high_cardinality")
                if "id" in col.lower() or "index" in col.lower():
                    flags.append("id_like")

            # Numeric skew
            if col in profile.numeric_columns:
                try:
                    skew = float(df[col].skew())
                    if abs(skew) > 2.0:
                        flags.append("skewed")
                except Exception:
                    pass

            feature_flags[col] = flags

        dropped = [c for c, f in feature_flags.items() if "constant" in f or "id_like" in f]
        if dropped:
            logger.warning(f"Columns suggested for removal: {dropped}")

        return feature_flags
