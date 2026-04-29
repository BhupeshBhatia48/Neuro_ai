"""
dataset_engine/dataset_validator.py
-------------------------------------
Validates a loaded DataFrame and produces a detailed ML-readiness
quality score before any processing begins.

Fixes applied
-------------
FIX-10 : Added quality_score (0–100), class imbalance ratio, per-column
         variance check, and missing-value ratio to ValidationReport.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class ValidationReport:
    is_valid: bool
    num_rows: int
    num_columns: int
    issues: List[str]
    warnings: List[str]
    # -- FIX-10: ML-readiness quality metrics ----------------------------------
    quality_score: float = 0.0          # 0–100
    missing_value_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    class_imbalance_ratio: float = 0.0  # max_class / min_class (1.0 = perfect)
    zero_variance_columns: List[str] = field(default_factory=list)
    column_missing_pct: Dict[str, float] = field(default_factory=dict)


class DatasetValidator:
    """
    Validates dataset integrity and scores ML readiness (0–100).
    Raises RuntimeError on critical failures.
    """

    MIN_ROWS    = 10
    MIN_COLUMNS = 2

    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> ValidationReport:
        """
        Parameters
        ----------
        df            : Raw loaded DataFrame.
        target_column : If provided, also checks class imbalance (classification).
        """
        issues:   List[str] = []
        warnings: List[str] = []

        # -- Critical checks ---------------------------------------------------
        if df.empty:
            issues.append("DataFrame is empty.")

        if len(df) < self.MIN_ROWS:
            issues.append(
                f"Dataset has only {len(df)} rows (minimum {self.MIN_ROWS})."
            )

        if len(df.columns) < self.MIN_COLUMNS:
            issues.append(
                f"Dataset has only {len(df.columns)} columns "
                f"(minimum {self.MIN_COLUMNS})."
            )

        dup_ratio = df.duplicated().sum() / max(len(df), 1)
        if dup_ratio > 0.8:
            issues.append(
                f"{dup_ratio:.0%} of rows are duplicates — dataset may be corrupt."
            )

        # -- Validate target column exists and is usable ----------------------
        if target_column:
            if target_column not in df.columns:
                issues.append(
                    f"Specified target column '{target_column}' not found. "
                    f"Available: {df.columns.tolist()[:8]}"
                )
            else:
                n_unique = df[target_column].nunique()
                if n_unique < 2:
                    issues.append(
                        f"Target column '{target_column}' has only {n_unique} "
                        f"unique value — cannot train a model."
                    )
                if df[target_column].isnull().mean() > 0.5:
                    issues.append(
                        f"Target column '{target_column}' has >50% missing values."
                    )

        # -- Check usable feature count ----------------------------------------
        feature_cols = [c for c in df.columns if c != target_column]
        usable_features = [
            c for c in feature_cols
            if df[c].nunique() > 1 and df[c].isnull().mean() < 0.9
        ]
        if len(usable_features) < 1:
            issues.append(
                f"No usable features found. All columns are either "
                f"constant or >90% missing."
            )

        # -- FIX-10: Quality metrics -------------------------------------------
        missing_ratio = float(df.isnull().mean().mean())
        if missing_ratio > 0.5:
            warnings.append(
                f"High missing-value ratio across all columns: {missing_ratio:.0%}."
            )

        # Per-column missing %
        col_missing = {
            col: round(float(df[col].isnull().mean()) * 100, 2)
            for col in df.columns
        }

        # Zero-variance columns
        zero_var_cols = [
            col for col in df.columns if df[col].nunique() <= 1
        ]
        if zero_var_cols:
            warnings.append(
                f"Zero-variance columns (will be dropped): {zero_var_cols}"
            )

        # Class imbalance ratio (only when target is known and categorical)
        imbalance_ratio = 1.0
        if target_column and target_column in df.columns:
            vc = df[target_column].value_counts()
            if len(vc) >= 2:
                imbalance_ratio = float(vc.iloc[0]) / float(vc.iloc[-1])
                if imbalance_ratio > 10:
                    warnings.append(
                        f"High class imbalance detected: "
                        f"{imbalance_ratio:.1f}x ratio "
                        f"(majority/minority class). Consider resampling."
                    )

        # -- FIX-10: Compute quality score 0–100 ------------------------------
        quality_score = self._compute_quality_score(
            num_rows=len(df),
            missing_ratio=missing_ratio,
            dup_ratio=float(dup_ratio),
            zero_var_count=len(zero_var_cols),
            total_cols=len(df.columns),
            imbalance_ratio=imbalance_ratio,
        )

        is_valid = len(issues) == 0

        report = ValidationReport(
            is_valid=is_valid,
            num_rows=len(df),
            num_columns=len(df.columns),
            issues=issues,
            warnings=warnings,
            quality_score=quality_score,
            missing_value_ratio=missing_ratio,
            duplicate_ratio=float(dup_ratio),
            class_imbalance_ratio=imbalance_ratio,
            zero_variance_columns=zero_var_cols,
            column_missing_pct=col_missing,
        )

        for w in warnings:
            logger.warning(w)

        logger.info(
            f"Dataset quality score: {quality_score:.1f}/100  "
            f"(missing={missing_ratio:.1%}, dupes={dup_ratio:.1%}, "
            f"imbalance={imbalance_ratio:.1f}x)"
        )

        if not is_valid:
            error_msg = "\n".join(issues)
            logger.error(f"Dataset validation failed:\n{error_msg}")
            raise RuntimeError(f"Dataset validation failed:\n{error_msg}")

        logger.success("Dataset validation passed.")
        return report

    # -- Quality scoring -------------------------------------------------------

    @staticmethod
    def _compute_quality_score(
        num_rows: int,
        missing_ratio: float,
        dup_ratio: float,
        zero_var_count: int,
        total_cols: int,
        imbalance_ratio: float,
    ) -> float:
        """
        Heuristic quality score out of 100.
        Higher = more ML-ready.
        """
        score = 100.0

        # Penalise missing values (up to -30)
        score -= min(missing_ratio * 60, 30)

        # Penalise duplicates (up to -20)
        score -= min(dup_ratio * 40, 20)

        # Penalise zero-variance columns (up to -15)
        if total_cols > 0:
            score -= min((zero_var_count / total_cols) * 30, 15)

        # Penalise class imbalance (up to -15)
        if imbalance_ratio > 1:
            score -= min((imbalance_ratio / 50) * 15, 15)

        # Penalise very small datasets (up to -20)
        if num_rows < 1000:
            score -= 20
        elif num_rows < 5000:
            score -= 10

        return round(max(score, 0.0), 1)

