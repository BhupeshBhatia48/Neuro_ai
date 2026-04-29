"""
preprocessing/cleaning.py
---------------------------
Dataset cleaning: removes structural problems before type detection.

Responsibilities:
  1. Strip stray quote characters ('f' stored as "'f'")
  2. Drop explicitly flagged columns (IDs, constants)
  3. Drop columns with >60% missing values
  4. Remove fully duplicate rows
  5. Fill remaining NaN values (median/mode)
  6. Coerce obvious numeric strings to float
  7. Convert pandas StringDtype to object (sklearn compatibility)

Note: DataTypeEngine handles the intelligent type classification.
      cleaning.py only handles structural issues.
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    HIGH_MISSING_THRESHOLD = 0.6

    def clean(
        self,
        df: pd.DataFrame,
        drop_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        initial_shape = df.shape

        # ── FIX: Convert pandas StringDtype → object (sklearn compatibility) ──
        # pandas 2.x uses StringDtype for string columns but sklearn/FLAML
        # expect object dtype. This causes "StringDtype cannot be interpreted"
        for col in df.columns:
            if hasattr(df[col], "dtype") and str(df[col].dtype) in (
                "string", "StringDtype", "string[python]", "string[pyarrow]"
            ):
                df[col] = df[col].astype(object)

        # ── Strip stray quote characters from all string columns ──────────────
        # CRITICAL pandas 2.x FIX: .str accessor returns StringDtype in pandas 2.x
        # which later causes "Cannot interpret '<StringDtype>' as a data type".
        # We MUST cast back to numpy object dtype after every .str operation.
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            try:
                df[col] = (
                    df[col]
                    .astype(object)      # ensure numpy object first
                    .str.strip("'\"")
                    .str.strip()
                    .astype(object)      # force back to numpy object dtype
                )
            except Exception:
                pass

        # Also neutralise any pandas StringDtype columns that slipped through
        # (e.g. from pd.read_csv with dtype_backend='numpy_nullable' or similar)
        for col in df.columns:
            try:
                dtype_name = str(df[col].dtype).lower()
                if "string" in dtype_name or dtype_name.startswith("string"):
                    df[col] = df[col].astype(object)
            except Exception:
                pass

        if len(str_cols) > 0:
            logger.info(
                f"[Cleaning] Stripped stray quotes from "
                f"{len(str_cols)} string columns"
            )

        # ── Drop explicitly flagged columns ───────────────────────────────────
        if drop_columns:
            to_drop = [c for c in drop_columns if c in df.columns]
            if to_drop:
                df.drop(columns=to_drop, inplace=True)
                logger.info(f"[Cleaning] Dropped flagged columns: {to_drop}")

        # ── Drop columns with >60% missing values ─────────────────────────────
        before = df.shape[1]
        df.dropna(
            axis=1,
            thresh=int(len(df) * (1 - self.HIGH_MISSING_THRESHOLD)),
            inplace=True,
        )
        dropped_missing = before - df.shape[1]
        if dropped_missing:
            logger.info(
                f"[Cleaning] Dropped {dropped_missing} high-missing columns "
                f"(>{self.HIGH_MISSING_THRESHOLD:.0%} missing)"
            )

        # ── Drop fully duplicate rows ─────────────────────────────────────────
        before_rows = len(df)
        df.drop_duplicates(inplace=True)
        removed_dup = before_rows - len(df)
        if removed_dup:
            logger.info(f"[Cleaning] Removed {removed_dup:,} duplicate rows")

        # ── Coerce obvious numeric strings to float ───────────────────────────
        # Catches columns like "3.14" stored as string
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            sample = df[col].dropna().head(50)
            converted = pd.to_numeric(sample, errors="coerce")
            if converted.notna().mean() > 0.9:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info(
                    f"[Cleaning] Coerced numeric string column '{col}' → float"
                )

        # ── Fill remaining numeric NaN with median ────────────────────────────
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            if df[col].isnull().any():
                median = df[col].median()
                df[col] = df[col].fillna(0.0 if np.isnan(median) else median)

        # ── Fill remaining categorical NaN with mode ──────────────────────────
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill     = mode_val[0] if len(mode_val) else "unknown"
                df[col]  = df[col].fillna(fill)

        # ── Final: ensure no pandas extension dtypes remain ───────────────────
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if "Int" in dtype_str or "Float" in dtype_str:
                # pandas nullable Int64/Float64 → standard numpy dtype
                df[col] = df[col].astype(
                    float if "Float" in dtype_str else "Int64"
                ).fillna(0)
                df[col] = df[col].astype(
                    np.float64 if "Float" in dtype_str else np.int64
                )

        logger.success(
            f"[Cleaning] Done: {initial_shape} → "
            f"{len(df):,} rows x {df.shape[1]} cols"
        )
        return df
