"""
preprocessing/pipeline_builder.py
------------------------------------
Production-grade preprocessing pipeline — handles ANY dataset.

Complete rewrite integrating DataTypeEngine for intelligent type detection.

SOLVES ALL DTYPE CRASHES:
  "could not convert string to float"  → numeric coercion + OrdinalEncoder
  "StringDtype cannot be interpreted"  → all strings encoded before training
  Models failing on raw text           → TF-IDF path for text columns
  Wrong task detection                 → DataTypeEngine detects from data
  No label encoding                    → TargetEncoder always applied

Pipeline flow:
  1. DataTypeEngine.analyse() → classify every column
  2. Drop: id, useless, >90% missing columns
  3. Datetime columns → decompose to year/month/day/hour
  4. Numeric strings → coerce to float
  5. Build ColumnTransformer per type:
       numeric     → Imputer(median) + StandardScaler
       categorical → Imputer(mode) + OHE (≤50 uniq) or OrdinalEncoder (>50)
       text        → TF-IDF (top 200 features)
  6. Split FIRST (no leakage), then fit on train only
  7. Post-encode validation: assert all features are numeric
  8. Feature reduction if >500 columns
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler,
)

from config.settings import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from preprocessing.data_type_engine import DataTypeEngine, TargetEncoder

MAX_OHE_CATEGORIES    = 50
MAX_FEATURES_THRESHOLD = 500
TFIDF_MAX_FEATURES    = 200


class PreprocessingPipeline:
    """
    Fully automatic preprocessing pipeline.
    Detects types, encodes, validates, then splits — zero data leakage.
    """

    def __init__(self):
        self.pipeline:        Optional[Pipeline]     = None
        self.target_encoder:  TargetEncoder          = TargetEncoder()
        self.feature_columns: List[str]              = []
        self.n_features_out:  int                    = 0
        self.type_profile                            = None   # DataTypeProfile
        self._feature_selector                       = None   # SelectKBest

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def build_and_fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        test_size: float   = DEFAULT_TEST_SIZE,
        random_state: int  = DEFAULT_RANDOM_STATE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Detect types, encode, split, fit — all in one call.
        Returns (X_train, X_test, y_train, y_test) as numeric DataFrames.
        """
        df = df.copy()

        # ── STEP 1: Detect all column types ───────────────────────────────────
        engine            = DataTypeEngine()
        self.type_profile = engine.analyse(
            df, target_col=target_column, task_type_hint=task_type
        )
        # Override task_type from data evidence
        task_type = self.type_profile.task_type

        # ── STEP 2: Preprocess the dataframe before split ─────────────────────
        df = self._preprocess_dataframe(df, target_column)

        # ── STEP 3: Separate X / y ────────────────────────────────────────────
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found after preprocessing. "
                f"Available: {df.columns.tolist()[:10]}"
            )

        y_raw = df[target_column]
        X_raw = df.drop(columns=[target_column])
        self.feature_columns = X_raw.columns.tolist()

        logger.info(f"[Pipeline] Features before encoding: {len(self.feature_columns)}")

        # ── STEP 4: Encode target ──────────────────────────────────────────────
        y = self.target_encoder.encode(y_raw, task_type)
        y = y.reset_index(drop=True)
        X_raw = X_raw.reset_index(drop=True)

        # ── STEP 5: Split FIRST — zero leakage ───────────────────────────────
        use_stratify = (
            task_type == "classification" and y.nunique() <= 20
        )
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y,
                test_size    = test_size,
                random_state = random_state,
                stratify     = y if use_stratify else None,
            )
        except ValueError:
            # Stratification failed (too few samples per class) — retry without
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y,
                test_size    = test_size,
                random_state = random_state,
            )
            use_stratify = False

        logger.info(
            f"[Pipeline] Split: train={len(X_train_raw):,} "
            f"test={len(X_test_raw):,} stratified={use_stratify}"
        )

        # ── STEP 6: Build ColumnTransformer ───────────────────────────────────
        transformers = self._build_transformers(X_train_raw)

        if not transformers:
            raise ValueError(
                "No valid columns remain after type detection and dropping. "
                "Check dataset — all columns may be IDs, constants, or empty."
            )

        ct            = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline = Pipeline([("preprocessor", ct)])

        # ── STEP 7: Fit on train ONLY, transform both ─────────────────────────
        # Purge StringDtype one final time right before sklearn sees the data.
        # This is the last line of defence against pandas 2.x StringDtype
        # that can slip through from any upstream step.
        X_train_raw = self._purge_string_dtype(X_train_raw)
        X_test_raw  = self._purge_string_dtype(X_test_raw)

        X_train_arr = self.pipeline.fit_transform(X_train_raw)
        X_test_arr  = self.pipeline.transform(X_test_raw)

        # Densify sparse matrix (TF-IDF produces sparse)
        if hasattr(X_train_arr, "toarray"):
            X_train_arr = X_train_arr.toarray()
            X_test_arr  = X_test_arr.toarray()

        X_train_arr = np.array(X_train_arr, dtype=np.float64)
        X_test_arr  = np.array(X_test_arr,  dtype=np.float64)

        # Replace inf/nan that may appear after TF-IDF or OrdinalEncoder
        X_train_arr = np.nan_to_num(X_train_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_arr  = np.nan_to_num(X_test_arr,  nan=0.0, posinf=0.0, neginf=0.0)

        n_features = X_train_arr.shape[1]
        logger.info(f"[Pipeline] Features after encoding: {n_features}")

        # ── STEP 8: Feature reduction ──────────────────────────────────────────
        if n_features > MAX_FEATURES_THRESHOLD:
            X_train_arr, X_test_arr = self._reduce_features(
                X_train_arr, X_test_arr, y_train.values, task_type
            )
            logger.info(
                f"[Pipeline] Features after SelectKBest: {X_train_arr.shape[1]}"
            )

        self.n_features_out = X_train_arr.shape[1]

        # ── STEP 9: Validate — no object dtype survives ───────────────────────
        self._validate_numeric(X_train_arr, "X_train")
        self._validate_numeric(X_test_arr,  "X_test")

        # ── STEP 10: Build DataFrames ─────────────────────────────────────────
        col_names = self._recover_column_names(X_train_arr.shape[1])

        X_train = pd.DataFrame(X_train_arr, columns=col_names)
        X_test  = pd.DataFrame(X_test_arr,  columns=col_names)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)

        logger.success(
            f"[Pipeline] Ready: X_train={X_train.shape} X_test={X_test.shape} "
            f"task={task_type}"
        )
        return X_train, X_test, y_train, y_test

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _preprocess_dataframe(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Apply type-specific pre-processing BEFORE building the sklearn pipeline.
        Handles: drop ID cols, parse datetime strings, coerce numeric strings.
        """
        profile = self.type_profile

        # Drop id/useless columns (but never the target)
        drop = [c for c in profile.drop_cols if c != target_column]
        if drop:
            df = df.drop(columns=[c for c in drop if c in df.columns])
            logger.info(f"[Pipeline] Dropped id/useless columns: {drop}")

        # Decompose datetime columns to numeric features
        dt_cols = [c for c in profile.datetime_cols if c != target_column]
        for col in dt_cols:
            if col not in df.columns:
                continue
            try:
                parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
                df[f"{col}_year"]  = parsed.dt.year.fillna(0).astype(int)
                df[f"{col}_month"] = parsed.dt.month.fillna(0).astype(int)
                df[f"{col}_day"]   = parsed.dt.day.fillna(0).astype(int)
                df[f"{col}_hour"]  = parsed.dt.hour.fillna(0).astype(int)
                df = df.drop(columns=[col])
                logger.info(
                    f"[Pipeline] Datetime '{col}' → "
                    f"year/month/day/hour columns"
                )
            except Exception as exc:
                logger.warning(f"[Pipeline] Datetime parse failed for '{col}': {exc}")

        # Coerce numeric strings to float
        num_str_cols = [
            c for c, cp in profile.column_profiles.items()
            if cp.col_type == "numeric" and c in df.columns
            and df[c].dtype == object
        ]
        for col in num_str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"[Pipeline] Coerced '{col}' to float")

        # Strip stray quotes from all remaining string columns.
        # CRITICAL pandas 2.x FIX: .astype(str) returns StringDtype in pandas 2.x,
        # which causes "Cannot interpret '<StringDtype>' as a data type" downstream.
        # Always cast BACK to numpy object dtype after any .str operation.
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            if col == target_column:
                continue
            df[col] = (
                df[col]
                .astype(object)           # ensure numpy object (not StringDtype)
                .fillna("")
                .astype(object)
                .str.strip("'\" ")
                .str.strip()
                .astype(object)           # force back to numpy object dtype
            )

        # Final sweep: neutralise any StringDtype columns that survive
        # (e.g. from pandas nullable dtypes, ArrowDtype, or read_csv options)
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype).lower()
                if "string" in dtype_str or dtype_str.startswith("str"):
                    df[col] = df[col].astype(object)
            except Exception:
                pass

        return df

    def _build_transformers(self, X_train: pd.DataFrame) -> list:
        """
        Build ColumnTransformer branches based on detected types.
        Uses X_train to measure cardinality — no leakage.
        """
        profile = self.type_profile

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        low_ohe_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=MAX_OHE_CATEGORIES,
            )),
        ])
        high_ord_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cast",    _StringCaster()),
            ("ord",     OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        transformers = []

        # Numeric columns (already numeric dtype or coerced)
        num_cols = [
            c for c in profile.numeric_cols
            if c in X_train.columns
        ]
        # Also include newly created datetime decomposition columns
        dt_derived = [
            c for c in X_train.columns
            if any(
                c.endswith(suf)
                for suf in ("_year", "_month", "_day", "_hour")
            )
        ]
        all_num = list(dict.fromkeys(num_cols + dt_derived))
        if all_num:
            transformers.append(("num", numeric_transformer, all_num))

        # Categorical: split OHE vs OrdinalEncoder by cardinality
        cat_low  = []
        cat_high = []
        for col in profile.categorical_cols:
            if col not in X_train.columns:
                continue
            n_unique = X_train[col].nunique()
            if n_unique <= MAX_OHE_CATEGORIES:
                cat_low.append(col)
            else:
                cat_high.append(col)

        if cat_low:
            transformers.append(("cat_low",  low_ohe_transformer,  cat_low))
        if cat_high:
            transformers.append(("cat_high", high_ord_transformer, cat_high))

        # Text columns → TF-IDF
        text_cols = [c for c in profile.text_cols if c in X_train.columns]
        for i, col in enumerate(text_cols):
            tfidf_pipe = Pipeline([
                ("fill", _TextFiller()),
                ("tfidf", TfidfVectorizer(
                    max_features  = TFIDF_MAX_FEATURES,
                    strip_accents = "unicode",
                    analyzer      = "word",
                    ngram_range   = (1, 2),
                )),
            ])
            # MUST pass [col] not col — ColumnTransformer iterates strings
            transformers.append((f"text_{i}", tfidf_pipe, [col]))

        log_parts = []
        if all_num:   log_parts.append(f"numeric={len(all_num)}")
        if cat_low:   log_parts.append(f"ohe={len(cat_low)}")
        if cat_high:  log_parts.append(f"ordinal={len(cat_high)}")
        if text_cols: log_parts.append(f"tfidf={len(text_cols)}")
        logger.info(f"[Pipeline] Transformers: {' | '.join(log_parts)}")

        return transformers

    def _reduce_features(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        task_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.warning(
            f"[Pipeline] {X_train.shape[1]} features > {MAX_FEATURES_THRESHOLD} "
            f"→ applying SelectKBest"
        )
        try:
            score_fn = f_classif if task_type == "classification" else f_regression
            k        = min(MAX_FEATURES_THRESHOLD, X_train.shape[1])
            self._feature_selector = SelectKBest(score_func=score_fn, k=k)
            Xt = self._feature_selector.fit_transform(X_train, y_train)
            Xs = self._feature_selector.transform(X_test)
            return Xt, Xs
        except Exception as exc:
            logger.warning(f"[Pipeline] SelectKBest failed ({exc}) — using all features")
            return X_train, X_test

    @staticmethod
    def _validate_numeric(arr: np.ndarray, name: str) -> None:
        """
        TASK 5: Assert all features are numeric.
        Raises immediately with a clear message if any non-numeric found.
        """
        if arr.dtype.kind not in ("f", "i", "u"):
            raise TypeError(
                f"[Pipeline] VALIDATION FAILED: {name} has dtype={arr.dtype}. "
                f"All features must be numeric before training. "
                f"Check DataTypeEngine output for unencoded columns."
            )
        if np.isnan(arr).any():
            logger.warning(
                f"[Pipeline] {name} contains NaN values — "
                f"replacing with 0 (nan_to_num already applied)"
            )

    def _recover_column_names(self, n_cols: int) -> List[str]:
        """Try to recover feature names from the pipeline."""
        try:
            names = (
                self.pipeline
                    .named_steps["preprocessor"]
                    .get_feature_names_out()
            )
            if len(names) >= n_cols:
                return list(names[:n_cols])
        except Exception:
            pass
        return [f"f_{i}" for i in range(n_cols)]

    @staticmethod
    def _purge_string_dtype(df: pd.DataFrame) -> pd.DataFrame:
        """
        Purge all pandas 2.x StringDtype columns by converting them to
        numpy object dtype. This is the definitive fix for:
            "Cannot interpret '<StringDtype(na_value=nan)>' as a data type"

        pandas 2.x introduced StringDtype as a first-class dtype. When any
        column has this dtype, numpy/sklearn/FLAML crash because they cannot
        use StringDtype as a numpy dtype descriptor.

        This method is called right before ColumnTransformer.fit_transform()
        as the last line of defence.
        """
        df = df.copy()
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype).lower()
                # Catch StringDtype, ArrowDtype(string), and any future variants
                if (
                    "string" in dtype_str
                    or dtype_str.startswith("str")
                    or isinstance(df[col].dtype, pd.StringDtype)
                ):
                    df[col] = df[col].astype(object)
            except Exception:
                pass
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data at inference time."""
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline not fitted. Call build_and_fit() first."
            )
        df = df.copy()
        if self.feature_columns:
            keep = [c for c in self.feature_columns if c in df.columns]
            df   = df[keep]
        arr = self.pipeline.transform(df)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        arr = np.nan_to_num(
            np.array(arr, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if self._feature_selector is not None:
            arr = self._feature_selector.transform(arr)
        return pd.DataFrame(arr)


# ── Custom sklearn transformers ───────────────────────────────────────────────

from sklearn.base import BaseEstimator, TransformerMixin


class _StringCaster(BaseEstimator, TransformerMixin):
    """Cast all values to string before OrdinalEncoder (handles mixed types)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.astype(str)
        import numpy as np
        return np.array(X, dtype=str)


class _TextFiller(BaseEstimator, TransformerMixin):
    """Fill NaN and convert to string for TF-IDF.
    
    ColumnTransformer passes a 2D array or DataFrame slice.
    TfidfVectorizer needs a 1D list of strings.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle all input types ColumnTransformer may pass
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
            return col.fillna("").astype(str).tolist()
        if isinstance(X, pd.Series):
            return X.fillna("").astype(str).tolist()
        # numpy array or similar
        import numpy as np
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return [str(v) if v is not None and str(v) != "nan" else "" for v in arr]
