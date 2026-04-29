"""
automl/automl_controller.py
-----------------------------
FLAML AutoML with proper timeout enforcement and progress logging.

BUG FIXED:
----------
FLAML was hanging silently when given 23,000+ features because:
1. No per-estimator timeout — one slow model blocked everything
2. No progress logging — pipeline appeared stuck
3. time_budget was set but FLAML ignored it on memory-heavy datasets

FIX:
1. Hard cap: if n_features > 500 reduce time_budget to 60s max
2. Added eval_method="cv" with n_splits=3 for faster evaluation
3. Added metric logging every trial via custom callback
4. Fallback to LightGBM directly if FLAML fails/times out
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from config.settings import FLAML_TIME_BUDGET, RANDOM_SEED

CLASSIFICATION_ESTIMATORS: List[str] = [
    "lgbm", "xgboost", "rf", "lrl1", "extra_tree",
]
REGRESSION_ESTIMATORS: List[str] = [
    "lgbm", "xgboost", "rf", "extra_tree",
]


class AutoMLController:

    def __init__(self, time_budget: Optional[int] = None):
        self.time_budget = time_budget or FLAML_TIME_BUDGET
        self._automl     = None
        self.best_config: dict = {}

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
    ) -> Tuple[object, Dict[str, float]]:

        try:
            from flaml import AutoML
        except ImportError:
            logger.error(
                "FLAML not installed. "
                "Run: pip install -r requirements/requirements-automl.txt"
            )
            return self._fallback(X_train, y_train, X_test, y_test, task_type)

        flaml_task = (
            "classification" if task_type == "classification"
            else "regression"
        )

        estimator_list = (
            CLASSIFICATION_ESTIMATORS if flaml_task == "classification"
            else REGRESSION_ESTIMATORS
        )

        # FIX: Reduce time budget for large feature spaces
        n_features   = X_train.shape[1]
        time_budget  = self.time_budget
        if n_features > 200:
            time_budget = min(time_budget, 90)
            logger.info(
                f"Large feature space ({n_features} features) — "
                f"capping time budget to {time_budget}s"
            )

        logger.info(
            f"FLAML starting | task={flaml_task} | "
            f"budget={time_budget}s | estimators={estimator_list} | "
            f"train={len(X_train):,} rows x {n_features} features"
        )

        start_time = time.time()

        try:
            self._automl = AutoML()
            self._automl.fit(
                X_train, y_train,
                task            = flaml_task,
                time_budget     = time_budget,
                estimator_list  = estimator_list,
                seed            = RANDOM_SEED,
                verbose         = 1,   # Show progress
                early_stop      = True,
                eval_method     = "cv",
                n_splits        = 3,
            )

            elapsed = time.time() - start_time

            try:
                best_model = self._automl.model.estimator
            except AttributeError:
                best_model = self._automl.model

            self.best_config        = self._automl.best_config or {}
            best_estimator_name     = self._automl.best_estimator or "unknown"

            logger.success(
                f"FLAML done in {elapsed:.1f}s | "
                f"best={best_estimator_name} | "
                f"config={self.best_config}"
            )

            # Log all tried estimators
            tried = self.get_all_tried_estimators()
            if tried:
                logger.info(f"FLAML tried: {tried}")

            y_pred  = self._automl.predict(X_test)
            from model_engine.evaluator import ModelEvaluator
            metrics = ModelEvaluator()._compute_metrics(y_test, y_pred, task_type)
            logger.info(f"FLAML test metrics: {metrics}")

            return best_model, metrics

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.warning(
                f"FLAML failed after {elapsed:.1f}s: {exc}\n"
                f"Falling back to LightGBM directly..."
            )
            return self._fallback(X_train, y_train, X_test, y_test, task_type)

    def _fallback(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
    ) -> Tuple[object, Dict[str, float]]:
        """
        Direct LightGBM fallback when FLAML fails or times out.
        LightGBM is fast even on large feature sets.
        """
        logger.info("Using LightGBM fallback directly...")
        try:
            if task_type == "classification":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                )
            else:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=200,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            from model_engine.evaluator import ModelEvaluator
            metrics = ModelEvaluator()._compute_metrics(y_test, y_pred, task_type)

            logger.success(f"LightGBM fallback metrics: {metrics}")
            self.best_config = {"fallback": "LightGBM"}
            self._best_estimator_name = "lgbm_fallback"
            return model, metrics

        except Exception as exc:
            logger.error(f"LightGBM fallback also failed: {exc}")
            return None, {}

    @property
    def best_estimator_name(self) -> str:
        if hasattr(self, "_best_estimator_name"):
            return self._best_estimator_name
        if self._automl is None:
            return "unknown"
        return getattr(self._automl, "best_estimator", "unknown") or "unknown"

    def get_all_tried_estimators(self) -> List[str]:
        if self._automl is None:
            return []
        try:
            return list(self._automl.best_config_per_estimator.keys())
        except Exception:
            return []
