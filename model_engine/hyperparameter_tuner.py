"""
model_engine/hyperparameter_tuner.py
--------------------------------------
Hyperparameter optimisation using Optuna (TPE Bayesian sampler).

Why Optuna over RandomizedSearchCV?
  - TPE sampler learns from previous trials -> finds good configs faster.
  - Native pruning: stops unpromising trials early (MedianPruner).
  - Clean study/trial API — easy to inspect, visualise, and resume.
  - Works with any model, not just sklearn estimators.
"""

from __future__ import annotations

import pickle
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from loguru import logger

# Silence Optuna's default INFO spam — we log ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


# -- Search spaces: model_name -> callable(trial) -> param dict -----------------
def _rf_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":     trial.suggest_int("max_depth", 3, 12),
        "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":    trial.suggest_int("num_leaves", 20, 300),
        "max_depth":     trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations":    trial.suggest_int("iterations", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":         trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
    }


def _lr_space(trial: optuna.Trial) -> dict:
    return {
        "C":       trial.suggest_float("C", 1e-4, 1e2, log=True),
        "solver":  trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": 2000,
    }


def _ridge_space(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True),
    }


SEARCH_SPACES: dict[str, Any] = {
    "RandomForest":       _rf_space,
    "XGBoost":            _xgb_space,
    "LightGBM":           _lgbm_space,
    "CatBoost":           _catboost_space,
    "LogisticRegression": _lr_space,
    "Ridge":              _ridge_space,
}


# -- Tuner ---------------------------------------------------------------------
class HyperparameterTuner:
    """
    Uses Optuna TPE to search hyperparameters for the best model.

    Usage
    -----
    tuner = HyperparameterTuner(n_trials=30, cv=3)
    best_model = tuner.tune(model, "XGBoost", X_train, y_train, "classification")
    tuner.save_study("xgboost_study.pkl")   # optional — resume later
    """

    def __init__(self, n_trials: int = 30, cv: int = 3, timeout: int | None = None):
        """
        Parameters
        ----------
        n_trials : Number of Optuna trials (more = better but slower).
        cv       : Cross-validation folds used to score each trial.
        timeout  : Optional wall-clock limit in seconds across all trials.
        """
        self.n_trials = n_trials
        self.cv       = cv
        self.timeout  = timeout
        self.study: optuna.Study | None = None

    # -- Public API ------------------------------------------------------------

    def tune(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str,
    ):
        """
        Run Optuna optimisation and return the best-fitted estimator.
        Falls back to the original model if the model isn't in SEARCH_SPACES
        or if tuning raises an exception.
        """
        space_fn = SEARCH_SPACES.get(model_name)
        if space_fn is None:
            logger.info(f"No Optuna search space for '{model_name}' — skipping tuning.")
            return model

        scoring = "f1_weighted" if task_type == "classification" else "r2"
        direction = "maximize"

        logger.info(
            f"Optuna tuning '{model_name}'  "
            f"(trials={self.n_trials}, cv={self.cv}, scoring={scoring}) …"
        )

        try:
            self.study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            )

            objective = self._make_objective(model, model_name, X_train, y_train, scoring)
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )

            best_params = self.study.best_params
            best_score  = self.study.best_value
            logger.success(
                f"✓ Best params for '{model_name}': {best_params}  "
                f"cv_{scoring}={best_score:.4f}"
            )

            # Re-fit on full training set with best params
            best_model = self._build_model(model, model_name, best_params)
            best_model.fit(X_train, y_train)
            return best_model

        except Exception as exc:
            logger.warning(f"Optuna tuning failed ({exc}) — returning original model.")
            return model

    def save_study(self, path: str) -> None:
        """Persist the Optuna study to disk for later inspection / resumption."""
        if self.study is None:
            logger.warning("No study to save.")
            return
        with open(path, "wb") as f:
            pickle.dump(self.study, f)
        logger.info(f"Optuna study saved -> {path}")

    @staticmethod
    def load_study(path: str) -> optuna.Study:
        with open(path, "rb") as f:
            return pickle.load(f)

    # -- Private helpers -------------------------------------------------------

    def _make_objective(self, base_model, model_name, X, y, scoring):
        """Returns an Optuna objective closure."""
        def objective(trial: optuna.Trial) -> float:
            space_fn = SEARCH_SPACES[model_name]
            params   = space_fn(trial)
            model    = self._build_model(base_model, model_name, params)
            scores   = cross_val_score(
                model, X, y,
                cv=self.cv,
                scoring=scoring,
                n_jobs=-1,
                error_score=0.0,
            )
            return float(scores.mean())
        return objective

    @staticmethod
    def _build_model(base_model, model_name: str, params: dict):
        """Clone base model and apply new params."""
        # CatBoost doesn't always support sklearn clone — handle separately
        if model_name == "CatBoost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = type(base_model)
            return cls(**{**{"verbose": 0, "random_state": 42}, **params})

        from sklearn.base import clone
        m = clone(base_model)
        m.set_params(**params)
        return m
