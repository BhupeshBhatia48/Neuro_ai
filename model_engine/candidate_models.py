"""
model_engine/candidate_models.py
----------------------------------
Central registry that maps model names -> instantiated sklearn-compatible estimators.
All models here support .fit(X, y) / .predict(X).
"""

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from config.settings import DEFAULT_RANDOM_STATE

RS = DEFAULT_RANDOM_STATE

# -- Classification ------------------------------------------------------------
CLASSIFICATION_MODELS: dict = {
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
    "XGBoost":            XGBClassifier(
                              n_estimators=200, random_state=RS,
                              eval_metric="logloss", verbosity=0, n_jobs=-1,
                          ),
    "LightGBM":           LGBMClassifier(n_estimators=200, random_state=RS, verbose=-1, n_jobs=-1),
    "CatBoost":           CatBoostClassifier(iterations=200, random_state=RS, verbose=0),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RS, n_jobs=-1),
    "ExtraTrees":         ExtraTreesClassifier(n_estimators=200, random_state=RS, n_jobs=-1),
    "SVM":                SVC(probability=True, random_state=RS),
}

# -- Regression ----------------------------------------------------------------
REGRESSION_MODELS: dict = {
    "RandomForest":      RandomForestRegressor(n_estimators=200, random_state=RS, n_jobs=-1),
    "XGBoost":           XGBRegressor(n_estimators=200, random_state=RS, verbosity=0, n_jobs=-1),
    "LightGBM":          LGBMRegressor(n_estimators=200, random_state=RS, verbose=-1, n_jobs=-1),
    "CatBoost":          CatBoostRegressor(iterations=200, random_state=RS, verbose=0),
    "LinearRegression":  LinearRegression(n_jobs=-1),
    "Ridge":             Ridge(),
    "ExtraTrees":        ExtraTreesRegressor(n_estimators=200, random_state=RS, n_jobs=-1),
    "SVR":               SVR(),
}


def get_model(name: str, task_type: str):
    """
    Returns a fresh (unfitted) estimator instance by name.
    Raises KeyError if not found.
    """
    registry = (
        CLASSIFICATION_MODELS if task_type == "classification"
        else REGRESSION_MODELS
    )

    if name not in registry:
        raise KeyError(
            f"Model '{name}' not found for task '{task_type}'. "
            f"Available: {list(registry.keys())}"
        )

    # Return a clone so the registry stays pristine
    from sklearn.base import clone
    try:
        return clone(registry[name])
    except Exception:
        return registry[name]   # CatBoost doesn't always support clone


def available_models(task_type: str) -> list:
    if task_type == "classification":
        return list(CLASSIFICATION_MODELS.keys())
    return list(REGRESSION_MODELS.keys())
