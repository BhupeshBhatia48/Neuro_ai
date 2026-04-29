"""
export_engine/pickle_exporter.py
----------------------------------
Serialises the trained model, preprocessing pipeline, and Optuna studies to disk.

Strategy:
  - sklearn / XGBoost / LightGBM / CatBoost pipelines -> joblib
    (handles numpy arrays inside estimators more efficiently than pickle)
  - Optuna Study objects -> pickle
    (joblib can't serialise Optuna internals reliably)
  - Raw Python objects -> pickle fallback
"""

from pathlib import Path
import pickle

import joblib
from loguru import logger

from config.settings import MODELS_DIR
from config.constants import MODEL_EXPORT_FILENAME, PREPROCESSING_EXPORT_FILENAME


class PickleExporter:
    """
    Exports and loads ML artefacts using the right serialiser for each type.
    """

    def export_model(self, model, name: str = "best") -> Path:
        """Export a fitted sklearn-compatible model with joblib."""
        dest = MODELS_DIR / f"{name}_{MODEL_EXPORT_FILENAME}"
        joblib.dump(model, dest, compress=3)
        logger.success(f"Model exported -> {dest}")
        return dest

    def export_pipeline(self, pipeline, name: str = "best") -> Path:
        """Export a fitted sklearn Pipeline with joblib."""
        dest = MODELS_DIR / f"{name}_{PREPROCESSING_EXPORT_FILENAME}"
        joblib.dump(pipeline, dest, compress=3)
        logger.success(f"Pipeline exported -> {dest}")
        return dest

    def export_optuna_study(self, study, name: str = "study") -> Path:
        """Export an Optuna Study with pickle (joblib can't handle it reliably)."""
        dest = MODELS_DIR / f"{name}_optuna_study.pkl"
        with open(dest, "wb") as f:
            pickle.dump(study, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.success(f"Optuna study exported -> {dest}")
        return dest

    def export_object(self, obj, filename: str) -> Path:
        """Generic pickle export for any serialisable object."""
        dest = MODELS_DIR / filename
        with open(dest, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.success(f"Object exported -> {dest}")
        return dest

    # -- Loaders ---------------------------------------------------------------

    @staticmethod
    def load_model(path: Path):
        return joblib.load(path)

    @staticmethod
    def load_pipeline(path: Path):
        return joblib.load(path)

    @staticmethod
    def load_optuna_study(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_object(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
