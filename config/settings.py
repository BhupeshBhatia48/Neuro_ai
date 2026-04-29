"""
config/settings.py
------------------
Runtime settings loaded from environment variables (.env file).

LLM Backend : OpenRouter  (https://openrouter.ai)
              Default model: openrouter/auto  (free, auto-routed best available model)
              LLM is OPTIONAL — set USE_LLM=false in .env to run fully offline.

Issues resolved
---------------
#1  : Default LLM model -> "openrouter/auto" (free, always available)
#6  : AUTOML_MODE setting — "auto" uses FLAML, "manual" uses ModelTrainer
#8  : Primary AutoML = FLAML only. PyCaret removed.
#9  : LLM optional — USE_LLM=false enables fully offline rule-based mode
#11 : Global reproducibility — RANDOM_SEED applied everywhere
#5  : MAX_DATASET_ROWS — sampling policy at 300k rows
#7  : Download security constants
"""

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# -- Load .env from project root -----------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# -- OpenRouter / LLM ----------------------------------------------------------
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", "")

# Issue #1: Default = openrouter/auto — free, routes to best available model
# Override in .env: OPENROUTER_MODEL=openai/gpt-4o  or  anthropic/claude-3.5-sonnet
LLM_MODEL: str      = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

OPENROUTER_BASE_URL: str  = "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL: str  = os.getenv("OPENROUTER_SITE_URL",  "https://neuro-ai.local")
OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "Neuro AI")

# Issue #9: LLM is fully optional — set USE_LLM=false to go fully offline
USE_LLM: bool = os.getenv("USE_LLM", "true").lower() not in ("false", "0", "no")

# -- Kaggle (optional) ---------------------------------------------------------
KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY: str      = os.getenv("KAGGLE_KEY", "")

# -- Paths ---------------------------------------------------------------------
DATASETS_DIR: Path = BASE_DIR / "datasets"
MODELS_DIR: Path   = BASE_DIR / "models"
LOGS_DIR: Path     = BASE_DIR / "logs"

for _dir in (DATASETS_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# -- API Server ----------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# -- Training ------------------------------------------------------------------
DEFAULT_TEST_SIZE: float       = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_RANDOM_STATE: int      = int(os.getenv("DEFAULT_RANDOM_STATE", "42"))
MAX_TRAINING_TIME_SECONDS: int = int(os.getenv("MAX_TRAINING_TIME_SECONDS", "300"))

# Issue #11: Global reproducibility seed applied to random, numpy, sklearn, torch
RANDOM_SEED: int = DEFAULT_RANDOM_STATE
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Issue #5: Dataset sampling — sample when rows exceed this threshold
MAX_DATASET_ROWS: int = int(os.getenv("MAX_DATASET_ROWS", str(300_000)))

# Issue #6: AutoML execution mode
# "auto"   -> FLAML AutoMLController handles everything
# "manual" -> ModelTrainer trains specific models from candidate_models.py
AUTOML_MODE: str = os.getenv("AUTOML_MODE", "auto").lower()  # "auto" | "manual"
FLAML_TIME_BUDGET: int = int(os.getenv("FLAML_TIME_BUDGET", "120"))  # seconds

# Issue #7: Download security constants
MAX_DOWNLOAD_SIZE_BYTES: int = int(
    os.getenv("MAX_DOWNLOAD_SIZE_BYTES", str(500 * 1024 * 1024))
)
ALLOWED_DOWNLOAD_DOMAINS: list = [
    "kaggle.com", "raw.githubusercontent.com", "github.com",
    "openml.org", "archive.ics.uci.edu", "data.world",
    "huggingface.co", "drive.google.com",
    "storage.googleapis.com", "s3.amazonaws.com",
]
ALLOWED_DOWNLOAD_EXTENSIONS: tuple = (
    ".csv", ".json", ".xlsx", ".xls", ".parquet", ".zip", ".tsv", ".data", ".arff"
)
