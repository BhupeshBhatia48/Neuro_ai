"""
config/constants.py
--------------------
Static, never-changing constants shared across the entire system.
"""

# -- Task Types ----------------------------------------------------------------
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION     = "regression"
TASK_TIME_SERIES    = "time_series"
TASK_NLP            = "nlp"
TASK_COMPUTER_VISION = "computer_vision"
TASK_CLUSTERING     = "clustering"

SUPPORTED_TASKS = [
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASK_TIME_SERIES,
    TASK_NLP,
    TASK_COMPUTER_VISION,
    TASK_CLUSTERING,
]

# -- Dataset Formats -----------------------------------------------------------
SUPPORTED_FORMATS = [".csv", ".json", ".xlsx", ".xls", ".parquet"]

# -- Dataset Size Thresholds (rows) --------------------------------------------
SMALL_DATASET_THRESHOLD  = 50_000
MEDIUM_DATASET_THRESHOLD = 500_000

# -- Model Recommendation Rules -----------------------------------------------
# Maps task -> candidate model names (ordered by preference)
MODEL_CANDIDATES = {
    TASK_CLASSIFICATION: {
        "small":  ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression",
                   "CatBoost", "ExtraTrees", "SVM"],
        "medium": ["LightGBM", "XGBoost", "RandomForest", "CatBoost",
                   "LogisticRegression", "ExtraTrees"],
        "large":  ["LightGBM", "XGBoost", "CatBoost", "RandomForest"],
    },
    TASK_REGRESSION: {
        "small":  ["RandomForest", "XGBoost", "LightGBM", "LinearRegression",
                   "CatBoost", "ExtraTrees", "Ridge"],
        "medium": ["LightGBM", "XGBoost", "RandomForest", "CatBoost", "ExtraTrees"],
        "large":  ["LightGBM", "XGBoost", "CatBoost", "RandomForest"],
    },
    TASK_TIME_SERIES: {
        "small":  ["ARIMA", "Prophet", "LSTM"],
        "medium": ["LSTM", "Prophet"],
        "large":  ["LSTM", "GRU"],
    },
    TASK_NLP: {
        "small":  ["BERT", "DistilBERT"],
        "medium": ["BERT", "RoBERTa"],
        "large":  ["RoBERTa", "GPT"],
    },
    TASK_COMPUTER_VISION: {
        "small":  ["ResNet18", "MobileNet"],
        "medium": ["ResNet50", "EfficientNet"],
        "large":  ["EfficientNetB7", "ViT"],
    },
    TASK_CLUSTERING: {
        "small":  ["KMeans", "DBSCAN", "AgglomerativeClustering"],
        "medium": ["KMeans", "MiniBatchKMeans"],
        "large":  ["MiniBatchKMeans"],
    },
}

# -- Evaluation Metrics --------------------------------------------------------
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
REGRESSION_METRICS     = ["rmse", "mae", "r2"]

# -- Export File Names ---------------------------------------------------------
MODEL_EXPORT_FILENAME        = "model.pkl"
PREPROCESSING_EXPORT_FILENAME = "preprocessing.pkl"
REPORT_FILENAME              = "training_report.txt"

# -- Search --------------------------------------------------------------------
MAX_SEARCH_RESULTS   = 10
DATASET_SEARCH_QUERY_TEMPLATE = "{keywords} dataset CSV filetype:csv site:kaggle.com OR site:github.com OR site:openml.org"
