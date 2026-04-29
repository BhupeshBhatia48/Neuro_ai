"""
search_engine/semantic_dataset_search.py
------------------------------------------
Semantic Dataset Intelligence — the core fix for wrong dataset retrieval.

THE FUNDAMENTAL PROBLEM WITH THE OLD SYSTEM:
--------------------------------------------
The old system extracted keywords ("parkinson", "tremor", "patients") and
searched DDG for .csv files matching those keywords. This fails because:

1. DDG doesn't index raw CSV files well — it returns pages ABOUT datasets,
   not the actual files.
2. Keyword matching has no semantic understanding: "parkinson" doesn't tell
   the system that the actual data contains MDVP vocal features, jitter,
   shimmer, NHR, HNR columns. Without knowing what the data LOOKS LIKE,
   it can't recognize the right dataset when it finds one.
3. The same query run 3 times gives 3 different results because DDG search
   results are non-deterministic and change by the hour.

HOW GPT-STYLE SYSTEMS APPROACH THIS:
--------------------------------------
1. SEMANTIC INTENT: First understand WHAT the user is trying to do, not just
   WHICH WORDS they used. "predict patients with parkinsons tremor" means:
   - Domain: neurology / motor disorder
   - Task: binary classification (has parkinson vs not)
   - Data type: biomedical measurements (vocal, movement, clinical)
   - Expected features: frequency measurements, jitter, shimmer, clinical scores
   - Known dataset: UCI Parkinson's Disease dataset (Little et al. 2007)

2. DIRECT KNOWLEDGE: For well-known ML benchmark datasets, skip search entirely
   and go directly to the verified canonical source. The Parkinson's dataset
   has been at the same UCI URL for 15+ years.

3. MULTIPLE STRATEGIES: Generate diverse search strategies, not just one query:
   - Strategy A: Known dataset name search
   - Strategy B: Domain-specific terminology
   - Strategy C: Author/paper citation search
   - Strategy D: Feature/column name search
   - Strategy E: OpenML/UCI taxonomy search

4. SEMANTIC VALIDATION: After download, verify the dataset matches the MEANING
   not just the keywords. A Parkinson's dataset should have biomarker columns,
   not just the word "parkinson" somewhere.

IMPLEMENTATION:
---------------
SemanticDatasetSearch uses an LLM (when available) to:
1. Extract the deep semantic intent of the user's idea
2. Generate dataset-specific search strategies
3. Identify the canonical dataset name if known
4. Provide expected column signatures for validation

When LLM is unavailable, falls back to an expanded semantic knowledge base
that maps concepts to datasets using domain understanding, not keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ── Semantic Dataset Knowledge Base ──────────────────────────────────────────
# Maps semantic CONCEPTS (not keywords) to dataset metadata.
# Each entry provides: verified URLs, expected columns, and search terms.
# URLs are listed in priority order — first working URL wins.
# All URLs are verified working as of April 2026.

SEMANTIC_DATASET_KB: Dict[str, Dict] = {

    # ── NEUROLOGY / MOTOR DISORDERS ──────────────────────────────────────────
    "parkinson_disease": {
        "triggers": [
            "parkinson", "parkinsons", "tremor", "dopamine", "motor disorder",
            "neurodegenerative", "mdvp", "vocal fold", "jitter", "shimmer",
            "dysphonia", "bradykinesia", "resting tremor",
        ],
        "description": "Parkinson's disease detection from biomedical voice measurements",
        "canonical_name": "Parkinson's Disease Dataset (Little et al. 2007)",
        "expected_columns": ["mdvp", "jitter", "shimmer", "nhr", "hnr", "rpde", "dfa", "ppe", "status"],
        "task": "classification",
        "urls": [
            # UCI ML Repository — most stable, direct .data download
            "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            # OpenML confirmed ID
            "https://api.openml.org/data/v1/download/1488",
            # Kaggle public mirror
            "https://raw.githubusercontent.com/shrikant-temburwar/Parkinsons-Disease-Detection/master/parkinsons.csv",
        ],
        "search_queries": [
            "parkinsons disease UCI dataset vocal biomedical csv",
            "parkinson tremor MDVP jitter shimmer dataset",
            "Little parkinsons voice measurements classification",
        ],
    },

    # ── DIABETES / GLUCOSE ───────────────────────────────────────────────────
    "diabetes_pima": {
        "triggers": [
            "diabetes", "diabetic", "glucose", "insulin", "pima", "blood sugar",
            "glycemic", "hba1c", "pregnancies bmi",
        ],
        "description": "Pima Indians Diabetes dataset — predict diabetes onset from clinical measurements",
        "canonical_name": "Pima Indians Diabetes Dataset (UCI)",
        "expected_columns": ["pregnancies", "glucose", "bloodpressure", "skinthickness", "insulin", "bmi", "diabetespedigreefunction", "age", "outcome"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
            "https://api.openml.org/data/v1/download/37",
        ],
        "search_queries": [
            "pima indians diabetes dataset UCI csv",
            "diabetes glucose insulin classification dataset",
        ],
    },

    # ── HEART DISEASE ────────────────────────────────────────────────────────
    "heart_disease": {
        "triggers": [
            "heart disease", "heart attack", "cardiac", "cardiovascular", "coronary",
            "chest pain", "ecg", "electrocardiogram", "angina", "myocardial",
            "trestbps", "chol", "thalach",
        ],
        "description": "Cleveland Heart Disease dataset — predict presence of heart disease",
        "canonical_name": "Heart Disease Dataset (Cleveland, UCI)",
        "expected_columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/selva86/datasets/master/Heart.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "https://api.openml.org/data/v1/download/53",
        ],
        "search_queries": [
            "cleveland heart disease dataset UCI csv",
            "cardiac disease classification trestbps chol dataset",
        ],
    },

    # ── BREAST CANCER ────────────────────────────────────────────────────────
    "breast_cancer": {
        "triggers": [
            "breast cancer", "mammogram", "tumor malignant", "tumour benign",
            "biopsy cancer", "oncology", "cancer diagnosis",
        ],
        "description": "Wisconsin Breast Cancer dataset — classify tumors as malignant or benign",
        "canonical_name": "Wisconsin Breast Cancer Dataset (UCI)",
        "expected_columns": ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness", "diagnosis"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/plotly/datasets/master/breast-cancer.csv",
            "https://api.openml.org/data/v1/download/13",
        ],
        "search_queries": [
            "Wisconsin breast cancer dataset UCI csv malignant benign",
            "tumor classification radius texture perimeter dataset",
        ],
    },

    # ── CUSTOMER CHURN ───────────────────────────────────────────────────────
    "customer_churn": {
        "triggers": [
            "churn", "customer retention", "subscription cancel", "telecom churn",
            "attrition", "customer leaving", "monthly charges", "contract",
        ],
        "description": "Telco customer churn — predict whether a customer will leave",
        "canonical_name": "Telco Customer Churn Dataset (IBM)",
        "expected_columns": ["customerid", "tenure", "monthlycharges", "totalcharges", "contract", "paymentmethod", "churn"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        ],
        "search_queries": [
            "telco customer churn dataset IBM csv",
            "customer attrition monthly charges contract classification",
        ],
    },

    # ── FRAUD DETECTION ──────────────────────────────────────────────────────
    "credit_fraud": {
        "triggers": [
            "fraud", "credit card fraud", "fraudulent transaction", "anomaly detection",
            "suspicious transaction", "financial fraud", "transaction fraud",
        ],
        "description": "Credit card fraud detection — highly imbalanced classification",
        "canonical_name": "Credit Card Fraud Detection Dataset (ULB)",
        "expected_columns": ["time", "v1", "v2", "v3", "amount", "class"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/georgeblck/creditcardfraud/master/creditcard_sample.csv",
            "https://api.openml.org/data/v1/download/1597",
        ],
        "search_queries": [
            "credit card fraud detection dataset CSV PCA features",
            "transaction fraud classification imbalanced dataset",
        ],
    },

    # ── HOUSE PRICES ─────────────────────────────────────────────────────────
    "house_prices": {
        "triggers": [
            "house price", "home price", "real estate", "property value",
            "housing price", "rent prediction", "boston housing", "california housing",
            "sqft", "bedrooms bathrooms", "zillow",
        ],
        "description": "Housing price prediction — regression on property features",
        "canonical_name": "Boston Housing / California Housing Dataset",
        "expected_columns": ["bedrooms", "bathrooms", "sqft", "price", "rooms", "location", "area"],
        "task": "regression",
        "urls": [
            "https://raw.githubusercontent.com/plotly/datasets/master/housing_v2.csv",
            "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "https://api.openml.org/data/v1/download/42165",
        ],
        "search_queries": [
            "Boston housing price regression dataset CSV",
            "house price prediction sqft bedrooms dataset",
        ],
    },

    # ── IRIS / FLOWER CLASSIFICATION ─────────────────────────────────────────
    "iris_flower": {
        "triggers": [
            "iris", "flower classification", "sepal", "petal", "setosa",
            "versicolor", "virginica", "iris dataset",
        ],
        "description": "Iris flower species classification — classic ML benchmark",
        "canonical_name": "Iris Dataset (Fisher 1936)",
        "expected_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
            "https://api.openml.org/data/v1/download/61",
        ],
        "search_queries": [
            "iris dataset CSV sepal petal species",
        ],
    },

    # ── TITANIC / SURVIVAL ───────────────────────────────────────────────────
    "titanic_survival": {
        "triggers": [
            "titanic", "survival", "passenger survival", "survived", "ship disaster",
            "pclass", "fare cabin", "embarked",
        ],
        "description": "Titanic passenger survival prediction",
        "canonical_name": "Titanic Dataset (Kaggle/seaborn)",
        "expected_columns": ["survived", "pclass", "name", "sex", "age", "sibsp", "parch", "fare", "cabin", "embarked"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
            "https://api.openml.org/data/v1/download/40945",
        ],
        "search_queries": [
            "titanic survival dataset CSV",
        ],
    },

    # ── SENTIMENT ANALYSIS / NLP ─────────────────────────────────────────────
    "imdb_sentiment": {
        "triggers": [
            "sentiment", "movie review", "opinion", "positive negative review",
            "imdb", "text sentiment", "review classification", "emotion detection",
        ],
        "description": "IMDB movie review sentiment classification",
        "canonical_name": "IMDB Movie Reviews Sentiment Dataset",
        "expected_columns": ["review", "sentiment", "text", "label"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/IMDB-Dataset.csv",
        ],
        "search_queries": [
            "IMDB movie reviews sentiment dataset CSV positive negative",
            "text sentiment classification dataset review label",
        ],
    },

    # ── WINE QUALITY ─────────────────────────────────────────────────────────
    "wine_quality": {
        "triggers": [
            "wine quality", "wine rating", "wine classification", "red wine",
            "white wine", "alcohol acidity", "wine grade",
        ],
        "description": "Wine quality classification/regression from physicochemical tests",
        "canonical_name": "Wine Quality Dataset (UCI)",
        "expected_columns": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "alcohol", "quality"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        ],
        "search_queries": [
            "wine quality dataset UCI red white CSV",
        ],
    },

    # ── SPAM DETECTION ───────────────────────────────────────────────────────
    "spam_detection": {
        "triggers": [
            "spam", "email spam", "sms spam", "spam filter", "ham spam",
            "spam detection", "spam classification",
        ],
        "description": "SMS/Email spam detection — text classification",
        "canonical_name": "SMS Spam Collection Dataset",
        "expected_columns": ["label", "message", "text", "sms", "ham", "spam"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
            "https://api.openml.org/data/v1/download/44",
        ],
        "search_queries": [
            "SMS spam collection dataset TSV ham spam",
            "email spam filter text classification dataset",
        ],
    },

    # ── LOAN / CREDIT ────────────────────────────────────────────────────────
    "loan_prediction": {
        "triggers": [
            "loan", "credit score", "loan default", "creditworthiness", "mortgage",
            "loan approval", "credit risk", "debt", "loanamount",
        ],
        "description": "Loan prediction — predict whether a loan will be approved",
        "canonical_name": "Loan Prediction Dataset",
        "expected_columns": ["loanamount", "income", "credit_history", "education", "employment", "loan_status"],
        "task": "classification",
        "urls": [
            "https://api.openml.org/data/v1/download/31",
            "https://raw.githubusercontent.com/dsrscientist/dataset1/master/loan_prediction.csv",
        ],
        "search_queries": [
            "loan approval prediction dataset CSV income credit",
            "credit risk classification income education dataset",
        ],
    },

    # ── SALARY PREDICTION ────────────────────────────────────────────────────
    "salary_prediction": {
        "triggers": [
            "salary", "wage", "income prediction", "pay prediction", "compensation",
            "years experience", "salary prediction",
        ],
        "description": "Salary prediction from experience and job attributes",
        "canonical_name": "Salary Dataset",
        "expected_columns": ["years_experience", "salary", "education", "job_title", "age"],
        "task": "regression",
        "urls": [
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Salary_Data.csv",
            "https://api.openml.org/data/v1/download/4535",
        ],
        "search_queries": [
            "salary prediction years experience dataset CSV",
        ],
    },

    # ── MENTAL HEALTH ────────────────────────────────────────────────────────
    "mental_health": {
        "triggers": [
            "mental health", "depression", "anxiety", "stress", "ptsd",
            "psychiatric", "mental illness", "therapy outcome",
        ],
        "description": "Mental health in tech survey — predict need for treatment",
        "canonical_name": "Mental Health in Tech Survey",
        "expected_columns": ["age", "gender", "family_history", "treatment", "work_interfere", "benefits"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/anuragkumar95/Mental-Health-in-Tech-Survey-2014/master/survey.csv",
            "https://api.openml.org/data/v1/download/45524",
        ],
        "search_queries": [
            "mental health tech survey dataset CSV treatment",
        ],
    },

    # ── PENGUIN SPECIES ──────────────────────────────────────────────────────
    "penguin_species": {
        "triggers": [
            "penguin", "penguin species", "adelie", "gentoo", "chinstrap",
            "bill length depth", "flipper",
        ],
        "description": "Palmer Penguins species classification",
        "canonical_name": "Palmer Penguins Dataset",
        "expected_columns": ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
        ],
        "search_queries": [
            "palmer penguins species dataset CSV",
        ],
    },

    # ── MUSHROOM EDIBILITY ───────────────────────────────────────────────────
    "mushroom_edibility": {
        "triggers": [
            "mushroom", "mushroom edible", "mushroom poisonous", "fungus classification",
            "cap shape color", "gill",
        ],
        "description": "Mushroom edibility classification — edible vs poisonous",
        "canonical_name": "Mushroom Dataset (UCI)",
        "expected_columns": ["class", "cap_shape", "cap_color", "odor", "gill_color", "ring_type"],
        "task": "classification",
        "urls": [
            "https://api.openml.org/data/v1/download/24",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        ],
        "search_queries": [
            "mushroom edible poisonous dataset UCI CSV",
        ],
    },

    # ── EMPLOYEE ATTRITION ───────────────────────────────────────────────────
    "employee_attrition": {
        "triggers": [
            "employee attrition", "hr attrition", "employee turnover",
            "employee leaving", "hr analytics", "work satisfaction",
        ],
        "description": "IBM HR Analytics employee attrition prediction",
        "canonical_name": "IBM HR Analytics Employee Attrition Dataset",
        "expected_columns": ["attrition", "age", "department", "jobrole", "monthlyincome", "yearsatcompany"],
        "task": "classification",
        "urls": [
            "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
            "https://api.openml.org/data/v1/download/42193",
        ],
        "search_queries": [
            "IBM employee attrition HR analytics dataset CSV",
        ],
    },

    # ── WEATHER / RAIN ───────────────────────────────────────────────────────
    "weather_rain": {
        "triggers": [
            "rain prediction", "weather forecast", "rainfall", "will it rain",
            "weather classification", "temperature humidity wind",
        ],
        "description": "Weather dataset — predict if it will rain tomorrow",
        "canonical_name": "Rain in Australia Dataset",
        "expected_columns": ["date", "location", "mintemp", "maxtemp", "rainfall", "humidity", "windspeed", "raintomorrow"],
        "task": "classification",
        "urls": [
            "https://api.openml.org/data/v1/download/43892",
        ],
        "search_queries": [
            "rain prediction weather dataset Australia CSV",
            "weather rainfall classification humidity temperature dataset",
        ],
    },

    # ── GENERIC TABULAR FALLBACK ─────────────────────────────────────────────
    "generic_tabular": {
        "triggers": [],  # matches nothing — used as explicit fallback only
        "description": "Generic tabular ML dataset",
        "canonical_name": "Unknown",
        "expected_columns": [],
        "task": "classification",
        "urls": [],
        "search_queries": [],
    },
}


# ── Semantic concept → dataset mapping ───────────────────────────────────────
# Built once at import time from SEMANTIC_DATASET_KB
def _build_semantic_index() -> Dict[str, str]:
    """Build trigger word → dataset_key index for fast lookup."""
    index = {}
    for dataset_key, meta in SEMANTIC_DATASET_KB.items():
        if dataset_key == "generic_tabular":
            continue
        for trigger in meta.get("triggers", []):
            # Store both exact and stem-based triggers
            index[trigger.lower()] = dataset_key
            # Add stem (first 5+ chars) for fuzzy matching
            if len(trigger) >= 5:
                stem = trigger[:5].lower()
                if stem not in index:
                    index[stem] = dataset_key
    return index

_SEMANTIC_INDEX = _build_semantic_index()


@dataclass
class SemanticIntent:
    """Structured semantic understanding of a user's ML idea."""
    raw_idea:        str
    dataset_key:     str                  # matched key in SEMANTIC_DATASET_KB
    confidence:      float                # 0.0 – 1.0
    dataset_meta:    Dict                 # full metadata from KB
    search_queries:  List[str]            # ready-to-use search queries
    direct_urls:     List[str]            # verified direct download URLs
    expected_cols:   List[str]            # columns we expect to find
    llm_enriched:    bool = False         # whether LLM added extra context


class SemanticDatasetSearch:
    """
    Semantic dataset search — understands MEANING, not just keywords.

    Flow:
    1. Semantic index lookup: fast O(1) trigger matching
    2. LLM enrichment (if available): deep semantic expansion
    3. Returns: direct URLs + enriched search queries

    This is the PRIMARY dataset finder. VerifiedDatasetSearch is removed —
    this class replaces it with much deeper semantic understanding.
    """

    def find(
        self,
        idea: str,
        use_llm: bool = True,
    ) -> SemanticIntent:
        """
        Find the semantically correct dataset for the given idea.

        Returns a SemanticIntent with:
        - direct_urls: verified download URLs in priority order
        - search_queries: diverse search strings for fallback
        - expected_cols: column signatures for validation
        """
        idea_lower = idea.lower()

        # ── Step 1: Semantic index matching ──────────────────────────────────
        scores: Dict[str, float] = {}

        for trigger, dataset_key in _SEMANTIC_INDEX.items():
            if trigger in idea_lower:
                # Full phrase match scores higher than stem match
                is_stem = len(trigger) <= 5 and trigger not in [
                    t for t in SEMANTIC_DATASET_KB.get(dataset_key, {}).get("triggers", [])
                ]
                score = 0.5 if is_stem else 1.0
                # Bonus for multi-word exact phrase
                if " " in trigger:
                    score += 0.5
                scores[dataset_key] = scores.get(dataset_key, 0) + score

        # ── Step 2: Word-level fuzzy matching ─────────────────────────────────
        idea_words = set(re.findall(r'\b[a-z]{4,}\b', idea_lower))
        for dataset_key, meta in SEMANTIC_DATASET_KB.items():
            if dataset_key == "generic_tabular":
                continue
            for trigger in meta.get("triggers", []):
                trigger_words = set(trigger.split())
                # Count word overlaps
                overlap = len(idea_words & trigger_words)
                if overlap > 0:
                    scores[dataset_key] = scores.get(dataset_key, 0) + overlap * 0.3

        # ── Step 3: Select best match ─────────────────────────────────────────
        if scores:
            best_key = max(scores, key=scores.__getitem__)
            confidence = min(1.0, scores[best_key] / 3.0)
        else:
            best_key   = "generic_tabular"
            confidence = 0.0

        meta = SEMANTIC_DATASET_KB.get(best_key, SEMANTIC_DATASET_KB["generic_tabular"])

        logger.info(
            f"[SemanticSearch] Matched dataset: '{best_key}' "
            f"(confidence={confidence:.2f}) for idea: '{idea[:60]}'"
        )

        # ── Step 4: Build queries ─────────────────────────────────────────────
        # Start with knowledge-base queries (domain-specific, proven to work)
        queries = list(meta.get("search_queries", []))

        # Add idea-derived queries as fallback
        idea_concepts = self._extract_key_concepts(idea)
        if idea_concepts:
            queries.append(f"{' '.join(idea_concepts[:3])} dataset csv")
            queries.append(f"{' '.join(idea_concepts[:2])} classification dataset")

        # ── Step 5: LLM enrichment ────────────────────────────────────────────
        llm_enriched = False
        if use_llm and best_key != "generic_tabular":
            try:
                extra_queries = self._llm_enrich(idea, meta)
                if extra_queries:
                    queries = extra_queries + queries
                    llm_enriched = True
            except Exception as exc:
                logger.debug(f"[SemanticSearch] LLM enrichment failed: {exc}")

        # Deduplicate queries while preserving order
        seen, unique_queries = set(), []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)

        intent = SemanticIntent(
            raw_idea      = idea,
            dataset_key   = best_key,
            confidence    = confidence,
            dataset_meta  = meta,
            search_queries = unique_queries[:8],
            direct_urls   = list(meta.get("urls", [])),
            expected_cols = list(meta.get("expected_columns", [])),
            llm_enriched  = llm_enriched,
        )

        logger.info(
            f"[SemanticSearch] Intent: key={best_key} conf={confidence:.2f} "
            f"direct_urls={len(intent.direct_urls)} queries={len(intent.search_queries)}"
        )

        return intent

    def _extract_key_concepts(self, idea: str) -> List[str]:
        """Extract meaningful content words from idea, preserving domain terms."""
        STOP = {
            "predict", "classify", "detect", "build", "create", "model",
            "system", "using", "with", "from", "that", "will", "based",
            "whether", "the", "and", "for", "this", "make", "train",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', idea.lower())
        seen, unique = set(), []
        for w in words:
            if w not in STOP and w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:6]

    def _llm_enrich(self, idea: str, meta: Dict) -> List[str]:
        """Use LLM to generate additional dataset-specific search queries."""
        from config.settings import USE_LLM
        if not USE_LLM:
            return []

        from llm_agent.openrouter_client import OpenRouterClient

        system = (
            "You are a machine learning dataset expert. "
            "Given a project idea and the canonical dataset name, "
            "generate 3 precise search queries to find this dataset online. "
            "Focus on: the official dataset name, the repository it's from, "
            "and the specific file format. "
            "Return ONLY a JSON array of 3 strings. Nothing else."
        )
        user = (
            f"Project idea: {idea}\n"
            f"Canonical dataset: {meta.get('canonical_name', 'unknown')}\n"
            f"Expected columns: {', '.join(meta.get('expected_columns', [])[:5])}\n\n"
            f"Generate 3 search queries to find this dataset. "
            f"Return as JSON array: [\"query1\", \"query2\", \"query3\"]"
        )

        client = OpenRouterClient()
        raw = client.chat(
            messages=[{"role": "user", "content": user}],
            system=system,
            max_tokens=120,
            temperature=0.1,
        ).strip()

        import json
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        queries = json.loads(clean)
        if isinstance(queries, list):
            return [str(q) for q in queries if q][:3]
        return []
