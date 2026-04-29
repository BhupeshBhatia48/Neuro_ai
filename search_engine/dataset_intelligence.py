"""
search_engine/dataset_intelligence.py
---------------------------------------
Smart dataset understanding layer.

Instead of keyword matching against a fixed list, this module:
1. Understands the problem deeply (task, domain, data type, use case)
2. Generates smart targeted search queries
3. Evaluates downloaded datasets for column relevance
4. Scores quality before accepting

This makes Neuro AI think like a data scientist, not a search engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# ── Domain knowledge: what columns indicate a relevant dataset ────────────────
# Format: domain_key -> list of column name patterns that MUST be present
# If at least MIN_MATCH of these patterns appear in the dataset columns,
# the dataset is considered relevant.

DOMAIN_COLUMN_SIGNATURES: Dict[str, Dict] = {

    # ── NLP / Sentiment ───────────────────────────────────────────────────────
    "sentiment": {
        "must_have_any": ["text", "review", "comment", "tweet", "message",
                          "content", "sentence", "description", "post", "body"],
        "target_patterns": ["sentiment", "label", "class", "rating",
                             "positive", "negative", "score"],
        "min_text_col":  1,
        "description":   "NLP sentiment dataset needs a text column",
    },
    "nlp": {
        "must_have_any": ["text", "review", "comment", "tweet", "message",
                          "content", "sentence", "description", "post"],
        "target_patterns": ["label", "class", "category", "sentiment", "tag"],
        "min_text_col":  1,
        "description":   "NLP dataset needs a text column",
    },
    "text_classification": {
        "must_have_any": ["text", "review", "content", "message", "tweet"],
        "target_patterns": ["label", "class", "category", "spam", "ham"],
        "min_text_col":  1,
        "description":   "Text classification needs text + label columns",
    },

    # ── Mental Health ─────────────────────────────────────────────────────────
    "mental_health": {
        "must_have_any": ["mental", "depression", "anxiety", "stress",
                          "treatment", "disorder", "therapy", "psychiatric",
                          "symptom", "diagnosis", "age", "gender",
                          "family_history", "work_interfere", "benefits",
                          "statement", "status"],
        "target_patterns": ["depression", "anxiety", "treatment", "diagnosis",
                             "label", "class", "status", "condition"],
        "min_match":     1,
        "description":   "Mental health dataset needs clinical or survey columns",
    },

    # ── Medical / Clinical ────────────────────────────────────────────────────
    "diabetes": {
        "must_have_any": ["glucose", "insulin", "bmi", "bloodpressure",
                          "blood_pressure", "pregnancies", "age",
                          "skinthickness", "diabetespedigree"],
        "target_patterns": ["outcome", "diabetes", "class", "label"],
        "min_match":     3,
        "description":   "Diabetes dataset needs glucose/insulin/BMI columns",
    },
    "heart_disease": {
        "must_have_any": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                          "restecg", "thalach", "exang", "oldpeak",
                          "cholesterol", "blood_pressure", "heart_rate"],
        "target_patterns": ["target", "output", "heart", "disease", "label"],
        "min_match":     3,
        "description":   "Heart disease dataset needs clinical vitals columns",
    },
    "cancer": {
        "must_have_any": ["radius", "texture", "perimeter", "area",
                          "smoothness", "concavity", "symmetry",
                          "tumor", "malignant", "benign", "mass"],
        "target_patterns": ["diagnosis", "label", "class", "target"],
        "min_match":     2,
        "description":   "Cancer dataset needs tumour measurement columns",
    },
    "parkinson": {
        "must_have_any": ["mdvp", "jitter", "shimmer", "nhr", "hnr",
                          "rpde", "dfa", "spread", "d2", "ppe",
                          "vocal", "tremor", "frequency"],
        "target_patterns": ["status", "label", "class", "diagnosis"],
        "min_match":     2,
        "description":   "Parkinson's dataset needs vocal/tremor measurement columns",
    },
    "medical": {
        "must_have_any": ["age", "gender", "symptoms", "diagnosis",
                          "treatment", "patient", "hospital", "doctor",
                          "blood", "pressure", "weight", "height"],
        "target_patterns": ["diagnosis", "outcome", "label", "disease", "status"],
        "min_match":     2,
        "description":   "Medical dataset needs patient clinical columns",
    },

    # ── Finance ───────────────────────────────────────────────────────────────
    "churn": {
        "must_have_any": ["customerid", "tenure", "monthlycharges",
                          "totalcharges", "contract", "paymentmethod",
                          "customer", "subscription", "account"],
        "target_patterns": ["churn", "exited", "label", "target", "left"],
        "min_match":     2,
        "description":   "Churn dataset needs customer subscription columns",
    },
    "fraud": {
        "must_have_any": ["amount", "v1", "v2", "time", "transaction",
                          "merchant", "category", "balance"],
        "target_patterns": ["class", "fraud", "is_fraud", "label", "target"],
        "min_match":     2,
        "description":   "Fraud dataset needs transaction/amount columns",
    },
    "loan": {
        "must_have_any": ["income", "loan", "credit", "employment",
                          "property", "coapplicant", "loanamount",
                          "education", "dependents", "self_employed"],
        "target_patterns": ["loan_status", "status", "approved", "label"],
        "min_match":     2,
        "description":   "Loan dataset needs income/credit/employment columns",
    },

    # ── Prices / Regression ───────────────────────────────────────────────────
    "house_price": {
        "must_have_any": ["bedrooms", "bathrooms", "sqft", "sqfeet",
                          "rooms", "floors", "area", "location",
                          "zipcode", "price", "lat", "long"],
        "target_patterns": ["price", "saleprice", "value", "cost"],
        "min_match":     2,
        "description":   "House price dataset needs property feature columns",
    },
    "salary": {
        "must_have_any": ["experience", "education", "age", "job",
                          "role", "department", "company", "skill",
                          "years_experience", "senior"],
        "target_patterns": ["salary", "wage", "income", "compensation", "pay"],
        "min_match":     1,
        "description":   "Salary dataset needs work experience columns",
    },
}


# ── Problem understanding rules ───────────────────────────────────────────────
PROBLEM_UNDERSTANDING_RULES = [
    # Format: (trigger_words, inferred_task, inferred_domain, data_type, use_case)

    # Sentiment / NLP
    (["sentiment", "opinion", "emotion", "feeling", "positive", "negative"],
     "classification", "sentiment", "text", "sentiment_analysis"),

    (["review", "rating", "feedback", "comment"],
     "classification", "sentiment", "text", "sentiment_analysis"),

    (["text classif", "text categor", "document classif"],
     "classification", "nlp", "text", "text_classification"),

    (["fake news", "misinformation", "spam", "phishing"],
     "classification", "nlp", "text", "text_classification"),

    # Mental Health
    (["mental health", "depression", "anxiety", "stress", "ptsd",
      "bipolar", "schizophrenia", "psychiatric", "psychology", "therapy"],
     "classification", "mental_health", "tabular_or_text", "diagnosis"),

    (["suicide", "self harm", "mental illness"],
     "classification", "mental_health", "text", "risk_detection"),

    # Medical
    (["diabetes", "glucose", "insulin", "diabetic"],
     "classification", "diabetes", "tabular", "prediction"),

    (["heart", "cardiac", "cardiovascular", "coronary", "ecg"],
     "classification", "heart_disease", "tabular", "prediction"),

    (["cancer", "tumor", "tumour", "malignant", "biopsy", "oncology"],
     "classification", "cancer", "tabular", "diagnosis"),

    (["parkinson", "tremor", "dopamine", "motor", "neurodegenerative"],
     "classification", "parkinson", "tabular", "diagnosis"),

    (["patient", "hospital", "clinical", "medical", "disease", "symptom",
      "diagnosis", "health condition"],
     "classification", "medical", "tabular", "prediction"),

    # Finance
    (["churn", "customer retention", "attrition", "subscription cancel"],
     "classification", "churn", "tabular", "prediction"),

    (["fraud", "scam", "anomaly detection", "suspicious transaction"],
     "classification", "fraud", "tabular", "detection"),

    (["loan", "credit", "mortgage", "default", "creditworthiness"],
     "classification", "loan", "tabular", "prediction"),

    # Prices
    (["house price", "home price", "real estate", "property value", "rent"],
     "regression", "house_price", "tabular", "price_prediction"),

    (["salary", "wage", "income", "compensation", "pay prediction"],
     "regression", "salary", "tabular", "price_prediction"),
]


@dataclass
class ProblemUnderstanding:
    """Structured understanding of a user's ML idea."""
    raw_idea:      str
    task_type:     str = "classification"
    domain:        str = "general"
    data_type:     str = "tabular"     # tabular | text | image | time_series
    use_case:      str = "prediction"
    key_entities:  List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    relevant_columns: List[str] = field(default_factory=list)


class ProblemAnalyser:
    """
    Understands the user's idea deeply before searching for a dataset.
    Converts natural language into structured domain understanding.
    """

    def analyse(self, idea: str) -> ProblemUnderstanding:
        idea_lower = idea.lower()

        pu = ProblemUnderstanding(raw_idea=idea)

        # Match against rules
        best_match = None
        best_score = 0
        for triggers, task, domain, data_type, use_case in PROBLEM_UNDERSTANDING_RULES:
            score = sum(1 for t in triggers if t in idea_lower)
            if score > best_score:
                best_score = score
                best_match = (task, domain, data_type, use_case)

        if best_match:
            pu.task_type, pu.domain, pu.data_type, pu.use_case = best_match

        # Extract key entities (non-stopword content words)
        STOP = {
            "predict", "classify", "detect", "build", "create", "model",
            "system", "analysis", "analyse", "analyze", "make", "using",
            "with", "from", "that", "will", "based", "help", "want",
            "need", "find", "show", "give", "also", "like", "just",
            "data", "dataset", "machine", "learning", "deep", "neural",
        }
        pu.key_entities = [
            w for w in re.findall(r'\b[a-z]{3,}\b', idea_lower)
            if w not in STOP
        ]

        # Get expected relevant columns from domain signature
        sig = DOMAIN_COLUMN_SIGNATURES.get(pu.domain, {})
        pu.relevant_columns = (
            sig.get("must_have_any", []) +
            sig.get("target_patterns", [])
        )

        # Generate smart search queries
        pu.search_queries = self._generate_queries(pu)

        logger.info(
            f"[ProblemAnalyser] Idea: '{idea}'\n"
            f"  task={pu.task_type} | domain={pu.domain} | "
            f"  data_type={pu.data_type} | use_case={pu.use_case}\n"
            f"  queries={pu.search_queries}"
        )

        return pu

    def _generate_queries(self, pu: ProblemUnderstanding) -> List[str]:
        """Generate diverse smart search queries instead of raw idea text."""
        entities = " ".join(pu.key_entities[:4])
        queries  = []

        # Query 1: most specific
        queries.append(
            f"{entities} dataset {pu.task_type} csv"
        )
        # Query 2: domain + task
        queries.append(
            f"{pu.domain.replace('_', ' ')} {pu.task_type} dataset csv"
        )
        # Query 3: use case specific
        if pu.data_type == "text":
            queries.append(f"{entities} text dataset nlp")
        elif pu.use_case == "diagnosis":
            queries.append(f"{entities} clinical dataset diagnosis")
        elif pu.use_case == "price_prediction":
            queries.append(f"{entities} regression dataset csv")

        # Query 4: Kaggle style
        queries.append(f"{entities} kaggle dataset")

        # Deduplicate
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique


class DatasetQualityEvaluator:
    """
    Evaluates whether a downloaded dataset is actually relevant
    and meets quality standards for the given problem.

    REPLACES the simple keyword-matching relevance check.
    Now checks:
    1. Domain column signature match
    2. Dataset size adequacy
    3. Target column presence
    4. Data quality (missing %, duplicates)
    5. Rejects known junk datasets (catalogs, indexes, geo data)
    """

    def evaluate(
        self,
        df: pd.DataFrame,
        pu: ProblemUnderstanding,
        filename: str = "",
    ) -> Tuple[bool, float, str]:
        """
        Returns (is_relevant, quality_score_0_to_1, reason).

        TASK 5 FIX: Scoring-based approach instead of hard rejection.
        Only outright junk files (catalog indexes, astronomical data) are
        rejected. Everything else gets a quality score and the pipeline
        picks the best available dataset rather than failing completely.
        """
        cols_lower = [c.lower().replace(" ", "_").replace("/", "_")
                      for c in df.columns]
        col_text   = " ".join(cols_lower)
        n_rows, n_cols = df.shape

        # ── HARD REJECT ONLY: known structural junk (not ML datasets) ─────
        # These are dataset catalogs/indexes, not actual ML data
        junk_patterns = [
            (["package", "rows", "cols"],           "R dataset catalog"),
            (["package", "item", "title"],           "Package index"),
            (["hallucination", "appropriateness"],   "LLM eval dataset"),
            (["obj_id", "alpha", "delta"],           "Astronomical SDSS data"),
            (["ra", "dec", "redshift"],              "Astronomy data"),
            (["shape_reported", "colors_reported"],  "UFO sightings"),
            (["dataset_name", "num_rows", "num_cols"], "Dataset registry"),
        ]
        for patterns, reason in junk_patterns:
            if sum(1 for p in patterns if p in col_text) >= 2:
                return False, 0.0, f"REJECTED structural junk: {reason}"

        # ── Absolute minimum size ─────────────────────────────────────────
        if n_rows < 20:
            return False, 0.0, f"Too small: only {n_rows} rows"
        if n_cols < 2:
            return False, 0.0, f"Too few columns: {n_cols}"

        # ── Quality score (soft scoring — no hard rejection beyond above) ──
        score = 0.50   # neutral start

        # Size scoring
        if n_rows >= 50000:  score += 0.20
        elif n_rows >= 10000: score += 0.15
        elif n_rows >= 1000:  score += 0.10
        elif n_rows >= 100:   score += 0.05
        else:                 score -= 0.10

        # Missing value penalty
        missing_pct = df.isnull().mean().mean()
        score -= min(0.25, missing_pct * 1.5)

        # Duplicate penalty
        dup_pct = df.duplicated().mean()
        score -= min(0.15, dup_pct)

        # ── Domain column signature match (bonus, not gating) ────────────
        sig = DOMAIN_COLUMN_SIGNATURES.get(pu.domain, {})
        if sig:
            must_have = sig.get("must_have_any", [])
            matches   = [p for p in must_have if p in col_text]
            if matches:
                score += min(0.20, 0.05 * len(matches))
            # No penalty for missing — might just be a different schema

        # ── Keyword match bonus ───────────────────────────────────────────
        key_entities = pu.key_entities[:6]
        if key_entities:
            entity_match = sum(1 for e in key_entities if e in col_text)
            try:
                sample_text = " ".join(
                    str(v) for v in df.head(5).values.flatten()
                ).lower()
                data_match = sum(1 for e in key_entities if e in sample_text)
            except Exception:
                data_match = 0

            total_match = entity_match + data_match
            if total_match >= 2:
                score += 0.15
            elif total_match == 1:
                score += 0.05
            # 0 matches -> no bonus, no penalty (score stays neutral)

        # ── NLP text column check (soft) ─────────────────────────────────
        if pu.data_type == "text":
            text_cols = [
                c for c in df.columns
                if df[c].dtype == object
                and df[c].dropna().str.len().mean() > 20
            ]
            if text_cols:
                score += 0.15   # bonus for NLP-appropriate columns
            else:
                score -= 0.20   # penalty but NOT rejection

        score = max(0.0, min(1.0, round(score, 3)))

        # Accept everything that scores > 0 (junk already rejected above)
        is_relevant = True
        reason = (
            f"score={score:.2f} | rows={n_rows:,} | cols={n_cols} | "
            f"missing={missing_pct:.1%} | "
            f"keyword_matches="
            f"{sum(1 for e in (pu.key_entities or [])[:6] if e in col_text)}"
            f"/{len((pu.key_entities or [])[:6])}"
        )
        return is_relevant, score, reason
