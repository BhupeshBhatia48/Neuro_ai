"""
model_engine/model_strategy_engine.py
---------------------------------------
The brain of the Neuro AI training pipeline.

Replaces static ModelSelector with an intelligent decision layer that
reads the DatasetProfile and decides:
  - Which models to use
  - Whether to use AutoML (FLAML)
  - Whether to use Deep Learning
  - Whether to override user choice based on data characteristics

Decision hierarchy (highest priority first):
  1. Time-series data       -> LSTM/GRU (DL)
  2. Text/NLP data          -> BERT/DistilBERT (DL)
  3. High categorical ratio -> CatBoost/LightGBM (no AutoML, fast)
  4. Large dataset (>100k)  -> LightGBM only (speed + memory)
  5. Medium dataset (>10k)  -> LightGBM + XGBoost + RF (AutoML)
  6. Small dataset (<10k)   -> Full model comparison (AutoML)
  7. Regression task        -> Adjusted model set
  8. Fallback               -> XGBoost + RandomForest (AutoML)

Backward compatibility:
  - ModelSelector is kept and used as fallback
  - All existing imports still work
  - If ModelStrategyEngine fails, pipeline falls back to ModelSelector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

# Thresholds
LARGE_DATASET_ROWS     = 100_000
MEDIUM_DATASET_ROWS    = 10_000
HIGH_CARDINALITY_RATIO = 0.4   # >40% categorical columns = high cardinality
HIGH_FEATURE_COUNT     = 500   # after encoding


@dataclass
class StrategyDecision:
    """Output of ModelStrategyEngine.decide()"""
    models:            List[str]
    use_automl:        bool
    use_deep_learning: bool
    reasoning:         str
    automl_time_budget: int = 120
    dl_model:          Optional[str] = None


class ModelStrategyEngine:
    """
    Intelligent model selection brain.
    Reads DatasetProfile and decides the optimal training strategy.
    """

    def decide(
        self,
        profile,
        user_model_choice: Optional[str] = None,
        effective_mode: str = "auto",
    ) -> StrategyDecision:
        """
        Input:
            profile           : DatasetProfile from DatasetAnalyzer
            user_model_choice : explicit user override (or None)
            effective_mode    : "auto" | "manual" from .env / sidebar

        Output:
            StrategyDecision with models, use_automl, use_deep_learning
        """
        try:
            decision = self._decide_internal(
                profile, user_model_choice, effective_mode
            )
            self._log_decision(decision)
            return decision
        except Exception as exc:
            logger.warning(
                f"[ModelStrategyEngine] Failed: {exc}. "
                f"Falling back to ModelSelector."
            )
            return self._fallback_decision(profile)

    def _decide_internal(
        self,
        profile,
        user_model_choice: Optional[str],
        effective_mode: str,
    ) -> StrategyDecision:

        task       = getattr(profile, "task_type",         "classification")
        n_rows     = getattr(profile, "num_rows",           0)
        n_features = getattr(profile, "num_features",       0)
        cat_cols   = getattr(profile, "categorical_columns", [])
        text_cols  = getattr(profile, "text_columns",       [])
        dt_cols    = getattr(profile, "datetime_columns",   [])
        num_cols   = getattr(profile, "numeric_columns",    [])

        total_cols        = max(len(cat_cols) + len(num_cols) + len(text_cols), 1)
        cat_ratio         = len(cat_cols) / total_cols
        has_text          = len(text_cols) > 0
        has_datetime      = len(dt_cols)   > 0
        is_time_series    = task == "time_series" or has_datetime
        is_regression     = task == "regression"

        # ── RULE 1: User explicitly chose a model ────────────────────────────
        if (user_model_choice
                and user_model_choice.lower() not in ("auto", "all", "")):
            return StrategyDecision(
                models            = [user_model_choice],
                use_automl        = False,
                use_deep_learning = user_model_choice in DL_MODEL_NAMES,
                reasoning         = f"User explicitly requested: {user_model_choice}",
                dl_model          = user_model_choice if user_model_choice in DL_MODEL_NAMES else None,
            )

        # ── RULE 2: Time-series data ─────────────────────────────────────────
        if is_time_series:
            return StrategyDecision(
                models            = ["LSTM", "GRU"],
                use_automl        = False,
                use_deep_learning = True,
                reasoning         = (
                    f"Time-series detected "
                    f"(task={task}, datetime_cols={dt_cols[:3]}). "
                    f"Using LSTM/GRU."
                ),
                dl_model          = "LSTM",
            )

        # ── RULE 3: Text/NLP data ────────────────────────────────────────────
        if has_text:
            return StrategyDecision(
                models            = ["DistilBERT", "BERT"],
                use_automl        = False,
                use_deep_learning = True,
                reasoning         = (
                    f"Text columns detected: {text_cols[:3]}. "
                    f"Using DistilBERT for NLP."
                ),
                dl_model          = "DistilBERT",
            )

        # ── RULE 4: High cardinality categorical ratio ────────────────────────
        if cat_ratio > HIGH_CARDINALITY_RATIO and len(cat_cols) > 3:
            if is_regression:
                models = ["CatBoost", "LightGBM", "XGBoost"]
            else:
                models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
            return StrategyDecision(
                models            = models,
                use_automl        = False,
                use_deep_learning = False,
                reasoning         = (
                    f"High categorical ratio ({cat_ratio:.0%}, "
                    f"{len(cat_cols)} cat cols). "
                    f"CatBoost/LightGBM handle categoricals natively."
                ),
                automl_time_budget = 60,
            )

        # ── RULE 5: Very large dataset ───────────────────────────────────────
        if n_rows > LARGE_DATASET_ROWS:
            if is_regression:
                models = ["LightGBM", "XGBoost"]
            else:
                models = ["LightGBM", "XGBoost", "RandomForest"]
            return StrategyDecision(
                models            = models,
                use_automl        = True,
                use_deep_learning = False,
                reasoning         = (
                    f"Large dataset ({n_rows:,} rows). "
                    f"LightGBM prioritised for speed."
                ),
                automl_time_budget = 90,
            )

        # ── RULE 6: Medium dataset ───────────────────────────────────────────
        if n_rows > MEDIUM_DATASET_ROWS:
            if is_regression:
                models = ["LightGBM", "XGBoost", "RandomForest",
                           "ExtraTrees", "Ridge"]
            else:
                models = ["LightGBM", "XGBoost", "RandomForest",
                           "ExtraTrees", "LogisticRegression"]
            return StrategyDecision(
                models            = models,
                use_automl        = True,
                use_deep_learning = False,
                reasoning         = (
                    f"Medium dataset ({n_rows:,} rows). "
                    f"Full comparison with AutoML."
                ),
                automl_time_budget = 120,
            )

        # ── RULE 7: Small dataset — full model comparison ────────────────────
        if is_regression:
            models = ["RandomForest", "XGBoost", "LightGBM",
                       "Ridge", "LinearRegression", "ExtraTrees"]
        else:
            models = ["RandomForest", "XGBoost", "LightGBM",
                       "LogisticRegression", "ExtraTrees", "SVM"]

        return StrategyDecision(
            models            = models,
            use_automl        = True,
            use_deep_learning = False,
            reasoning         = (
                f"Small dataset ({n_rows:,} rows, {n_features} features). "
                f"Full model comparison."
            ),
            automl_time_budget = 120,
        )

    def _fallback_decision(self, profile) -> StrategyDecision:
        """Falls back to ModelSelector when strategy engine fails."""
        try:
            from model_engine.model_selector import ModelSelector
            selector = ModelSelector()
            models   = selector.recommend(profile)
            return StrategyDecision(
                models            = models,
                use_automl        = True,
                use_deep_learning = False,
                reasoning         = "Fallback to ModelSelector (strategy failed)",
            )
        except Exception:
            return StrategyDecision(
                models            = ["XGBoost", "RandomForest", "LightGBM"],
                use_automl        = True,
                use_deep_learning = False,
                reasoning         = "Hard fallback defaults",
            )

    def _log_decision(self, d: StrategyDecision) -> None:
        logger.info("=" * 55)
        logger.info("[ModelStrategyEngine] Strategy Decision:")
        logger.info(f"  Models selected  : {d.models}")
        logger.info(f"  Use AutoML       : {d.use_automl}")
        logger.info(f"  Use DeepLearning : {d.use_deep_learning}")
        logger.info(f"  AutoML budget    : {d.automl_time_budget}s")
        logger.info(f"  Reasoning        : {d.reasoning}")
        logger.info("=" * 55)

    def refine_with_llm(
        self,
        decision: StrategyDecision,
        profile,
        idea: str,
    ) -> StrategyDecision:
        """
        Optional: use LLM to refine the model list.
        Only called when USE_LLM=True. Rule-based decision stays as primary.
        LLM can only ADD models from allowed list, not replace the decision.
        """
        from config.settings import USE_LLM
        if not USE_LLM:
            return decision

        try:
            from llm_agent.openrouter_client import OpenRouterClient

            allowed = [
                "XGBoost", "LightGBM", "RandomForest", "CatBoost",
                "LogisticRegression", "LinearRegression", "Ridge",
                "ExtraTrees", "SVM", "LSTM", "GRU", "DistilBERT",
            ]
            system_prompt = (
                "You are an ML model selection assistant. "
                "Given a dataset profile, suggest the best model names from "
                f"this allowed list ONLY: {allowed}. "
                "Reply with a comma-separated list of model names. Nothing else."
            )
            user_prompt = (
                f"Idea: {idea}\n"
                f"Task: {getattr(profile,'task_type','classification')}\n"
                f"Rows: {getattr(profile,'num_rows',0):,}\n"
                f"Features: {getattr(profile,'num_features',0)}\n"
                f"Text columns: {len(getattr(profile,'text_columns',[]))}\n"
                f"Categorical columns: {len(getattr(profile,'categorical_columns',[]))}\n"
                f"Current models: {decision.models}\n"
                f"Should I add or swap any models? "
                f"Reply with comma-separated names only."
            )

            client   = OpenRouterClient()
            response = client.chat(
                messages      = [{"role": "user", "content": user_prompt}],
                system_prompt = system_prompt,
                max_tokens    = 60,
                temperature   = 0.0,
            ).strip()

            suggested = [
                m.strip() for m in response.split(",")
                if m.strip() in allowed
            ]

            if suggested:
                # Merge: keep existing + add new suggestions, deduplicate
                merged = list(dict.fromkeys(decision.models + suggested))
                decision.models   = merged[:6]   # cap at 6
                decision.reasoning += f" | LLM added: {suggested}"
                logger.info(
                    f"[ModelStrategyEngine] LLM refined models: {decision.models}"
                )

        except Exception as exc:
            logger.warning(f"[ModelStrategyEngine] LLM refinement failed: {exc}")

        return decision


# Keep DL_MODEL_NAMES consistent with main.py
DL_MODEL_NAMES = {
    "LSTM", "GRU", "MLP", "BERT", "DistilBERT", "RoBERTa",
    "ResNet18", "ResNet50", "EfficientNet", "ViT", "MobileNet",
    "ARIMA", "Prophet",
}
