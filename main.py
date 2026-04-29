"""
main.py
--------
Neuro AI — Autonomous ML Engineer

BUGS FIXED IN THIS VERSION:
----------------------------
BUG 1 (CRITICAL - WHY CHESS DATASET KEPT APPEARING):
  Every run called downloader.download(url, name="dataset")
  This saved EVERY dataset as datasets/dataset.csv
  So run 1 (chess) saved datasets/dataset.csv
  Run 2 (churn) failed to download -> local_dataset_path = None
  Pipeline raised PipelineError BUT Streamlit showed previous chess results
  
  FIX: Generate a unique name per idea using a slug from the idea text.
       Each idea gets its own file: datasets/customer_churn_abc123.csv
       Old files are never reused for new ideas.

BUG 2: File upload mode added to Streamlit.
  User can now upload their own CSV directly — no search needed.
  This is the most reliable way to use the system with known datasets.

BUG 3: Dataset search fallback was silently succeeding with wrong data.
  FIX: Added content validation in downloader (HTML detection).
  FIX: If ALL downloads fail, raise clear PipelineError immediately.
"""

from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from config.settings import (
    MODELS_DIR, MAX_DATASET_ROWS, DEFAULT_RANDOM_STATE,
    AUTOML_MODE, USE_LLM,
)

from llm_agent.idea_parser import IdeaParser
from llm_agent.keyword_extractor import KeywordExtractor
from llm_agent.task_classifier import TaskClassifier

from search_engine.dataset_sources import MultiSourceDatasetSearch
from search_engine.result_filter import ResultFilter
from search_engine.dataset_ranker import DatasetRanker
from search_engine.dataset_intelligence import ProblemAnalyser, DatasetQualityEvaluator
from search_engine.concept_expander import ConceptExpander

from dataset_engine.dataset_downloader import DatasetDownloader
from dataset_engine.dataset_loader import DatasetLoader
from dataset_engine.dataset_validator import DatasetValidator

from data_analysis.dataset_analyzer import DatasetAnalyzer
from data_analysis.feature_detector import FeatureDetector
from data_analysis.target_detector import TargetDetector, AmbiguousTargetError

from preprocessing.cleaning import DataCleaner
from preprocessing.pipeline_builder import PreprocessingPipeline

from model_engine.model_selector import ModelSelector
from model_engine.model_strategy_engine import ModelStrategyEngine
from model_engine.trainer import ModelTrainer
from model_engine.evaluator import ModelEvaluator
from model_engine.hyperparameter_tuner import HyperparameterTuner

from automl.automl_controller import AutoMLController

from export_engine.pickle_exporter import PickleExporter
from export_engine.report_generator import ReportGenerator

console = Console()

DL_MODEL_NAMES = {
    "LSTM", "GRU", "MLP", "BERT", "DistilBERT", "RoBERTa",
    "ResNet18", "ResNet50", "EfficientNet", "EfficientNetB7",
    "ViT", "MobileNet", "ARIMA", "Prophet",
}


class PipelineError(RuntimeError):
    pass


def _idea_to_slug(idea: str) -> str:
    """
    Convert idea text to a safe filename slug.
    'Predict customer churn' -> 'predict_customer_churn'
    BUG FIX: Each idea now gets a UNIQUE filename so old datasets
    are never accidentally reused for a different idea.
    """
    slug = re.sub(r"[^a-zA-Z0-9\s]", "", idea.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = slug[:40]  # max 40 chars
    # Add short hash to guarantee uniqueness even for similar ideas
    uid  = hashlib.md5(idea.encode()).hexdigest()[:6]
    return f"{slug}_{uid}"


def _make_fallback_pu(idea: str):
    """Create a minimal ProblemUnderstanding for fallback when analyser fails."""
    from search_engine.dataset_intelligence import ProblemUnderstanding
    pu = ProblemUnderstanding(raw_idea=idea)
    pu.key_entities = [
        w for w in idea.lower().split()
        if len(w) > 3 and w not in {
            "predict", "classify", "detect", "model", "build",
            "create", "using", "with", "from", "that", "will",
        }
    ]
    return pu


def _is_dataset_relevant(idea: str, df: "pd.DataFrame") -> bool:
    """
    Returns False if the dataset is clearly unrelated to the idea.

    IMPROVED: Now catches catalog/index files and meta-datasets that
    describe other datasets instead of containing actual ML data.
    e.g. columns like: Package, Item, Title, Rows, Cols, n_binary
    These are dataset registries/indexes, not actual datasets.
    """
    if not idea or len(df.columns) == 0:
        return True

    cols     = df.columns.tolist()
    col_text = " ".join(cols).lower()
    col_set  = {c.lower().strip() for c in cols}

    # ── HARD REJECT 1: Dataset catalog/index files ───────────────────────────
    # These files describe other datasets, not actual ML data
    catalog_signals = [
        {"rows", "cols", "package"},          # R dataset catalog
        {"rows", "cols", "n_binary"},          # dataset registry
        {"package", "item", "title"},          # package catalog
        {"dataset", "rows", "variables"},      # dataset index
        {"name", "rows", "columns", "source"}, # dataset listing
        {"dataset_name", "num_rows", "num_cols"},
    ]
    for signal_set in catalog_signals:
        if len(signal_set & col_set) >= 3:
            logger.warning(
                f"[Relevance] Detected CATALOG/INDEX file "
                f"(columns: {cols[:6]}) — this lists other datasets, "
                f"not actual ML data. Rejecting."
            )
            return False

    # ── HARD REJECT 2: Known wrong dataset patterns ──────────────────────────
    wrong_patterns = [
        # LLM evaluation dataset
        {"hallucination", "appropriateness", "prompt category"},
        {"hallucination/accuracy", "bias", "interesting?"},
        # UFO sightings
        {"shape reported", "colors reported"},
        # Pure geo data
        {"longitude", "latitude", "elevation"},
    ]
    for pattern in wrong_patterns:
        if sum(1 for s in pattern if s in col_text) >= 2:
            return False

    # ── KEYWORD MATCH: idea words vs column names ────────────────────────────
    # STOP_WORDS: only remove truly generic pipeline words
    # DO NOT remove domain-specific words like health, patients, medical, etc.
    STOP_WORDS = {
        "predict", "classify", "detect", "build", "create", "make",
        "model", "system", "that", "will", "from", "with", "using",
        "based", "whether", "data", "dataset", "machine", "learning",
        "train", "test", "deep", "neural", "want", "need", "help",
        "find", "show", "give", "make", "also", "like", "just",
    }
    # Keep domain-specific terms: health, patient, medical, knee, etc.
    idea_words = [
        w for w in idea.lower().split()
        if len(w) > 3 and w not in STOP_WORDS
    ]

    if not idea_words:
        return True  # Not enough keywords to judge

    # If ANY idea keyword matches any column name or column text — accept
    if any(w in col_text for w in idea_words):
        return True

    # Also check if idea words appear in actual DATA values (sample rows)
    try:
        sample_text = " ".join(
            str(v) for v in df.head(5).values.flatten()
        ).lower()
        if any(w in sample_text for w in idea_words):
            return True
    except Exception:
        pass

    # Zero keyword matches + 1+ meaningful domain keyword = likely wrong dataset
    # (threshold lowered from 2 to 1 to catch cases like knee/star mismatch)
    if len(idea_words) >= 1:
        logger.warning(
            f"[Relevance] Keywords {idea_words} not found in "
            f"columns {cols[:8]} or data values. Dataset likely irrelevant."
        )
        return False

    return True  # Default allow


def _warn_dataset_relevance(idea: str, df: "pd.DataFrame", path) -> None:
    """
    Checks if the downloaded dataset looks relevant to the user's idea.
    Warns clearly if the dataset seems completely unrelated.
    This prevents silent wrong-dataset situations (e.g. UFO sightings
    downloaded for a content recommendation task).
    """
    idea_lower = idea.lower()
    col_names  = " ".join(df.columns.tolist()).lower()

    # Red flag column names that suggest clearly wrong datasets
    red_flags = {
        "ufo":           ["city", "shape reported", "colors reported", "duration"],
        "weather":       ["temperature", "humidity", "wind_speed", "precipitation"],
        "earthquake":    ["magnitude", "depth", "latitude", "longitude"],
        "covid":         ["confirmed", "deaths", "recovered", "province"],
    }

    for wrong_topic, flag_cols in red_flags.items():
        matches = sum(1 for fc in flag_cols if fc in col_names)
        if matches >= 2 and wrong_topic not in idea_lower:
            logger.warning(
                f"[Relevance Warning] The downloaded dataset appears to be "
                f"a {wrong_topic.upper()} dataset (columns: {df.columns.tolist()[:6]}) "
                f"but your idea was about '{idea}'. "
                f"This may produce meaningless results. "
                f"Consider uploading the correct dataset using the sidebar file upload."
            )
            return

    # Check if idea keywords appear anywhere in column names
    idea_words = [
        w for w in idea_lower.split()
        if len(w) > 3 and w not in {
            "predict", "classify", "detect", "build", "create",
            "make", "model", "system", "that", "will", "from",
            "with", "using", "based", "whether", "patient", "data",
        }
    ]
    if idea_words:
        match_count = sum(1 for w in idea_words if w in col_names)
        if match_count == 0 and len(idea_words) >= 2:
            logger.warning(
                f"[Relevance Warning] None of the idea keywords {idea_words} "
                f"appear in the dataset columns {df.columns.tolist()[:8]}. "
                f"The dataset may not be relevant to your idea. "
                f"Upload your own dataset using the sidebar for best results."
            )


def run_pipeline(
    user_idea: str,
    dataset_url: Optional[str]      = None,
    dataset_file: Optional[Path]    = None,   # NEW: direct file upload path
    user_model_choice: Optional[str] = None,
    target_column: Optional[str]    = None,
    tune: bool  = False,
    mode: Optional[str] = None,
) -> dict:
    """
    Full Neuro AI pipeline.

    Parameters
    ----------
    user_idea        : Natural-language idea
    dataset_url      : Direct download URL (skips search)
    dataset_file     : Path to a locally uploaded file (most reliable)
    user_model_choice: "auto" | model name | number
    target_column    : Override auto-detection
    tune             : Run Optuna tuning after training
    mode             : "auto" (FLAML) | "manual" (ModelTrainer)
    """
    effective_mode = (mode or AUTOML_MODE).lower()

    console.print(Panel(
        f"[bold cyan]Neuro AI[/]\n{user_idea or '(Direct mode)'}\n"
        f"[dim]mode={effective_mode}  llm={'on' if USE_LLM else 'off'}[/]",
        expand=False,
    ))

    # ── STEP 1: Deep problem understanding ───────────────────────────────────
    logger.info("Step 1 -> Understanding problem deeply...")
    pu = None   # ProblemUnderstanding object

    if not user_idea and (dataset_url or dataset_file):
        from llm_agent.idea_parser import ParsedIdea
        parsed_idea = ParsedIdea(
            raw_idea    = "Direct file mode",
            task_type   = "classification",
            domain      = "general",
            keywords    = ["dataset"],
            parsed_by   = "rules",
        )
    else:
        # Smart problem analysis FIRST
        try:
            pu = ProblemAnalyser().analyse(user_idea or "")
            logger.info(
                f"[ProblemAnalyser] task={pu.task_type} | "
                f"domain={pu.domain} | data_type={pu.data_type} | "
                f"use_case={pu.use_case}"
            )
            logger.info(
                f"[ProblemAnalyser] Smart queries: {pu.search_queries}"
            )
            logger.info(
                f"[ProblemAnalyser] Expected columns: "
                f"{pu.relevant_columns[:8]}"
            )
        except Exception as exc:
            logger.warning(f"ProblemAnalyser failed: {exc} — using fallback")
            pu = None

        # LLM/rule-based idea parser (for task_type, keywords)
        parsed_idea = IdeaParser().parse(user_idea or "")

        # Sync task_type from ProblemAnalyser if available
        if pu and pu.task_type != "classification":
            parsed_idea.task_type = pu.task_type

    logger.info(
        f"Parsed -> task={parsed_idea.task_type}  "
        f"domain={parsed_idea.domain if hasattr(parsed_idea,'domain') else 'unknown'}  "
        f"by={parsed_idea.parsed_by}"
    )

    # ── STEP 2: Semantic concept expansion + query generation ────────────────
    expansion    = None
    idea_kws_for_filter = []

    try:
        expander  = ConceptExpander()
        expansion = expander.expand(
            idea      = user_idea or "",
            task_type = parsed_idea.task_type,
            domain    = getattr(parsed_idea, "domain", "general") or "general",
        )
        # Collect all terms (core + expanded) for ResultFilter scoring
        idea_kws_for_filter = (
            expansion.core_terms +
            expansion.expanded_terms[:4]
        )
        logger.info(
            f"[ConceptExpander] core={expansion.core_terms} | "
            f"expanded={expansion.expanded_terms[:5]} | "
            f"source={expansion.source}"
        )
    except Exception as exc:
        logger.warning(f"ConceptExpander failed: {exc} — using basic queries")
        expansion           = None
        idea_kws_for_filter = []

    # Build final queries: ProblemAnalyser base + expanded semantic queries
    extractor = KeywordExtractor()
    if pu and pu.search_queries:
        # Start with ProblemAnalyser queries, enrich with expansion
        base_queries = pu.search_queries
        keywords_str = " ".join(pu.key_entities[:5])
        # Merge: ProblemAnalyser queries first, then expansion variations
        extra = extractor.extract(parsed_idea, expansion=expansion)
        # Combine deduped
        seen_q: set = set(base_queries)
        for q in extra:
            if q not in seen_q:
                seen_q.add(q)
                base_queries = base_queries + [q]
        queries = base_queries[:8]
        logger.info(
            f"Using {len(queries)} queries "
            f"(ProblemAnalyser + ConceptExpander)"
        )
    else:
        queries      = extractor.extract(parsed_idea, expansion=expansion)
        keywords_str = " ".join(parsed_idea.keywords)

    if not idea_kws_for_filter and pu:
        idea_kws_for_filter = pu.key_entities[:6]

    # ── STEP 3: Get Dataset ───────────────────────────────────────────────────
    local_dataset_path: Optional[Path] = None

    # BUG FIX: Generate unique name per idea so files never clash
    dataset_name = _idea_to_slug(user_idea or "direct_upload")

    if dataset_file and Path(dataset_file).exists():
        # Option A: User uploaded a file directly — most reliable
        logger.info(f"Step 3 -> Using uploaded file: {dataset_file}")
        local_dataset_path = Path(dataset_file)

    elif dataset_url:
        # Option B: Direct URL provided
        logger.info(f"Step 3 -> Downloading from URL: {dataset_url}")
        local_dataset_path = DatasetDownloader().download(
            dataset_url, name=dataset_name
        )
        if not local_dataset_path:
            raise PipelineError(
                f"Failed to download from: {dataset_url}\n"
                f"Possible reasons:\n"
                f"  - URL requires login (Kaggle, HuggingFace)\n"
                f"  - URL returned HTML instead of a data file\n"
                f"  - Domain not in allowed list\n"
                f"Please upload the file directly using the file upload in the sidebar."
            )

    else:
        # Option C: Auto search — least reliable
        logger.info(
            "Step 3 -> Searching for dataset "
            "(DDG + OpenML + UCI + HuggingFace + Kaggle)..."
        )
        raw_results = MultiSourceDatasetSearch(max_per_source=5).search(
            queries,
            keywords_str          = keywords_str,
            problem_understanding = pu,
            raw_idea              = user_idea or "",
        )
        # Pass idea keywords at construction for relevance scoring (Task 5)
        # Use core + expanded terms for broader semantic matching
        filtered = ResultFilter().filter(
            raw_results,
            idea_keywords=idea_kws_for_filter or (pu.key_entities[:6] if pu else []),
        )
        ranked   = DatasetRanker(top_n=5).rank(filtered)

        if not ranked:
            raise PipelineError(
                "No dataset candidates found from any search source.\n"
                "SOLUTION: Upload your dataset file directly using the "
                "file upload in the sidebar. This is the most reliable approach."
            )

        logger.info(f"Found {len(ranked)} candidates. Trying each...")
        downloader        = DatasetDownloader()
        rejected_urls     = []
        scored_candidates = []   # (quality_score, path, df, reason)

        for i, candidate in enumerate(ranked, 1):
            logger.info(f"  Trying [{i}/{len(ranked)}]: {candidate.url}")
            candidate_path = downloader.download(
                candidate.url, name=f"{dataset_name}_{i}"
            )
            if not candidate_path:
                logger.warning(f"  Download failed: {candidate.url}")
                continue

            # ── Quick relevance check BEFORE accepting this dataset ──────────
            try:
                candidate_df = DatasetLoader().load(candidate_path)
            except Exception as exc:
                logger.warning(f"  Could not load {candidate_path.name}: {exc}")
                candidate_path.unlink(missing_ok=True)
                continue

            if len(candidate_df) == 0:
                logger.warning(f"  Empty file: {candidate_path.name}")
                candidate_path.unlink(missing_ok=True)
                continue

            # Use smart DatasetQualityEvaluator instead of simple keyword check
            try:
                evaluator = DatasetQualityEvaluator()
                is_relevant, quality, reason = evaluator.evaluate(
                    candidate_df,
                    pu if pu is not None else _make_fallback_pu(user_idea or ""),
                    filename=candidate_path.name,
                )
            except Exception as exc:
                logger.warning(f"  Quality eval failed: {exc} — accepting by default")
                is_relevant, quality, reason = True, 0.5, "eval failed"

            if not is_relevant:
                logger.warning(
                    f"  REJECTED: {candidate_path.name} | {reason}"
                )
                rejected_urls.append(f"{candidate.url} ({reason})")
                candidate_path.unlink(missing_ok=True)
                continue

            # Dataset passed quality evaluation — collect with score
            scored_candidates.append((quality, candidate_path, candidate_df, reason))
            logger.info(
                f"  SCORED [{i}/{len(ranked)}]: {candidate_path.name} | "
                f"score={quality:.2f} | {reason}"
            )
            # Continue trying remaining candidates to find the best one
            # Stop early if we have a high-quality match
            if quality >= 0.70:
                logger.info(f"  High-quality match found (score={quality:.2f}) — stopping search")
                break

        # Select best-scored candidate
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_quality, best_path, best_df, best_reason = scored_candidates[0]
            local_dataset_path = best_path
            df                 = best_df
            logger.success(
                f"Best dataset selected: {best_path.name} | "
                f"score={best_quality:.2f} | {best_reason}"
            )
            # Clean up unchosen candidate files
            for _, path, _, _ in scored_candidates[1:]:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

        if not local_dataset_path:
            rejected_msg = (
                "\nRejected candidates:\n" +
                "\n".join(f"  - {u}" for u in rejected_urls)
                if rejected_urls else ""
            )
            raise PipelineError(
                f"Could not find a valid dataset for: '{user_idea}'\n"
                f"Tried {len(ranked)} candidates.{rejected_msg}\n\n"
                f"SOLUTION: Upload your own dataset using the sidebar."
            )

    if not local_dataset_path or not Path(local_dataset_path).exists():
        raise PipelineError(
            "Dataset file not found. Please upload your dataset directly."
        )

    # ── STEP 4-5: Load dataset ───────────────────────────────────────────────
    # For direct URL / file upload: load now
    # For auto-search: df was already set inside the scored_candidates loop
    if dataset_url or dataset_file:
        logger.info(f"Step 4 -> Loading: {local_dataset_path}")
        df = DatasetLoader().load(local_dataset_path)
        if len(df) == 0:
            raise PipelineError(f"Dataset has 0 rows: {local_dataset_path}")
        logger.info(f"Dataset ready: {len(df):,} rows x {df.shape[1]} columns")
        if not _is_dataset_relevant(user_idea or "", df):
            _warn_dataset_relevance(user_idea or "", df, local_dataset_path)
    elif "df" not in dir():
        # Fallback: df not set (search loop produced no candidates)
        raise PipelineError(
            "No dataset was loaded. All search candidates were rejected or failed. "
            "Please upload your dataset directly using the sidebar."
        )
    else:
        logger.info(
            f"Step 4 -> Dataset already loaded: "
            f"{len(df):,} rows x {df.shape[1]} columns"
        )

    if len(df) > MAX_DATASET_ROWS:
        logger.warning(
            f"Dataset has {len(df):,} rows — sampling to {MAX_DATASET_ROWS:,}"
        )
        df = df.sample(
            n=MAX_DATASET_ROWS, random_state=DEFAULT_RANDOM_STATE
        ).reset_index(drop=True)

    # ── STEP 5: Validate ──────────────────────────────────────────────────────
    logger.info("Step 5 -> Validating dataset quality...")
    val_report = DatasetValidator().validate(df, target_column=target_column)
    logger.info(f"Quality score: {val_report.quality_score}/100")

    # ── STEP 6: Analysis ──────────────────────────────────────────────────────
    logger.info("Step 6 -> Analysing dataset...")
    profile       = DatasetAnalyzer().analyze(df)
    feature_flags = FeatureDetector().detect(df, profile)

    raw_drop = {
        c for c, flags in feature_flags.items()
        if "constant" in flags or "id_like" in flags
    }

    # ── STEP 7: Target Detection ──────────────────────────────────────────────
    if target_column:
        if target_column not in df.columns:
            raise PipelineError(
                f"Specified target column '{target_column}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )
        profile.target_column = target_column
        logger.info(f"Using specified target column: '{target_column}'")
    else:
        detector = TargetDetector()
        try:
            detected, confidence = detector.detect(
                df, parsed_idea.task_type, raise_on_ambiguity=True
            )
            profile.target_column = detected
            logger.info(
                f"Target detected: '{detected}' "
                f"(confidence={confidence})"
            )
        except AmbiguousTargetError as e:
            logger.warning(
                f"Ambiguous target. Candidates: {e.candidates}. "
                "Trying LLM fallback..."
            )
            llm_target = detector.llm_fallback(
                df, parsed_idea.task_type, user_idea or ""
            )
            if llm_target:
                profile.target_column = llm_target
                logger.success(f"LLM selected target: '{llm_target}'")
            else:
                profile.target_column = e.candidates[0]
                logger.warning(
                    f"Using first candidate: '{profile.target_column}'. "
                    "Use --target to override."
                )

    if not profile.target_column:
        raise PipelineError(
            "Could not detect target column. "
            "Please specify it using --target <column_name> or the sidebar."
        )

    drop_cols = list(raw_drop - {profile.target_column})

    # Refine task type
    target_nunique    = int(df[profile.target_column].nunique())
    profile.task_type = TaskClassifier().classify(
        initial_task=parsed_idea.task_type,
        target_unique_values=target_nunique,
        has_datetime_index=len(profile.datetime_columns) > 0,
        has_text_columns=len(profile.text_columns) > 0,
    )
    logger.info(f"Final task type: {profile.task_type}")
    logger.info(f"Target column  : {profile.target_column}")
    logger.info(f"Target classes : {df[profile.target_column].unique().tolist()[:10]}")

    # ── STEP 8: Clean + Preprocess ────────────────────────────────────────────
    logger.info("Step 8 -> Cleaning...")
    df_clean = DataCleaner().clean(df, drop_columns=drop_cols)

    logger.info("Step 8b -> Preprocessing (split-first, zero leakage)...")
    prep_pipeline = PreprocessingPipeline()
    X_train, X_test, y_train, y_test = prep_pipeline.build_and_fit(
        df_clean, profile.target_column, profile.task_type
    )
    # Sync task_type from DataTypeEngine (may override LLM/rule-based detection)
    if (prep_pipeline.type_profile
            and prep_pipeline.type_profile.task_type != profile.task_type):
        logger.info(
            f"[Pipeline] Task type updated: "
            f"{profile.task_type} -> {prep_pipeline.type_profile.task_type}"
        )
        profile.task_type = prep_pipeline.type_profile.task_type

    # ── STEP 9: Intelligent Model Strategy Decision ──────────────────────────
    logger.info("Step 9 -> ModelStrategyEngine deciding training strategy...")
    strategy = ModelStrategyEngine()
    decision = strategy.decide(
        profile,
        user_model_choice = user_model_choice,
        effective_mode    = effective_mode,
    )

    # Optional LLM refinement (non-blocking)
    if USE_LLM:
        try:
            decision = strategy.refine_with_llm(
                decision, profile, user_idea or ""
            )
        except Exception:
            pass

    chosen_models = decision.models
    use_automl    = decision.use_automl
    use_dl        = decision.use_deep_learning

    # Respect explicit user mode override from sidebar/CLI
    if effective_mode == "manual":
        use_automl = False
    elif effective_mode == "auto" and not use_dl:
        use_automl = True

    logger.info(f"Strategy -> models={chosen_models} automl={use_automl} dl={use_dl}")

    # ── STEP 10: Training — driven by strategy decision ───────────────────────
    results    = []
    best       = None
    best_model = None

    # ── 10a: Deep Learning path ───────────────────────────────────────────────
    if use_dl:
        logger.info(
            f"Step 10 -> Deep Learning path "
            f"(model={decision.dl_model or chosen_models[0]})"
        )
        try:
            from deep_learning.dl_trainer import DeepLearningTrainer
            from model_engine.evaluator import ModelResult
            dl_name    = decision.dl_model or next(
                (m for m in chosen_models if m in DL_MODEL_NAMES),
                chosen_models[0]
            )
            dl_trainer = DeepLearningTrainer(
                model_name=dl_name, task_type=profile.task_type
            )
            dl_result = dl_trainer.train(X_train, X_test, y_train, y_test)
            primary_key = "f1_score" if profile.task_type == "classification" else "r2"
            results = [ModelResult(
                name           = dl_name,
                metrics        = dl_result.metrics,
                primary_metric = dl_result.metrics.get(primary_key, 0.0),
            )]
            best       = dl_name
            best_model = dl_trainer.model
            logger.success(f"Deep Learning training complete: {dl_name}")
        except Exception as exc:
            logger.error(
                f"Deep Learning failed: {exc}. Falling back to tabular path."
            )
            use_dl     = False
            use_automl = True   # fall through to AutoML

    # ── 10b: AutoML path (FLAML) ──────────────────────────────────────────────
    if not use_dl and use_automl:
        logger.info(
            f"Step 10 -> AutoML path (FLAML) | "
            f"budget={decision.automl_time_budget}s | "
            f"candidate models={chosen_models}"
        )
        automl_ctrl = AutoMLController(time_budget=decision.automl_time_budget)
        best_model, automl_metrics = automl_ctrl.run(
            X_train, y_train, X_test, y_test, profile.task_type
        )
        if best_model is not None:
            from model_engine.evaluator import ModelResult
            primary_key = "f1_score" if profile.task_type == "classification" else "r2"
            results = [ModelResult(
                name           = automl_ctrl.best_estimator_name,
                metrics        = automl_metrics,
                primary_metric = automl_metrics.get(primary_key, 0.0),
            )]
            best  = automl_ctrl.best_estimator_name
            tried = automl_ctrl.get_all_tried_estimators()
            logger.success(
                f"AutoML complete: best={best} | tried={tried}"
            )
        else:
            logger.warning("AutoML failed -> falling back to manual trainer")
            use_automl = False

    # ── 10c: Manual training path ─────────────────────────────────────────────
    if not use_dl and not use_automl or (not use_dl and not results):
        logger.info(
            f"Step 10 -> Manual training: {chosen_models}"
        )
        # Filter out any DL models — manual trainer only handles sklearn/boosting
        tabular_models = [
            m for m in chosen_models
            if m not in DL_MODEL_NAMES
        ] or ["XGBoost", "RandomForest", "LightGBM"]

        trainer = ModelTrainer()
        trained_models, X_test, y_test = trainer.train_all(
            X_train, X_test, y_train, y_test,
            tabular_models, profile.task_type
        )
        if not trained_models:
            raise PipelineError(
                "No models trained successfully. "
                f"Models attempted: {tabular_models}"
            )

        plots_dir = Path(str(MODELS_DIR)) / "plots"
        evaluator = ModelEvaluator()
        results, best = evaluator.evaluate_all(
            trained_models, X_test, y_test,
            profile.task_type, plots_dir=plots_dir,
        )
        best_model = trained_models[best]
        logger.success(
            f"Manual training complete: best={best} | "
            f"trained={list(trained_models.keys())}"
        )

        if tune:
            logger.info("Optuna tuning...")
            tuner = HyperparameterTuner(n_trials=30, cv=3)
            best_model = tuner.tune(
                best_model, best, X_train, y_train, profile.task_type
            )
            if tuner.study:
                PickleExporter().export_optuna_study(tuner.study, name=best)

    # ── STEP 11: Export ───────────────────────────────────────────────────────
    logger.info("Step 11 -> Exporting model and pipeline...")
    exporter      = PickleExporter()
    model_path    = exporter.export_model(best_model, name=best or "best")
    pipeline_path = exporter.export_pipeline(
        prep_pipeline.pipeline, name=best or "best"
    )

    # ── STEP 11b: Generate predict.py inference script ────────────────────────
    logger.info("Step 11b -> Generating predict.py inference script...")
    predict_script_path = None
    predict_code_str    = ""
    try:
        from export_engine.inference_generator import InferenceGenerator
        ig = InferenceGenerator()
        # Pass the inner LabelEncoder (not the TargetEncoder wrapper)
        # inference_generator calls .classes_ directly on this object
        _le = getattr(prep_pipeline.target_encoder, "label_encoder", None)
        predict_script_path = ig.generate(
            df            = df_clean,
            target_column = profile.target_column,
            task_type     = profile.task_type,
            model_name    = best or "best",
            model_path    = model_path,
            pipeline_path = pipeline_path,
            label_encoder = _le,
        )
        predict_code_str = ig.get_code_string(
            df            = df_clean,
            target_column = profile.target_column,
            task_type     = profile.task_type,
            model_name    = best or "best",
            model_path    = model_path,
            pipeline_path = pipeline_path,
            label_encoder = _le,
        )
        logger.success(f"predict.py saved -> {predict_script_path}")
    except Exception as exc:
        logger.warning(f"Could not generate predict.py: {exc}")
        predict_script_path = model_path  # fallback to model path
        predict_code_str    = f"# predict.py generation failed: {exc}"

    logger.info("Step 12 -> Generating report...")
    report_path = ReportGenerator().generate(
        idea=user_idea or "Direct mode",
        profile=profile,
        results=results,
        best_model_name=best or "best",
        model_path=model_path,
        pipeline_path=pipeline_path,
    )

    console.print(Panel(
        f"[bold green]DONE![/]\n\n"
        f"Best model   : [cyan]{best}[/]\n"
        f"Mode used    : {effective_mode}\n"
        f"Quality score: {val_report.quality_score}/100\n"
        f"Dataset      : {local_dataset_path.name if local_dataset_path else 'unknown'}\n"
        f"Report       : {report_path}",
        title="Neuro AI Complete",
        expand=False,
    ))

    return {
        "best_model_name":   best,
        "metrics":           results[0].metrics if results else {},
        "model_path":        str(model_path),
        "pipeline_path":     str(pipeline_path),
        "report_path":       str(report_path),
        "quality_score":     val_report.quality_score,
        "mode_used":         effective_mode,
        "dataset_used":      str(local_dataset_path) if local_dataset_path else "",
        "task_type":         profile.task_type,
        "target_column":     profile.target_column,
        "rows_trained_on":   len(df),
        "feature_columns":   [c for c in df_clean.columns if c != profile.target_column],
        "predict_script_path": str(predict_script_path),
        "predict_code":      predict_code_str,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Neuro AI")
    ap.add_argument("--idea",   default="", help="ML idea in plain English")
    ap.add_argument("--url",    default=None, help="Direct dataset URL")
    ap.add_argument("--file",   default=None, help="Local dataset file path")
    ap.add_argument("--model",  default=None, help="auto | model name | number")
    ap.add_argument("--target", default=None, help="Target column name")
    ap.add_argument("--mode",   default=None, help="auto | manual")
    ap.add_argument("--tune",   action="store_true")
    args = ap.parse_args()

    try:
        run_pipeline(
            user_idea        = args.idea,
            dataset_url      = args.url,
            dataset_file     = Path(args.file) if args.file else None,
            user_model_choice= args.model,
            target_column    = args.target,
            tune             = args.tune,
            mode             = args.mode,
        )
    except PipelineError as e:
        logger.error(f"Pipeline stopped: {e}")
        raise SystemExit(1)
