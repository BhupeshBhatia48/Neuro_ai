"""
search_engine/concept_expander.py
------------------------------------
Semantic concept expansion layer.

Transforms keyword-only search into meaning-based retrieval by
expanding the user's core idea into related concepts, synonyms,
and underlying signal representations.

Design decisions:
-----------------
1. NO hardcoded domain dictionaries (e.g., Parkinson → gait).
2. Two-level expansion:
     a) Rule-based expansion using WordNet synonyms (if available)
        This is zero-cost, offline, and language-grounded.
     b) LLM expansion via OpenRouter (if USE_LLM=True and available)
        This understands domain semantics deeply.
     c) Structural fallback: linguistic morphology rules
        Works even when WordNet and LLM are both unavailable.
3. Outputs are LIMITED to top 8 expanded terms to avoid noise.
4. Integration: ConceptExpander output is fed into KeywordExtractor
   which builds the final search queries.
5. Scoring: ResultFilter gets expanded terms for a moderate
   bonus score — not enough to override direct matches.

Example (NOT hardcoded — dynamically generated):
    Input: "predict patients with parkinsons tremor"
    WordNet synonyms of 'tremor': ['shaking', 'quivering', 'vibration']
    LLM expansion: ['gait', 'movement disorder', 'motor', 'accelerometer']
    Structural: ['tremor data', 'tremor signals', 'tremor measurements']
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


# ── Structural suffix patterns applied to core terms ─────────────────────────
# These generate search-ready variations without any domain knowledge.
_SIGNAL_SUFFIXES = ["signals", "measurements", "data", "records", "readings"]
_DATASET_SUFFIXES = ["dataset", "dataset csv", "data csv"]

# Generic words that should not be expanded (too broad to be useful)
_EXPANSION_STOPWORDS = {
    "data", "dataset", "patient", "patients", "people", "person",
    "predict", "classify", "model", "system", "analysis", "problem",
    "using", "based", "with", "from", "that", "this", "which",
}


@dataclass
class ExpansionResult:
    """Output of ConceptExpander.expand()"""
    core_terms:          List[str]   # cleaned original concept words
    expanded_terms:      List[str]   # semantically related terms
    semantic_variations: List[str]   # search-ready phrase variations
    source:              str = "none"  # "wordnet" | "llm" | "structural"


class ConceptExpander:
    """
    Expands user idea into semantically related concepts.

    Expansion hierarchy (best to worst):
    1. LLM (USE_LLM=True)  — deep semantic understanding
    2. WordNet synonyms    — linguistically grounded, offline
    3. Structural rules    — morphological variations, always works
    """

    MAX_EXPANDED     = 8   # max expanded terms to avoid noise
    MAX_VARIATIONS   = 6   # max search-ready phrase variations

    def expand(
        self,
        idea: str,
        task_type: str  = "classification",
        domain: str     = "general",
    ) -> ExpansionResult:
        """
        Parameters
        ----------
        idea      : Raw user idea string
        task_type : classification | regression | nlp | time_series
        domain    : detected domain (general, medical, finance, etc.)

        Returns
        -------
        ExpansionResult with core_terms, expanded_terms, semantic_variations
        """
        # Step 1: Extract core terms from idea
        core_terms = self._extract_core_terms(idea)

        logger.info(
            f"[ConceptExpander] idea='{idea[:60]}' "
            f"core_terms={core_terms}"
        )

        # Step 2: Try expansion methods in priority order
        expanded: List[str] = []
        source               = "structural"

        # Method A: LLM expansion (richest — understands domain semantics)
        llm_terms = self._try_llm_expansion(idea, core_terms, task_type, domain)
        if llm_terms:
            expanded = llm_terms
            source   = "llm"

        # Method B: WordNet synonyms (offline, linguistically grounded)
        if not expanded:
            wn_terms = self._try_wordnet_expansion(core_terms)
            if wn_terms:
                expanded = wn_terms
                source   = "wordnet"

        # Method C: Structural rules (always available, language patterns)
        if not expanded:
            expanded = self._structural_expansion(core_terms, task_type)
            source   = "structural"

        # Step 3: Deduplicate, remove core terms already in expanded
        expanded = self._clean_expanded(expanded, core_terms)

        # Step 4: Generate search-ready phrase variations
        variations = self._generate_variations(core_terms, expanded)

        result = ExpansionResult(
            core_terms          = core_terms,
            expanded_terms      = expanded[:self.MAX_EXPANDED],
            semantic_variations = variations[:self.MAX_VARIATIONS],
            source              = source,
        )

        logger.info(
            f"[ConceptExpander] source={source} | "
            f"expanded={result.expanded_terms} | "
            f"variations={result.semantic_variations}"
        )
        return result

    # ── Step 1: Core term extraction ─────────────────────────────────────────

    def _extract_core_terms(self, idea: str) -> List[str]:
        """
        Extract meaningful concept words from the idea.
        Strips stopwords and short tokens.
        """
        PIPELINE_VERBS = {
            "predict", "classify", "detect", "build", "create", "make",
            "develop", "train", "analyze", "analyse", "generate", "using",
            "based", "whether", "from", "with", "help", "find", "show",
            "give", "want", "need", "like", "also", "just", "into",
            "will", "that", "this", "when", "what", "how", "why",
        }
        tokens = re.findall(r'\b[a-z]{3,}\b', idea.lower())
        core   = [
            t for t in tokens
            if t not in PIPELINE_VERBS and t not in _EXPANSION_STOPWORDS
        ]
        # Deduplicate preserving order
        seen, unique = set(), []
        for t in core:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:6]   # top 6 most meaningful

    # ── Step 2a: LLM expansion ────────────────────────────────────────────────

    def _try_llm_expansion(
        self,
        idea: str,
        core_terms: List[str],
        task_type: str,
        domain: str,
    ) -> List[str]:
        """
        Use OpenRouter LLM to expand concepts semantically.
        Returns empty list on any failure — non-blocking.
        """
        try:
            from config.settings import USE_LLM
            if not USE_LLM:
                return []

            from llm_agent.openrouter_client import OpenRouterClient

            system_prompt = (
                "You are a dataset search expert. "
                "Given a machine learning idea, generate related concept terms "
                "that would help find relevant datasets. "
                "Focus on: underlying signals, measurements, phenomena, "
                "synonyms, and related technical terms. "
                "DO NOT include generic words like 'data', 'dataset', 'csv'. "
                "Reply with ONLY a comma-separated list of 5-8 single words "
                "or short phrases. Nothing else."
            )
            user_prompt = (
                f"ML idea: {idea}\n"
                f"Task type: {task_type}\n"
                f"Core terms: {', '.join(core_terms)}\n\n"
                f"Generate 5-8 related concept terms for dataset search. "
                f"Think about: underlying signals, measurements, synonyms, "
                f"related phenomena. "
                f"Reply with comma-separated terms only."
            )

            client   = OpenRouterClient()
            response = client.chat(
                messages      = [{"role": "user", "content": user_prompt}],
                system_prompt = system_prompt,
                max_tokens    = 80,
                temperature   = 0.3,
            ).strip()

            # Parse comma-separated response
            raw_terms = [t.strip().lower() for t in response.split(",")]
            valid     = [
                t for t in raw_terms
                if 2 < len(t) < 40
                and t not in _EXPANSION_STOPWORDS
                and not any(bad in t for bad in ["dataset", "csv", "the ", "and "])
            ]

            if valid:
                logger.info(f"[ConceptExpander] LLM expanded: {valid[:8]}")
            return valid[:8]

        except Exception as exc:
            logger.debug(f"[ConceptExpander] LLM expansion failed: {exc}")
            return []

    # ── Step 2b: WordNet expansion ────────────────────────────────────────────

    def _try_wordnet_expansion(self, core_terms: List[str]) -> List[str]:
        """
        Use NLTK WordNet to find synonyms and hypernyms.
        Returns empty list if NLTK is not installed.
        No hardcoding — pure linguistic lookup.
        """
        try:
            import nltk
            from nltk.corpus import wordnet as wn

            # Ensure WordNet data is available
            try:
                wn.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            expanded = []
            for term in core_terms[:4]:   # expand top 4 terms only
                synsets = wn.synsets(term)
                for syn in synsets[:2]:    # top 2 synsets per term
                    # Synonyms (lemma names)
                    for lemma in syn.lemmas()[:3]:
                        word = lemma.name().replace("_", " ").lower()
                        if (word != term
                                and len(word) > 2
                                and word not in _EXPANSION_STOPWORDS):
                            expanded.append(word)
                    # Hypernyms (broader concepts)
                    for hyper in syn.hypernyms()[:1]:
                        for lemma in hyper.lemmas()[:2]:
                            word = lemma.name().replace("_", " ").lower()
                            if (word != term
                                    and len(word) > 2
                                    and word not in _EXPANSION_STOPWORDS):
                                expanded.append(word)

            # Deduplicate
            seen, unique = set(), []
            for t in expanded:
                if t not in seen and t not in core_terms:
                    seen.add(t)
                    unique.append(t)

            if unique:
                logger.info(
                    f"[ConceptExpander] WordNet expanded: {unique[:8]}"
                )
            return unique[:8]

        except ImportError:
            logger.debug("[ConceptExpander] NLTK not installed — skipping WordNet")
            return []
        except Exception as exc:
            logger.debug(f"[ConceptExpander] WordNet failed: {exc}")
            return []

    # ── Step 2c: Structural expansion ────────────────────────────────────────

    def _structural_expansion(
        self,
        core_terms: List[str],
        task_type: str,
    ) -> List[str]:
        """
        Morphological and task-aware expansion.
        Uses linguistic patterns (not domain knowledge) to generate
        related search terms from the core words.

        Techniques used:
        - Noun → adjective form (tremor → trembling, motion → moving)
        - Noun → verb form (analysis → analyze)
        - Task-type measurement terms
        - Signal/sensor terminology for numerical data
        """
        expanded = []

        # Morphological transforms
        # (These are generic linguistic patterns, not domain rules)
        SUFFIX_MAP = {
            # Nominalization reverse: -ion → -e (detection → detect)
            "tion": "",    # "detection" → "detect"
            "sion": "",    # "regression" → "regress"
            "ment": "",    # "movement" → "move"
            "ness": "",    # "ickness" → "ick" (usually useless, skip)
            "ity":  "",    # "density" → "dens"
            "ing":  "",    # "trembling" → "trembl"
            "ance": "ant", # "resistance" → "resistant"
        }

        for term in core_terms[:4]:
            # Morphological variant
            for suffix, replacement in SUFFIX_MAP.items():
                if term.endswith(suffix) and len(term) > len(suffix) + 2:
                    stem = term[:-len(suffix)] + replacement
                    if len(stem) > 2 and stem not in _EXPANSION_STOPWORDS:
                        expanded.append(stem)
                    break

            # Task-type measurement variants
            if task_type in ("classification", "regression"):
                expanded.append(f"{term} measurement")
                expanded.append(f"{term} feature")
            elif task_type == "time_series":
                expanded.append(f"{term} signal")
                expanded.append(f"{term} sensor")
            elif task_type == "nlp":
                expanded.append(f"{term} text")
                expanded.append(f"{term} review")

        # Deduplicate
        seen, unique = set(), []
        for t in expanded:
            t = t.strip()
            if t and t not in seen and t not in core_terms:
                seen.add(t)
                unique.append(t)

        logger.info(
            f"[ConceptExpander] Structural expanded: {unique[:8]}"
        )
        return unique[:8]

    # ── Step 3: Clean expanded terms ─────────────────────────────────────────

    def _clean_expanded(
        self,
        expanded: List[str],
        core_terms: List[str],
    ) -> List[str]:
        """Remove duplicates, core terms already present, and generic words."""
        core_set = set(core_terms)
        seen, clean = set(), []
        for t in expanded:
            t = t.strip().lower()
            if (t
                    and t not in seen
                    and t not in core_set
                    and t not in _EXPANSION_STOPWORDS
                    and len(t) > 2):
                seen.add(t)
                clean.append(t)
        return clean

    # ── Step 4: Generate search-ready variations ─────────────────────────────

    def _generate_variations(
        self,
        core_terms: List[str],
        expanded_terms: List[str],
    ) -> List[str]:
        """
        Build search-ready phrase variations from core + top expanded terms.
        Combines terms generically — no domain knowledge required.
        """
        variations = []

        # Core term variations
        if core_terms:
            concept = " ".join(core_terms[:3])
            variations.append(f"{concept} dataset csv")
            variations.append(f"{concept} data")

        # Top expanded terms as standalone search phrases
        for term in expanded_terms[:3]:
            variations.append(f"{term} dataset csv")

        # Hybrid: first core + first expanded
        if core_terms and expanded_terms:
            hybrid = f"{core_terms[0]} {expanded_terms[0]}"
            variations.append(f"{hybrid} dataset")

        # Signal variations for top expanded terms
        for term in expanded_terms[:2]:
            variations.append(f"{term} data records")

        # Deduplicate
        seen, unique = set(), []
        for v in variations:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                unique.append(v)

        return unique
