"""
llm_agent/keyword_extractor.py
--------------------------------
Semantic query generator — integrates ConceptExpander output.

Flow:
  ParsedIdea
      ↓
  ConceptExpander (semantic expansion)
      ↓
  KeywordExtractor (builds queries from core + expanded terms)
      ↓
  Search

Changes from previous version:
- Accepts optional ExpansionResult to build semantically rich queries
- Queries now include expanded/synonym terms when available
- Falls back to core-concept-only queries if no expansion available
- Still no hardcoded domain mappings
- Still prevents invalid task/term combinations
"""

from __future__ import annotations

import re
from typing import List, Optional

from loguru import logger


_PIPELINE_VERBS = {
    "predict", "classify", "detect", "build", "create", "make",
    "develop", "train", "model", "analyse", "analyze", "generate",
    "using", "based", "whether", "that", "will", "from", "with",
    "help", "want", "need", "find", "show", "give", "also", "like",
    "just", "some", "into", "this", "when", "what", "how", "why",
}

_TASK_TERMS = {
    "time_series":    ["time series", "historical data", "temporal", "forecasting"],
    "regression":     ["regression", "continuous", "numeric prediction"],
    "classification": ["classification", "labeled", "categories"],
    "nlp":            ["text dataset", "corpus", "natural language"],
    "clustering":     ["clustering", "unsupervised", "segments"],
}

_INVALID_PAIRS = {
    "time_series":    {"classification", "binary labels", "labeled categories"},
    "nlp":            {"regression", "numeric prediction", "continuous"},
    "classification": {"forecasting", "time series", "temporal"},
    "regression":     {"text dataset", "corpus", "labeled categories"},
}


def _extract_core_concepts(idea: str) -> List[str]:
    tokens = re.findall(r'\b[a-z]{3,}\b', idea.lower())
    core   = [t for t in tokens if t not in _PIPELINE_VERBS]
    seen, unique = set(), []
    for t in core:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _valid_combination(task_type: str, term: str) -> bool:
    blacklist = _INVALID_PAIRS.get(task_type, set())
    return not any(bad in term for bad in blacklist)


class KeywordExtractor:
    """
    Builds diverse, semantically rich search queries.

    When an ExpansionResult is provided, queries include both
    core concept terms AND semantically expanded terms, giving
    the search engine a much wider net to find relevant datasets.
    """

    def extract(
        self,
        parsed_idea,
        expansion=None,   # Optional[ExpansionResult] from ConceptExpander
    ) -> List[str]:
        """
        Parameters
        ----------
        parsed_idea : ParsedIdea object
        expansion   : Optional ExpansionResult from ConceptExpander
                      If provided, generates additional semantic queries
        """
        raw_idea  = getattr(parsed_idea, "raw_idea",  "") or ""
        task_type = getattr(parsed_idea, "task_type", "classification") or "classification"
        keywords  = getattr(parsed_idea, "keywords",  []) or []

        # Core concept extraction
        core = _extract_core_concepts(raw_idea)

        concept    = " ".join(core[:4])
        short      = " ".join(core[:2])
        very_short = core[0] if core else ""

        if not concept:
            concept = " ".join(str(k) for k in keywords[:4])
        if not concept:
            concept = raw_idea[:60]

        queries: List[str] = []

        # ── Block 1: Core concept queries (always generated) ─────────────────
        # Q1: most specific
        queries.append(f"{concept} dataset csv")

        # Q2: task-enriched
        for term in _TASK_TERMS.get(task_type, []):
            if _valid_combination(task_type, term):
                queries.append(f"{concept} {term}")
                break

        # Q3: shorter concept
        if short and short != concept:
            queries.append(f"{short} dataset")

        # Q4: very short + generic
        if very_short:
            queries.append(f"{very_short} data records csv")

        # ── Block 2: Semantic expansion queries (when expansion available) ────
        if expansion is not None:
            expanded   = expansion.expanded_terms   or []
            variations = expansion.semantic_variations or []

            # Q5-Q8: top expanded terms as standalone dataset queries
            for term in expanded[:3]:
                q = f"{term} dataset csv"
                if _valid_combination(task_type, q):
                    queries.append(q)

            # Q9-Q11: semantic variations from ConceptExpander
            for var in variations[:3]:
                if _valid_combination(task_type, var):
                    queries.append(var)

            # Q12: hybrid — first core + first expanded
            if core and expanded:
                hybrid = f"{core[0]} {expanded[0]} dataset"
                if _valid_combination(task_type, hybrid):
                    queries.append(hybrid)

            logger.info(
                f"[QueryGen] Expansion source={expansion.source} | "
                f"Added {len(expanded[:3])} expanded + "
                f"{len(variations[:3])} variation queries"
            )

        # ── Block 3: Raw idea fallback ────────────────────────────────────────
        idea_clean = re.sub(
            r'\b(' + '|'.join(_PIPELINE_VERBS) + r')\b', '', raw_idea.lower()
        ).strip()[:80]
        if idea_clean:
            queries.append(f"{idea_clean} dataset")

        # Deduplicate, remove empty, cap at 8
        seen:   set       = set()
        unique: List[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen and len(q) > 4:
                seen.add(q)
                unique.append(q)

        if not unique:
            unique = [f"{raw_idea[:60]} dataset csv"]

        logger.info(
            f"[QueryGen] Total queries={len(unique[:8])}: {unique[:8]}"
        )
        return unique[:8]
