"""
search_engine/result_filter.py
--------------------------------
TASK 3 + TASK 5: Scoring-based filtering — no hard rejections.

Changes:
  - Results scored on RELEVANCE first, source trust second
  - Keyword match is a SCORING signal, NOT a hard filter
  - Weak matches get a penalty, not removal
  - Source bias removed — jbrownlee/seaborn not favoured over relevance
  - Junk sources penalised heavily
  - Every result that passes basic sanity gets a score
  - No result is dropped just because it has low keyword match
"""

from typing import List, Optional

from loguru import logger
from search_engine.duckduckgo_search import SearchResult
from config.constants import SUPPORTED_FORMATS


# Domains that confirm a result is dataset-hosting
DATASET_DOMAINS = [
    "kaggle.com", "github.com", "openml.org",
    "archive.ics.uci.edu", "data.world",
    "huggingface.co", "raw.githubusercontent.com",
    "zenodo.org", "figshare.com", "datahub.io",
]

DIRECT_EXTENSIONS = (".csv", ".json", ".xlsx", ".xls", ".parquet",
                     ".zip", ".tsv", ".data")

# Sources known to return irrelevant junk
JUNK_SIGNALS = [
    "ai-watch", "jrc-data", "jrc-catalogue",
    "sdss.org", "skyserver", "astroquery",
    "export-jrc", "catalogue",
]


class ResultFilter:
    """
    TASK 3 + TASK 5: Score-based filter.
    Uses keyword match + source + format signals as additive scores.
    NEVER hard-rejects based on keyword count alone.
    """

    def __init__(self, idea_keywords: Optional[List[str]] = None):
        """
        idea_keywords: optional list of core concept words from the idea.
        Used for relevance scoring (not hard filtering).
        """
        self.idea_keywords = idea_keywords or []

    def filter(
        self,
        results: List[SearchResult],
        idea_keywords: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        # Allow idea_keywords to be passed directly (from ConceptExpander)
        # This includes both core terms AND semantically expanded terms
        if idea_keywords:
            self.idea_keywords = idea_keywords
        scored = []
        for r in results:
            score, keep = self._score(r)
            if keep:
                r.score = score
                scored.append(r)

        # Log score distribution
        if scored:
            top = sorted(scored, key=lambda x: x.score, reverse=True)[:3]
            for r in top:
                logger.info(
                    f"  [Filter] score={r.score:.2f} url={r.url[:60]}"
                )

        return scored

    def _score(self, r: SearchResult):
        """
        Returns (score, keep_bool).
        keep=False only for known junk sources.
        All other results kept with varying scores.
        """
        url     = (r.url     or "").lower()
        title   = (r.title   or "").lower()
        snippet = (r.snippet or "").lower()
        combined_text = f"{title} {snippet} {url}"

        # ── Hard reject: known junk sources ──────────────────────────────────
        for junk in JUNK_SIGNALS:
            if junk in url:
                return 0.0, False

        score = 0.0

        # ── Dataset presence signal ───────────────────────────────────────────
        # Result must mention dataset OR be a direct file link
        is_direct    = any(url.endswith(ext) for ext in DIRECT_EXTENSIONS)
        is_data_page = (
            "dataset" in combined_text or
            "data" in combined_text or
            is_direct
        )
        if not is_data_page:
            return 0.0, False   # Not a dataset result at all

        # ── Keyword + expanded term relevance scoring ───────────────────────
        # idea_keywords now includes ConceptExpander output:
        # e.g., for 'parkinson tremor': ['parkinson', 'tremor', 'gait',
        #         'movement', 'motor', 'accelerometer', ...]
        # A dataset matching 'gait' gets a moderate boost even if
        # 'parkinson' doesn't appear in its URL/title/snippet.
        if self.idea_keywords:
            matches = sum(1 for kw in self.idea_keywords if kw in combined_text)
            if matches >= 3:
                score += 0.50    # multiple expanded terms — high confidence
            elif matches == 2:
                score += 0.40    # two matches — good relevance
            elif matches == 1:
                score += 0.15    # single match (may be expanded term) — small boost
            else:
                # Partial-word fallback — catches 'trembl' for 'trembling'
                partial = sum(
                    1 for kw in self.idea_keywords
                    if len(kw) >= 4 and any(kw[:4] in t for t in combined_text.split())
                )
                if partial >= 2:
                    score += 0.10   # partial semantic matches
                elif partial == 1:
                    score += 0.05
                else:
                    score -= 0.05   # no match at all — small penalty, not rejection

        # ── Direct file link bonus ────────────────────────────────────────────
        if is_direct:
            score += 0.30

        # ── Dataset domain presence ───────────────────────────────────────────
        if any(d in url for d in DATASET_DOMAINS):
            score += 0.20

        # ── Format keyword bonus ──────────────────────────────────────────────
        if any(fmt.lstrip(".") in snippet for fmt in SUPPORTED_FORMATS):
            score += 0.10

        # ── KnownDatasets cap — prevent dominance ────────────────────────────
        # KnownDatasets items have title "Known: ..."
        if r.title and r.title.startswith("Known:"):
            score = min(score, 0.70)

        # Ensure score is non-negative (fallback datasets still usable)
        score = max(0.05, round(score, 3))

        return score, True
