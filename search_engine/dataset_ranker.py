"""
search_engine/dataset_ranker.py
---------------------------------
TASK 3: Relevance-first ranking — removes source bias.

Previous version favoured jbrownlee/seaborn sources with a static +0.10 bonus.
This version ranks purely on the score already assigned by ResultFilter
(which combines keyword relevance + format + domain signals).

Changes:
  - No source-based score boosting
  - KnownDatasets capped at 0.70 to give dynamic sources a fair chance
  - Diversity: max 2 results per repository owner
  - Junk sources hard-penalised (already done in ResultFilter)
  - Logs which source each candidate came from with its score
"""

from typing import List

from loguru import logger
from search_engine.duckduckgo_search import SearchResult


JUNK_SIGNALS = [
    "ai-watch", "jrc-data", "catalogue", "sdss", "skyserver",
    "astroquery", "export-jrc",
]


class DatasetRanker:

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rank by score (set by ResultFilter) with diversity deduplication.
        Ranking priority: relevance > quality > source
        """
        if not results:
            return []

        # Remove any junk that slipped through
        clean = [
            r for r in results
            if not any(j in (r.url or "").lower() for j in JUNK_SIGNALS)
        ]
        if not clean:
            clean = results  # fallback: use all if everything was junk

        # Cap KnownDatasets score so dynamic sources compete fairly
        for r in clean:
            if r.title and r.title.startswith("Known:"):
                r.score = min(r.score, 0.70)

        # Sort by score descending (relevance first)
        sorted_results = sorted(clean, key=lambda r: r.score, reverse=True)

        # Diversity: max 2 results from the same repository owner
        final:        List[SearchResult] = []
        owner_counts: dict = {}

        for r in sorted_results:
            owner = _repo_owner(r.url)
            if owner_counts.get(owner, 0) < 2:
                final.append(r)
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
            if len(final) >= self.top_n:
                break

        # Fallback: if diversity filter left too few, fill from remainder
        if len(final) < self.top_n:
            for r in sorted_results:
                if r not in final:
                    final.append(r)
                if len(final) >= self.top_n:
                    break

        # Log final ranking
        logger.info("[DatasetRanker] Final ranking:")
        for i, r in enumerate(final, 1):
            source = _source_label(r)
            logger.info(
                f"  [{i}] score={r.score:.3f} source={source:<20} "
                f"url={r.url[:55]}"
            )

        return final


def _repo_owner(url: str) -> str:
    """Extract repository owner for diversity filtering."""
    if not url:
        return "unknown"
    try:
        url_l = url.lower()
        if "githubusercontent.com" in url_l or "github.com" in url_l:
            parts = url_l.split("/")
            # https://raw.githubusercontent.com/{owner}/{repo}/...
            # https://github.com/{owner}/{repo}/...
            idx = next(
                (i for i, p in enumerate(parts)
                 if "github" in p), None
            )
            if idx is not None and idx + 1 < len(parts):
                return parts[idx + 1]
        if "openml" in url_l:
            return "openml"
        if "uci.edu" in url_l:
            return "uci"
        if "kaggle" in url_l:
            return "kaggle"
        if "huggingface" in url_l:
            return "huggingface"
        return url_l.split("/")[2] if url_l.count("/") >= 2 else url_l[:20]
    except Exception:
        return url[:20]


def _source_label(r: SearchResult) -> str:
    """Human-readable source label."""
    url = (r.url or "").lower()
    if r.title and r.title.startswith("Known:"):
        return "KnownDatasets"
    if "jbrownlee"     in url: return "jbrownlee"
    if "openml"        in url: return "OpenML"
    if "uci.edu"       in url: return "UCI"
    if "seaborn-data"  in url: return "seaborn"
    if "plotly"        in url: return "plotly"
    if "datasciencedojo" in url: return "datasciencedojo"
    if "githubusercontent" in url: return "GitHub"
    if "huggingface"   in url: return "HuggingFace"
    if "kaggle"        in url: return "Kaggle"
    return "Web"
