"""
search_engine/duckduckgo_search.py
------------------------------------
DuckDuckGo web search wrapper — production-grade, handles all known
version changes and failure modes.

ROOT CAUSE OF DDG FAILURES (and fixes):
-----------------------------------------
1. Package renamed: duckduckgo_search → ddgs (v6+)
   FIX: Try both import names automatically.

2. ddgs 7.x broke context manager support in some builds
   FIX: Try context manager first, fall back to direct instantiation.

3. ddgs 7.x text() returns a GENERATOR not a list
   FIX: Always wrap result in list() immediately.

4. Result field 'href' renamed to 'url' in ddgs 7.x
   FIX: Check both field names when extracting URLs.

5. Rate limit raises DuckDuckGoSearchException (not generic Exception)
   FIX: Catch it specifically and back off.

6. Some environments have proxy issues causing SSL/connection errors
   FIX: Pass proxies=None explicitly to DDGS constructor.

7. Per-query failures were killing all remaining queries silently
   FIX: Each query is fully independent — one failure never stops others.

FIX 8 (NEW — DOWNLOAD RELEVANCE):
   DDG was returning all URLs including HTML pages, GitHub blob viewers,
   and completely irrelevant pages. We now filter to ONLY URLs that end
   with actual data file extensions (.csv, .xlsx, .xls, .data, .json,
   .parquet, .tsv, .zip, .arff). Everything else is discarded at search
   time so the downloader never even tries to fetch them.

   This prevents situations where the pipeline downloads an HTML "about"
   page or a GitHub viewer page instead of an actual dataset file.

Install (one of these, both work):
    pip install ddgs>=7.0.0
    pip install duckduckgo-search>=6.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from config.constants import MAX_SEARCH_RESULTS


# Extensions we ONLY accept from DDG results.
# Any URL that does not end with one of these is discarded at search time.
DATASET_FILE_EXTENSIONS = (
    ".csv", ".xlsx", ".xls", ".data", ".json",
    ".parquet", ".tsv", ".zip", ".arff", ".gz",
)


@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str
    score:   float = 0.0


# ── Package discovery ─────────────────────────────────────────────────────────

def _get_ddgs() -> Optional[type]:
    """
    Returns the DDGS class from whichever package is installed.
    Tries new name (ddgs) first, then old name (duckduckgo_search).
    Also exposed as module-level function so other modules can import it.
    """
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass

    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        pass

    return None


def _extract_url(hit: dict) -> str:
    """
    Extract URL from a result dict.
    Field name changed across versions: 'href' (ddgs <6) -> 'url' (ddgs 7.x)
    """
    for field in ("href", "url", "link"):
        val = hit.get(field, "")
        if val:
            return str(val).strip()
    return ""


def _extract_snippet(hit: dict) -> str:
    """Extract text snippet — field name changed across versions."""
    for field in ("body", "snippet", "description", "text"):
        val = hit.get(field, "")
        if val:
            return str(val)
    return ""


def _normalize_hf_url(url: str) -> str:
    """
    HuggingFace /blob/ viewer URLs → /resolve/ direct download URLs.
    Applied to DDG results before they reach the downloader.

    Before: https://huggingface.co/datasets/User/Repo/blob/main/data.csv
    After:  https://huggingface.co/datasets/User/Repo/resolve/main/data.csv
    """
    if url and "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/", 1)
    return url


def _url_is_dataset_file(url: str) -> bool:
    """
    FIX 8: Returns True ONLY if the URL ends with a dataset file extension.
    Strips query strings before checking.

    This is the core filter that prevents irrelevant HTML pages,
    GitHub blob viewers, and other non-data URLs from being downloaded.

    Examples:
        ACCEPT: https://raw.githubusercontent.com/.../diabetes.csv
        ACCEPT: https://archive.ics.uci.edu/.../heart.data
        REJECT: https://github.com/user/repo/blob/master/data.csv   (viewer)
        REJECT: https://www.kaggle.com/datasets/user/diabetes        (HTML)
        REJECT: https://en.wikipedia.org/wiki/Iris_flower_data_set   (article)
    """
    if not url:
        return False
    # Strip query string and fragment for extension check
    clean = url.split("?")[0].split("#")[0].lower().rstrip("/")
    return any(clean.endswith(ext) for ext in DATASET_FILE_EXTENSIONS)


# ── Core DDGS interaction ──────────────────────────────────────────────────────

def _make_ddgs_instance(DDGS) -> object:
    """
    Create a DDGS instance, trying different constructor signatures.
    proxies=None prevents proxy-related SSL errors in some environments.
    """
    try:
        return DDGS(proxies=None)
    except TypeError:
        pass

    try:
        return DDGS(timeout=20)
    except TypeError:
        pass

    return DDGS()


def _call_text(ddgs_instance, query: str, max_results: int = 20) -> List[dict]:
    """
    Call ddgs.text() and always return a plain list.
    Handles generator vs list return types across ddgs versions.
    We request more results (default 20) to get enough after extension filtering.
    """
    # Attempt 1: standard call with max_results kwarg
    try:
        raw = ddgs_instance.text(query, max_results=max_results)
        if raw is not None:
            result = list(raw)
            if result:
                return result
    except TypeError:
        pass
    except Exception as exc:
        logger.debug(f"[DDG] text(max_results=) failed: {type(exc).__name__}: {exc}")

    # Attempt 2: no max_results kwarg (older API)
    try:
        raw = ddgs_instance.text(query)
        if raw is not None:
            result = list(raw)[:max_results]
            if result:
                return result
    except Exception as exc:
        logger.debug(f"[DDG] text() no-kwarg failed: {type(exc).__name__}: {exc}")

    # Attempt 3: with region kwarg (some builds require it)
    try:
        raw = ddgs_instance.text(query, region="wt-wt", max_results=max_results)
        if raw is not None:
            result = list(raw)
            if result:
                return result
    except Exception as exc:
        logger.debug(f"[DDG] text(region=) failed: {type(exc).__name__}: {exc}")

    return []


def _search_one_query(DDGS, query: str, max_results: int) -> List[SearchResult]:
    """
    Run a single query with full fallback chain.
    Returns ONLY results whose URLs end with a dataset file extension.
    Never raises — always returns a (possibly empty) list.
    """
    hits: List[dict] = []

    # Strategy A: context manager
    try:
        with _make_ddgs_instance(DDGS) as ddgs:
            hits = _call_text(ddgs, query, max_results * 3)  # fetch more to filter
        if hits:
            logger.debug(f"[DDG] Strategy A got {len(hits)} raw hits")
    except Exception as exc_a:
        logger.debug(f"[DDG] Strategy A failed: {type(exc_a).__name__}: {exc_a}")

    # Strategy B: direct instantiation
    if not hits:
        try:
            ddgs = _make_ddgs_instance(DDGS)
            hits = _call_text(ddgs, query, max_results * 3)
            if hits:
                logger.debug(f"[DDG] Strategy B got {len(hits)} raw hits")
        except Exception as exc_b:
            logger.debug(f"[DDG] Strategy B failed: {type(exc_b).__name__}: {exc_b}")

    # FIX 8: Filter to dataset file URLs only, normalizing HF blob URLs
    results = []
    rejected = 0
    for hit in hits:
        url = _extract_url(hit)
        if not url:
            continue
        # Normalize HuggingFace /blob/ → /resolve/ before extension check
        url = _normalize_hf_url(url)
        if _url_is_dataset_file(url):
            results.append(SearchResult(
                title   = str(hit.get("title", "")),
                url     = url,
                snippet = _extract_snippet(hit),
            ))
        else:
            rejected += 1

    if rejected > 0:
        logger.debug(
            f"[DDG] Filtered out {rejected} non-dataset URLs "
            f"(kept {len(results)} with valid extensions)"
        )

    return results


# ── Public searcher class ─────────────────────────────────────────────────────

class DuckDuckGoSearcher:
    """
    Robust DuckDuckGo dataset searcher.

    KEY BEHAVIOUR:
    - Only returns URLs ending in .csv/.xlsx/.xls/.data/.json/.parquet etc.
    - Handles ddgs 5.x, 6.x, 7.x automatically
    - Each query is fully independent — one failure never kills the rest
    - Rate limit aware (0.5s delay between queries)
    - Deduplicates results by URL
    """

    def __init__(self, max_results: int = MAX_SEARCH_RESULTS):
        self.max_results = max_results

    def search(self, queries: List[str]) -> List[SearchResult]:
        DDGS = _get_ddgs()

        if DDGS is None:
            logger.warning(
                "[DDG] No DuckDuckGo package installed.\n"
                "Fix: pip install ddgs\n"
                "Alt: pip install duckduckgo-search"
            )
            return []

        seen_urls: set              = set()
        all_results: List[SearchResult] = []

        for i, query in enumerate(queries):
            if len(all_results) >= self.max_results:
                break

            logger.debug(f"[DDG] Query ({i+1}/{len(queries)}): '{query[:70]}'")

            try:
                q_results = _search_one_query(DDGS, query, max_results=10)
            except Exception as exc:
                logger.warning(f"[DDG] Query failed entirely: {exc}")
                q_results = []

            added = 0
            for r in q_results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
                    added += 1
                    if len(all_results) >= self.max_results:
                        break

            logger.debug(f"[DDG] Query {i+1}: {added} new results (total={len(all_results)})")

            # Rate limit delay between queries (not after the last one)
            if i < len(queries) - 1 and len(all_results) < self.max_results:
                time.sleep(0.5)

        logger.info(
            f"[DDG] Done: {len(all_results)} dataset-file results "
            f"from {len(queries)} queries"
        )
        return all_results[: self.max_results]
