"""
dataset_engine/dataset_downloader.py
--------------------------------------
Downloads dataset files with security guardrails and smart HTML handling.

TASK 2: Smart HTML page handling.
When a URL returns HTML (e.g. UCI ML Repository pages), instead of
immediately rejecting it, we now:
  1. Detect HTML content
  2. Parse the page to extract direct dataset file links
  3. Retry the download using the extracted link
  4. UCI-specific: look for /machine-learning-databases/ links
  5. Fallback: if no file found in the HTML, skip gracefully

This means UCI archive pages can now be successfully handled.
"""

import hashlib
import re
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from loguru import logger

from config.settings import (
    DATASETS_DIR,
    ALLOWED_DOWNLOAD_DOMAINS,
    ALLOWED_DOWNLOAD_EXTENSIONS,
    MAX_DOWNLOAD_SIZE_BYTES,
)
from config.constants import SUPPORTED_FORMATS


class DatasetDownloadError(Exception):
    pass


class DatasetDownloader:

    CHUNK_SIZE = 1024 * 64  # 64 KB

    HTML_SIGNATURES = [
        b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML",
        b"<head>",    b"<HEAD>",    b"<body>", b"<BODY>",
    ]

    # File extensions that indicate actual dataset files (not pages)
    DATASET_EXTENSIONS = (".csv", ".data", ".xls", ".xlsx", ".json",
                          ".parquet", ".tsv", ".zip", ".gz")

    def download(self, url: str, name: str = "dataset") -> Optional[Path]:
        """
        Download a dataset from a URL.
        If URL returns HTML, attempt to extract a direct file link from the page.
        """
        try:
            # ── Normalize URL FIRST — fix known bad URL patterns ─────────────
            # HuggingFace: /blob/ links are HTML viewers, not direct downloads.
            # Replace /blob/ with /resolve/ to get the raw file download URL.
            # Example:
            #   .../blob/main/parkinsons.csv  → .../resolve/main/parkinsons.csv
            url = self._normalize_url(url)

            if name == "dataset":
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                name = f"dataset_{url_hash}"

            self._validate_domain(url)
            self._validate_extension(url)

            logger.info(f"Downloading: {url}")
            response = requests.get(
                url, stream=True, timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NeuroAI/1.0)"},
                allow_redirects=True,
            )
            response.raise_for_status()

            # Size check from header
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
                raise DatasetDownloadError(
                    f"File too large: {int(content_length)/1e6:.1f} MB"
                )

            content_type = response.headers.get("Content-Type", "").lower()

            # TASK 2: HTML response — try to extract dataset link
            if "text/html" in content_type or "application/xhtml" in content_type:
                logger.info(
                    f"URL returned HTML — parsing page for dataset links: {url}"
                )
                extracted = self._extract_dataset_link_from_html(
                    response.text, url
                )
                if extracted:
                    # Normalize the extracted URL (e.g. HF /blob/ → /resolve/)
                    extracted = self._normalize_url(extracted)
                    # Guard against infinite loop: don't re-download same URL
                    if extracted == url:
                        raise DatasetDownloadError(
                            f"Extracted URL is same as original — cannot resolve: {url}"
                        )
                    logger.info(f"Extracted dataset link: {extracted}")
                    return self.download(extracted, name=name)
                else:
                    raise DatasetDownloadError(
                        f"URL returned an HTML page with no extractable "
                        f"dataset file links: {url}"
                    )

            ext       = self._detect_extension(url, response)
            dest_path = DATASETS_DIR / f"{name}{ext}"

            downloaded  = 0
            first_chunk = True

            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                    if not chunk:
                        continue

                    # First chunk HTML detection (fallback for wrong content-type)
                    if first_chunk:
                        first_chunk = False
                        chunk_head  = chunk[:200]
                        for sig in self.HTML_SIGNATURES:
                            if chunk_head.upper().startswith(sig.upper()):
                                dest_path.unlink(missing_ok=True)
                                # Try HTML extraction on the partial content
                                try:
                                    text = chunk_head.decode("utf-8", errors="ignore")
                                    extracted = self._extract_dataset_link_from_html(
                                        text, url
                                    )
                                    if extracted:
                                        logger.info(
                                            f"Extracted link from partial HTML: {extracted}"
                                        )
                                        return self.download(extracted, name=name)
                                except Exception:
                                    pass
                                raise DatasetDownloadError(
                                    f"File is an HTML page (content-type lied). URL: {url}"
                                )

                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_SIZE_BYTES:
                        dest_path.unlink(missing_ok=True)
                        raise DatasetDownloadError(
                            f"Exceeded {MAX_DOWNLOAD_SIZE_BYTES/1e6:.0f} MB limit."
                        )
                    fh.write(chunk)

            if downloaded < 100:
                dest_path.unlink(missing_ok=True)
                raise DatasetDownloadError(
                    f"File too small ({downloaded} bytes) — likely empty or redirect."
                )

            logger.success(
                f"Downloaded: {dest_path.name} ({downloaded/1e6:.2f} MB)"
            )

            if ext == ".zip":
                dest_path = self._extract_zip(dest_path)

            return dest_path

        except DatasetDownloadError as exc:
            logger.error(f"Download rejected: {exc}")
            return None
        except requests.HTTPError as exc:
            logger.error(f"HTTP {exc.response.status_code} for {url}")
            return None
        except Exception as exc:
            logger.error(f"Download failed: {exc}")
            return None

    # ── URL normalization ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalise known bad URL patterns into direct-download URLs.

        HuggingFace:
          /blob/  links are HTML viewer pages — they show a file browser,
          not the raw file. Replace with /resolve/ to get the direct CDN URL.

          Before: https://huggingface.co/datasets/User/Repo/blob/main/data.csv
          After:  https://huggingface.co/datasets/User/Repo/resolve/main/data.csv

          The /resolve/ endpoint redirects to the CDN and returns the raw bytes.

        GitHub (already handled by existing raw.githubusercontent logic, but for safety):
          /blob/ viewer URLs are NOT converted here — the downloader's HTML
          extractor already handles these by finding the raw download link in the page.
        """
        if not url:
            return url

        # HuggingFace: /blob/ → /resolve/
        if "huggingface.co" in url and "/blob/" in url:
            fixed = url.replace("/blob/", "/resolve/", 1)
            logger.info(
                f"[Downloader] HuggingFace URL normalised: "
                f"/blob/ → /resolve/"
            )
            return fixed

        return url

    # ── TASK 2: HTML parsing for dataset file extraction ─────────────────────

    def _extract_dataset_link_from_html(
        self,
        html: str,
        page_url: str,
    ) -> Optional[str]:
        """
        Parses an HTML page and extracts the most likely direct dataset
        file link.

        Priority:
          1. UCI ML Repository file links (/machine-learning-databases/)
          2. Any href ending in a dataset extension
          3. Links containing 'download' + dataset extension pattern
        """
        # Build base URL for resolving relative links
        parsed   = urlparse(page_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Extract all href values
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        hrefs += re.findall(r'href=([^\s>]+)', html, re.IGNORECASE)

        candidates: List[str] = []

        for href in hrefs:
            href = href.strip()

            # Skip anchors, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:")):
                continue

            # Resolve relative URLs
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = base_url + href
            else:
                full_url = urljoin(page_url, href)

            # UCI-specific: /machine-learning-databases/ paths are dataset files
            if "machine-learning-databases" in full_url:
                for ext in self.DATASET_EXTENSIONS:
                    if full_url.lower().endswith(ext):
                        candidates.insert(0, full_url)  # high priority
                        break
                else:
                    # It's a directory — try common file names
                    candidates.append(full_url)

            # Any link ending with a dataset extension
            elif any(
                full_url.lower().endswith(ext)
                for ext in self.DATASET_EXTENSIONS
            ):
                candidates.append(full_url)

            # Links with 'download' in them
            elif "download" in full_url.lower() and any(
                ext in full_url.lower() for ext in self.DATASET_EXTENSIONS
            ):
                candidates.append(full_url)

        if not candidates:
            logger.warning(
                f"No dataset file links found in HTML page: {page_url}"
            )
            return None

        # Prefer .csv and .data over others
        for ext in (".csv", ".data", ".tsv", ".xlsx"):
            for c in candidates:
                if c.lower().endswith(ext):
                    return c

        return candidates[0]

    # ── Security / Extension helpers ──────────────────────────────────────────

    def _validate_domain(self, url: str) -> None:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ""

        allowed = list(set(ALLOWED_DOWNLOAD_DOMAINS + [
            "raw.githubusercontent.com",
            "github.com",
            "openml.org",
            "api.openml.org",
            "archive.ics.uci.edu",
            "data.world",
            "huggingface.co",
            "datasets-server.huggingface.co",
            "storage.googleapis.com",
            "s3.amazonaws.com",
            "kaggle.com",
            "drive.google.com",
        ]))

        if not any(
            hostname == d or hostname.endswith("." + d)
            for d in allowed
        ):
            raise DatasetDownloadError(
                f"Domain '{hostname}' is not in the allowed list."
            )

    def _validate_extension(self, url: str) -> None:
        url_path     = url.split("?")[0].lower()
        last_segment = url_path.split("/")[-1]
        if "." in last_segment:
            if not any(
                url_path.endswith(ext)
                for ext in ALLOWED_DOWNLOAD_EXTENSIONS
            ):
                raise DatasetDownloadError(
                    f"File extension not allowed: {url}"
                )

    def _detect_extension(
        self, url: str, response: requests.Response
    ) -> str:
        url_path = url.split("?")[0].lower()
        for fmt in SUPPORTED_FORMATS:
            if url_path.endswith(fmt):
                return fmt

        cd = response.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            fname = cd.split("filename=")[-1].strip().strip('"')
            for fmt in SUPPORTED_FORMATS:
                if fname.lower().endswith(fmt):
                    return fmt

        ct = response.headers.get("Content-Type", "").lower()
        if "csv"      in ct: return ".csv"
        if "json"     in ct: return ".json"
        if "zip"      in ct: return ".zip"
        if "parquet"  in ct: return ".parquet"
        if "excel" in ct or "spreadsheet" in ct: return ".xlsx"
        return ".csv"

    def _extract_zip(self, zip_path: Path) -> Path:
        extract_dir = zip_path.parent / zip_path.stem
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        for fmt in SUPPORTED_FORMATS:
            found = list(extract_dir.rglob(f"*{fmt}"))
            if found:
                logger.info(f"Extracted: {found[0].name}")
                return found[0]
        raise FileNotFoundError(
            f"No supported file found inside ZIP: {zip_path}"
        )
