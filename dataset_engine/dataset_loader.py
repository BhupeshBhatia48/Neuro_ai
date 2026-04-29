"""
dataset_engine/dataset_loader.py
----------------------------------
Loads dataset files from disk into a pandas DataFrame.

BUG FIXED:
----------
OpenML's /data/v1/download/ endpoint returns ARFF format files,
NOT CSV. An ARFF file looks like:

  @relation diabetes
  @attribute age REAL
  @attribute class {tested_negative, tested_positive}
  @data
  6,148,72,...,tested_positive
  1,85,66,...,tested_negative

pandas.read_csv() fails on this with:
  ParserError: Expected 1 fields in line 20, saw 10

FIX: Added ARFF parser that extracts the @data section and
converts it to a proper DataFrame with correct column names.

Also added: better error messages showing the first few bytes
of the file so users understand what format was downloaded.
"""

from pathlib import Path
from typing import Optional
import io
import re

import pandas as pd
from loguru import logger


class DatasetLoader:

    def load(self, file_path: Path) -> pd.DataFrame:
        file_path = Path(file_path)
        suffix    = file_path.suffix.lower()
        logger.info(f"Loading: {file_path.name}")

        # Detect ARFF regardless of extension (OpenML returns .csv that is ARFF)
        if self._is_arff(file_path):
            logger.info(f"Detected ARFF format in {file_path.name} — parsing as ARFF")
            return self._load_arff(file_path)

        loaders = {
            ".csv":     lambda p: self._load_csv(p),        # smart separator detection
            ".tsv":     lambda p: pd.read_csv(p, sep="\t",  low_memory=False),
            ".json":    lambda p: pd.read_json(p),
            ".xlsx":    lambda p: pd.read_excel(p, engine="openpyxl"),
            ".xls":     lambda p: pd.read_excel(p),
            ".parquet": lambda p: pd.read_parquet(p),
            ".data":    lambda p: self._load_data_file(p),  # UCI .data files
            ".arff":    lambda p: self._load_arff(p),
        }

        if suffix not in loaders:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported: {list(loaders.keys())}"
            )

        try:
            df = loaders[suffix](file_path)
            logger.success(
                f"Loaded {len(df):,} rows x {len(df.columns)} columns"
            )
            return df
        except pd.errors.ParserError as exc:
            # Try ARFF fallback for files that look like CSV but are not
            logger.warning(
                f"CSV parsing failed ({exc}). "
                f"Trying ARFF parser as fallback..."
            )
            return self._load_arff(file_path)
        except Exception as exc:
            # Show first bytes of file to help diagnose format issues
            try:
                with open(file_path, "rb") as f:
                    preview = f.read(200).decode("utf-8", errors="replace")
                logger.error(
                    f"Failed to load {file_path.name}: {exc}\n"
                    f"File preview (first 200 bytes):\n{preview}"
                )
            except Exception:
                pass
            raise

    # ── Smart CSV loader ──────────────────────────────────────────────────────

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Smart CSV loader that automatically detects the separator.

        FIX for: "CSV parsing failed (Error tokenizing data. C error:
                  Expected 1 fields in line 13, saw 4)"

        This error happens when a file has .csv extension but is actually
        tab-separated (.tsv), semicolon-separated, or pipe-separated.
        We try separators in order: comma → tab → semicolon → pipe → whitespace.

        Each attempt validates that the result has >= 2 columns (a single-column
        result almost always means the wrong separator was used).
        """
        separators = [
            (",",    "comma"),
            ("\t",   "tab"),
            (";",    "semicolon"),
            ("|",    "pipe"),
        ]

        last_exc = None
        for sep, sep_name in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, low_memory=False)
                if df.shape[1] >= 2:
                    if sep != ",":
                        logger.info(
                            f"[Loader] '{file_path.name}' parsed as "
                            f"{sep_name}-separated (not comma)"
                        )
                    return df
                # Single column — wrong separator, try next
            except pd.errors.ParserError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                continue

        # Last resort: python engine with flexible separator detection
        # NOTE: python engine does NOT support low_memory — omit it
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
            logger.info(
                f"[Loader] '{file_path.name}' parsed with auto-detected separator"
            )
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last_exc = exc

        # If everything failed, raise the original error
        if last_exc is not None:
            raise last_exc
        raise ValueError(f"Could not parse '{file_path.name}' as CSV with any separator")

    # ── ARFF Parser ───────────────────────────────────────────────────────────

    def _is_arff(self, file_path: Path) -> bool:
        """Check if file is ARFF by reading first few lines."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(10):
                    line = f.readline().strip().lower()
                    if line.startswith("@relation") or line.startswith("@attribute"):
                        return True
        except Exception:
            pass
        return False

    def _load_arff(self, file_path: Path) -> pd.DataFrame:
        """
        Parse ARFF format files (returned by OpenML and WEKA).

        ARFF structure:
          @relation name
          @attribute col1 REAL
          @attribute col2 {val1, val2}
          @data
          1.0, 2.0, val1
          ...
        """
        attributes = []
        data_lines = []
        in_data    = False

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()

                # Skip empty lines and comments
                if not stripped or stripped.startswith("%"):
                    continue

                lower = stripped.lower()

                if lower.startswith("@attribute"):
                    # Parse attribute name
                    parts = stripped.split(None, 2)  # @attribute name type
                    if len(parts) >= 2:
                        attr_name = parts[1].strip("'\"")
                        attributes.append(attr_name)

                elif lower.startswith("@data"):
                    in_data = True

                elif in_data:
                    if stripped and not stripped.startswith("%"):
                        data_lines.append(stripped)

        if not attributes:
            raise ValueError(
                f"Could not parse ARFF file {file_path.name}: "
                f"no @attribute declarations found."
            )

        if not data_lines:
            raise ValueError(
                f"Could not parse ARFF file {file_path.name}: "
                f"no @data section found."
            )

        # Parse data section as CSV
        csv_content = "\n".join(data_lines)
        try:
            df = pd.read_csv(
                io.StringIO(csv_content),
                header=None,
                names=attributes,
                na_values=["?", "NA", "N/A", ""],
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse ARFF data section: {exc}"
            ) from exc

        # Strip quotes from string values
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip("'\" ").str.strip()

        logger.success(
            f"ARFF loaded: {len(df):,} rows x {len(df.columns)} columns | "
            f"columns: {df.columns.tolist()}"
        )
        return df

    # ── UCI .data file loader ─────────────────────────────────────────────────

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load UCI .data files (no header, space or comma separated).
        Tries comma first, then space/tab.
        """
        try:
            df = pd.read_csv(file_path, header=None, low_memory=False)
            if df.shape[1] > 1:
                # Generate column names: col_0, col_1, ..., class
                cols = [f"col_{i}" for i in range(df.shape[1] - 1)] + ["class"]
                if len(cols) == df.shape[1]:
                    df.columns = cols
                logger.success(
                    f"UCI .data loaded: {len(df):,} rows x {df.shape[1]} columns"
                )
                return df
        except Exception:
            pass

        # Try space/whitespace separated
        df = pd.read_csv(file_path, header=None, sep=r"\s+", low_memory=False)
        logger.success(
            f"UCI .data loaded (whitespace): {len(df):,} rows x {df.shape[1]} columns"
        )
        return df
