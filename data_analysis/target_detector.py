"""
data_analysis/target_detector.py
----------------------------------
Identifies the most likely target column.

BUG FIXED (CRITICAL):
---------------------
The LLM fallback was being called for medical/health datasets where the
idea contained clinical descriptions. OpenRouter was returning a full
medical diagnosis paragraph instead of a column name. This text then
entered the pipeline as a "target column name" — when the trainer tried
to use it, Ridge/sklearn tried to convert the paragraph to float and
crashed with: "could not convert string to float: 'The most likely
diagnosis...'"

ROOT CAUSE: llm_fallback() sends the user's idea + dataset columns to
the LLM and asks "which column is the target?". For medical ideas the
LLM sometimes ignores the instruction and writes a medical diagnosis
instead of a column name.

FIX 1: System prompt is now extremely strict — instructs the LLM to
        return ONLY 1-3 words (a column name), nothing else.
FIX 2: After getting LLM response, validate it is actually a column
        name in the dataset. If not, strip the response to first word
        and try again. If still not valid, return None.
FIX 3: max_tokens reduced from 50 to 20 — forces shorter response.
FIX 4: Added hard-coded fallback list of common target names so LLM
        is rarely needed for standard datasets.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger


TARGET_NAME_HINTS = [
    "target", "label", "class", "output", "result",
    "y", "outcome", "prediction", "response",
    "survived", "diagnosis", "fraud", "churn",
    "price", "salary", "score", "rating", "default",
    "status", "disease", "cancer", "death", "income",
    "species", "category", "type", "grade", "quality",
    "approved", "admit", "admitted", "pass", "fail",
    "buy", "sold", "purchased", "converted", "clicked",
    "spam", "ham", "sentiment", "positive", "negative",
    "win", "loss", "won", "nowin", "result",
]

CONFIDENCE_HIGH   = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW    = "low"


class AmbiguousTargetError(Exception):
    def __init__(self, candidates: List[str], message: str):
        self.candidates = candidates
        super().__init__(message)


class TargetDetector:

    def detect(
        self,
        df: pd.DataFrame,
        task_hint: str = "classification",
        raise_on_ambiguity: bool = True,
    ) -> Tuple[Optional[str], str]:

        cols       = df.columns.tolist()
        lower_cols = {c.lower(): c for c in cols}

        # 1. Exact name match -> HIGH confidence
        for hint in TARGET_NAME_HINTS:
            if hint in lower_cols:
                col = lower_cols[hint]
                logger.info(f"[TargetDetector] HIGH — name match: '{col}'")
                return col, CONFIDENCE_HIGH

        # Partial match — column name CONTAINS a hint word
        for hint in TARGET_NAME_HINTS:
            for col_lower, col_orig in lower_cols.items():
                if hint in col_lower and col_lower != hint:
                    logger.info(
                        f"[TargetDetector] HIGH — partial name match: "
                        f"'{col_orig}' contains '{hint}'"
                    )
                    return col_orig, CONFIDENCE_HIGH

        # 2. Last column heuristic -> MEDIUM confidence
        last_col     = cols[-1]
        last_nunique = df[last_col].nunique()

        if task_hint == "classification" and last_nunique <= 20:
            logger.info(
                f"[TargetDetector] MEDIUM — last col '{last_col}' "
                f"({last_nunique} unique values)"
            )
            return last_col, CONFIDENCE_MEDIUM

        if task_hint == "regression":
            if pd.api.types.is_numeric_dtype(df[last_col]):
                logger.info(
                    f"[TargetDetector] MEDIUM — last numeric col '{last_col}'"
                )
                return last_col, CONFIDENCE_MEDIUM

        # 3. Lowest cardinality column -> LOW confidence
        obj_cols   = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        candidates = sorted(obj_cols, key=lambda c: df[c].nunique())

        if candidates:
            best = candidates[0]
            logger.warning(
                f"[TargetDetector] LOW — best candidate: '{best}'. "
                f"Top: {candidates[:3]}"
            )
            if raise_on_ambiguity:
                raise AmbiguousTargetError(
                    candidates=candidates[:3],
                    message=(
                        f"Cannot confidently detect the target column. "
                        f"Top candidates: {candidates[:3]}. "
                        f"Use --target <column_name> to specify it."
                    ),
                )
            return best, CONFIDENCE_LOW

        # 4. Fallback: lowest-cardinality numeric column for regression
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            best = min(num_cols, key=lambda c: df[c].nunique())
            logger.warning(f"[TargetDetector] LOW — numeric fallback: '{best}'")
            if raise_on_ambiguity:
                raise AmbiguousTargetError(
                    candidates=num_cols[:3],
                    message=f"Cannot detect target. Candidates: {num_cols[:3]}",
                )
            return best, CONFIDENCE_LOW

        logger.error("[TargetDetector] Could not detect any target column.")
        return None, CONFIDENCE_LOW

    @staticmethod
    def llm_fallback(
        df: pd.DataFrame,
        task_hint: str,
        idea: str,
    ) -> Optional[str]:
        """
        FIX: Strict prompt that forces LLM to return ONLY a column name.
        Validates that response is actually a column in the dataset.
        max_tokens=20 prevents long medical/explanatory responses.
        """
        try:
            from llm_agent.openrouter_client import OpenRouterClient
            import json

            cols_summary = {
                col: {
                    "dtype":   str(df[col].dtype),
                    "nunique": int(df[col].nunique()),
                    "sample":  [
                        str(v)[:50]        # truncate long values
                        for v in df[col].dropna().head(3).tolist()
                    ],
                }
                for col in df.columns
            }

            # FIX: Extremely strict system prompt
            system_prompt = (
                "You are a column name selector. "
                "You MUST respond with ONLY a single column name from the list provided. "
                "Do NOT write any explanation, diagnosis, medical advice, or analysis. "
                "Do NOT write sentences. "
                "Respond with EXACTLY one column name from the dataset and nothing else."
            )

            user_prompt = (
                f"Dataset columns:\n{json.dumps(list(df.columns), indent=2)}\n\n"
                f"Task: {task_hint}\n"
                f"Project idea: {idea}\n\n"
                f"Which column is the TARGET (dependent variable)?\n"
                f"Reply with ONLY the exact column name. One word or phrase only."
            )

            client = OpenRouterClient()
            raw    = client.chat(
                messages       = [{"role": "user", "content": user_prompt}],
                system_prompt  = system_prompt,
                max_tokens     = 20,   # FIX: short enough to prevent paragraphs
                temperature    = 0.0,
            ).strip().strip('"\'').strip()

            logger.info(f"[TargetDetector] LLM raw response: '{raw[:100]}'")

            # FIX: Validate response is an actual column name
            if raw in df.columns:
                logger.success(f"[TargetDetector] LLM selected: '{raw}'")
                return raw

            # FIX: Try first word/phrase of response
            first_token = raw.split("\n")[0].split(".")[0].strip()
            if first_token in df.columns:
                logger.success(
                    f"[TargetDetector] LLM first token selected: '{first_token}'"
                )
                return first_token

            # FIX: Try fuzzy match — find column whose name appears in response
            raw_lower = raw.lower()
            for col in df.columns:
                if col.lower() in raw_lower:
                    logger.success(
                        f"[TargetDetector] LLM fuzzy match: '{col}' found in response"
                    )
                    return col

            logger.warning(
                f"[TargetDetector] LLM response '{raw[:80]}' is not a valid "
                f"column name. Valid columns: {df.columns.tolist()}"
            )
            return None

        except Exception as exc:
            logger.error(f"[TargetDetector] LLM fallback failed: {exc}")
            return None
