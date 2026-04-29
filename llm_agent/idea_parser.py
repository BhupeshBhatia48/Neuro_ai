"""
llm_agent/idea_parser.py
-------------------------
Parses a free-text project idea into a structured ParsedIdea.

Issue #1  : Default model is openrouter/auto (free)
Issue #1  : Rule-based fallback when OpenRouter API fails
Issue #9  : LLM is optional — if USE_LLM=false, skips API call entirely
            and runs pure rule-based parsing from keyword matching.

Fallback chain
--------------
1. OpenRouter LLM call (if USE_LLM=true and key is set)
2. Rule-based keyword parser (always available, no API needed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from config.settings import LLM_MODEL, LLM_MAX_TOKENS, USE_LLM, OPENROUTER_API_KEY
from config.constants import SUPPORTED_TASKS


# -- Data Model ----------------------------------------------------------------
@dataclass
class ParsedIdea:
    raw_idea:    str
    task_type:   str
    domain:      str
    keywords:    List[str]
    description: str = ""
    parsed_by:   str = "llm"   # "llm" | "rules"

    def size_bucket(self, num_rows: int) -> str:
        from config.constants import SMALL_DATASET_THRESHOLD, MEDIUM_DATASET_THRESHOLD
        if num_rows < SMALL_DATASET_THRESHOLD:   return "small"
        elif num_rows < MEDIUM_DATASET_THRESHOLD: return "medium"
        return "large"


# -- Rule-based fallback parser ------------------------------------------------

# Task-type keyword triggers (ordered, first match wins)
_TASK_RULES = [
    ("time_series",      ["time series", "forecast", "temporal", "arima", "lstm", "sequence", "trend"]),
    ("computer_vision",  ["image", "vision", "cnn", "resnet", "object detect", "segmentation", "photo"]),
    ("nlp",              ["text", "nlp", "sentiment", "translation", "language", "bert", "gpt", "ner"]),
    ("clustering",       ["cluster", "segment", "group", "unsupervised", "kmeans", "dbscan"]),
    ("regression",       ["predict price", "estimate", "forecast value", "regression", "house price",
                          "salary", "revenue", "amount", "continuous"]),
    ("classification",   ["classify", "classification", "detect", "diagnose", "predict whether",
                          "churn", "fraud", "spam", "survive", "default", "binary"]),
]

# Domain keyword mapping
_DOMAIN_RULES = {
    "healthcare":  ["disease", "medical", "patient", "hospital", "cancer", "diagnosis", "health"],
    "finance":     ["stock", "price", "fraud", "loan", "bank", "credit", "financial", "revenue"],
    "ecommerce":   ["product", "sales", "customer", "churn", "retail", "order", "cart"],
    "nlp":         ["text", "sentiment", "review", "language", "document", "tweet"],
    "climate":     ["weather", "climate", "temperature", "rain", "flood", "sensor"],
    "transport":   ["taxi", "flight", "traffic", "vehicle", "uber", "delivery"],
    "sports":      ["football", "cricket", "nba", "sports", "player", "game"],
}


def _rule_based_parse(user_idea: str) -> ParsedIdea:
    """
    Pure rule-based parser — no API call, zero dependencies.
    Extracts task_type, domain and keywords from keyword matching.
    """
    idea_lower = user_idea.lower()

    # Detect task type
    task_type = "classification"  # default
    for task, triggers in _TASK_RULES:
        if any(t in idea_lower for t in triggers):
            task_type = task
            break

    # Detect domain
    domain = "general"
    for dom, triggers in _DOMAIN_RULES.items():
        if any(t in idea_lower for t in triggers):
            domain = dom
            break

    # Extract keywords: nouns/meaningful words (simple heuristic)
    stop_words = {
        "a", "an", "the", "and", "or", "of", "to", "in", "for",
        "with", "using", "on", "by", "i", "want", "build", "create",
        "make", "develop", "model", "predict", "that", "my", "me",
        "can", "will", "is", "are", "this", "based",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", user_idea)
    keywords = list(dict.fromkeys(
        w.lower() for w in words if w.lower() not in stop_words
    ))[:5]

    logger.info(
        f"[IdeaParser] Rule-based parse -> task={task_type}  "
        f"domain={domain}  keywords={keywords}"
    )
    return ParsedIdea(
        raw_idea=user_idea,
        task_type=task_type,
        domain=domain,
        keywords=keywords,
        description=f"Rule-based parse: {task_type} task in {domain} domain.",
        parsed_by="rules",
    )


# -- LLM parser ----------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert ML project analyst.\n"
    "Given a user's project idea, extract exactly these fields and return "
    "ONLY valid JSON — no markdown, no explanation, pure JSON:\n"
    "{\n"
    f'  "task_type": "<one of: {", ".join(SUPPORTED_TASKS)}>",\n'
    '  "domain": "<short domain name, e.g. healthcare>",\n'
    '  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],\n'
    '  "description": "<one sentence describing the ML task>"\n'
    "}"
)


class IdeaParser:
    """
    Parses a free-text ML idea.

    Strategy (Issue #1 + #9):
      1. If USE_LLM=true and OPENROUTER_API_KEY is set -> try OpenRouter LLM
      2. If LLM fails or is disabled -> run rule-based parser as fallback
    """

    def parse(self, user_idea: str) -> ParsedIdea:
        # Issue #9: If LLM disabled, go straight to rules
        if not USE_LLM or not OPENROUTER_API_KEY:
            logger.info("[IdeaParser] LLM disabled or no API key — using rule-based parser.")
            return _rule_based_parse(user_idea)

        # Try LLM with fallback
        try:
            return self._llm_parse(user_idea)
        except Exception as exc:
            # Issue #1: Rule-based fallback if API fails
            logger.warning(
                f"[IdeaParser] OpenRouter API failed ({exc}). "
                "Falling back to rule-based parser."
            )
            return _rule_based_parse(user_idea)

    def _llm_parse(self, user_idea: str) -> ParsedIdea:
        from llm_agent.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
        logger.debug(f"[IdeaParser] Calling OpenRouter ({LLM_MODEL}) …")

        data = client.chat_json(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_idea}],
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.1,
        )

        task_type = data.get("task_type", "classification").lower().strip()
        if task_type not in SUPPORTED_TASKS:
            logger.warning(
                f"[IdeaParser] LLM returned unknown task_type '{task_type}' "
                "— using rule-based override."
            )
            return _rule_based_parse(user_idea)

        parsed = ParsedIdea(
            raw_idea=user_idea,
            task_type=task_type,
            domain=data.get("domain", "general"),
            keywords=data.get("keywords", []),
            description=data.get("description", ""),
            parsed_by="llm",
        )
        logger.info(
            f"[IdeaParser] LLM parse -> task={parsed.task_type}  "
            f"domain={parsed.domain}  keywords={parsed.keywords}"
        )
        return parsed
