"""
llm_agent/openrouter_client.py
--------------------------------
Thin wrapper around the OpenRouter REST API.

OpenRouter is an OpenAI-compatible gateway that routes requests to 200+
LLM models (GPT-4o, Gemini, Claude, Llama, Mistral, Deepseek, etc.)
via a single API key and a single endpoint.

Docs : https://openrouter.ai/docs
Models: https://openrouter.ai/models

Usage
-----
    from llm_agent.openrouter_client import OpenRouterClient

    client = OpenRouterClient()

    # Simple text completion
    text = client.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # With a system prompt
    text = client.chat(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "What is XGBoost?"}],
        max_tokens=256,
    )
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_SITE_NAME,
    LLM_MODEL,
    LLM_MAX_TOKENS,
)


class OpenRouterError(Exception):
    """Raised on non-2xx HTTP responses from OpenRouter."""


class OpenRouterClient:
    """
    OpenAI-compatible client for OpenRouter.

    Parameters
    ----------
    api_key   : OpenRouter API key (defaults to OPENROUTER_API_KEY from settings).
    model     : Model slug (defaults to LLM_MODEL from settings).
    max_tokens: Default max output tokens.
    timeout   : HTTP request timeout in seconds.
    """

    CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"

    def __init__(
        self,
        api_key: Optional[str]  = None,
        model: Optional[str]    = None,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
    ):
        self.api_key    = api_key    or OPENROUTER_API_KEY
        self.model      = model      or LLM_MODEL
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.timeout    = timeout

        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable. "
                "Get a key at https://openrouter.ai/keys"
            )

    # -- Public API ------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply text.

        Parameters
        ----------
        messages    : List of {"role": "user"|"assistant", "content": "..."}.
        system      : Optional system prompt prepended to the messages list.
        max_tokens  : Override default max output tokens.
        temperature : Sampling temperature (lower = more deterministic).

        Returns
        -------
        str — the assistant's reply text.

        Raises
        ------
        OpenRouterError on API errors.
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model":       self.model,
            "messages":    full_messages,
            "max_tokens":  max_tokens or self.max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        headers = {
            "Authorization":  f"Bearer {self.api_key}",
            "Content-Type":   "application/json",
            # OpenRouter recommended headers for dashboard attribution
            "HTTP-Referer":   OPENROUTER_SITE_URL,
            "X-Title":        OPENROUTER_SITE_NAME,
        }

        logger.debug(
            f"[OpenRouter] POST {self.CHAT_ENDPOINT}  "
            f"model={self.model}  messages={len(full_messages)}"
        )

        try:
            response = requests.post(
                self.CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = ""
            try:
                body = exc.response.json()
            except Exception:
                body = exc.response.text
            raise OpenRouterError(
                f"OpenRouter HTTP {exc.response.status_code}: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        data = response.json()

        # Extract reply text
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise OpenRouterError(
                f"Unexpected OpenRouter response format: {data}"
            ) from exc

        logger.debug(
            f"[OpenRouter] Response received  "
            f"tokens_used={data.get('usage', {}).get('total_tokens', '?')}"
        )
        return text.strip()

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
    ) -> dict:
        """
        Like chat(), but parses and returns the response as a JSON dict.
        Strips markdown code fences before parsing.

        Raises
        ------
        ValueError if the response is not valid JSON.
        OpenRouterError on API errors.
        """
        raw = self.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Strip optional ```json ... ``` fences
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenRouter response was not valid JSON:\n{raw}"
            ) from exc
