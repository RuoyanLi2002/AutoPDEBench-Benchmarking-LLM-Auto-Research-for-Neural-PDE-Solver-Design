"""
llm_client.py — Thin wrapper around the Google Gemini API (OpenAI-compatible).

Google exposes an OpenAI-compatible endpoint for the Gemini API, so this still
uses the `openai` Python package (>=1.0), just pointed at the Gemini base URL.
Requires the GEMINI_API_KEY environment variable. Swap the underlying client
here if you want to use a different provider; the rest of the agent only
depends on `LLMClient.chat(...)`.
"""

import os
from typing import Optional

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMClient:
    def __init__(self, model_name: str = "gemini-3.1-pro-preview",
                 temperature: float = 0.7):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required. Install with `pip install openai`."
            ) from e

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before running the agent."
            )

        self._client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        self.model_name = model_name
        self.temperature = temperature

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """Send a single user turn to the model and return the text reply.

        `max_tokens` is only included in the request when explicitly set, so we
        never serialize a `null` value (which the API rejects).
        """
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content