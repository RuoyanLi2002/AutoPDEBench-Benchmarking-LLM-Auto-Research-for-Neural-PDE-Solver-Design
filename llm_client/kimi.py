"""
llm_client.py — Thin wrapper around the Moonshot AI (Kimi) API for
Kimi K2.7 Code.

Moonshot exposes an OpenAI-compatible endpoint, so this still uses the
`openai` Python package (>=1.0), just pointed at the Moonshot base URL.
Requires the MOONSHOT_API_KEY environment variable. Swap the underlying
client here if you want to use a different provider; the rest of the agent
only depends on `LLMClient.chat(...)`.

Notes specific to Kimi K2.7 Code:
- Model IDs: "kimi-k2.7-code" (default) or "kimi-k2.7-code-highspeed"
  (faster serving tier, recommended for agent loops).
- The model always runs in thinking mode; reasoning tokens are billed as
  output tokens, so budget max_tokens generously.
- Sampling overrides (temperature, top_p) are ignored by the service,
  which uses fixed sampling settings. The parameter is kept here only for
  interface compatibility with the DashScope client.
"""

import os
from typing import Optional

# International endpoint. If your Moonshot account is on the China platform,
# use "https://api.moonshot.cn/v1" instead.
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"


class LLMClient:
    def __init__(self, model_name: str = "kimi-k2.7-code",
                 temperature: float = 1):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required. Install with `pip install openai`."
            ) from e

        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MOONSHOT_API_KEY environment variable is not set. "
                "Set it before running the agent."
            )

        self._client = OpenAI(api_key=api_key, base_url=MOONSHOT_BASE_URL)
        self.model_name = model_name
        # Kept for interface compatibility; the K2.7 Code service ignores
        # sampling overrides and uses fixed settings.
        self.temperature = 1

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """Send a single user turn to the model and return the text reply.

        `max_tokens` is only included in the request when explicitly set, so we
        never serialize a `null` value (which the API rejects).

        The returned text is the final answer (`message.content`); the model's
        chain-of-thought, if exposed, arrives separately in
        `message.reasoning_content` and is intentionally discarded here.
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