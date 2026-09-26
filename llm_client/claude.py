"""
llm_client.py — Thin wrapper around the Anthropic Messages API.

Requires the `anthropic` Python package and the ANTHROPIC_API_KEY environment
variable. Swap the underlying client here if you want to use a different
provider; the rest of the agent only depends on `LLMClient.chat(...)`.
"""

import os
from typing import Optional

# Anthropic requires max_tokens on every request (unlike OpenAI, where it is
# optional). Used whenever the caller does not pass max_tokens explicitly.
# On Fable 5, adaptive thinking is always on and thinking tokens count against
# max_tokens, so this must cover thinking + response. Only generated tokens
# are billed, so a generous limit costs nothing when unused.
DEFAULT_MAX_TOKENS = 32000


class LLMClient:
    def __init__(self, model_name: str = "claude-fable-5", temperature: float = 0.7):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "The `anthropic` package is required. Install with `pip install anthropic`."
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before running the agent."
            )

        self._client = Anthropic(api_key=api_key)
        self.model_name = model_name
        # Kept for call-site compatibility but never sent: Fable 5 and
        # Opus 4.7+ return a 400 error for non-default temperature/top_p/top_k.
        self.temperature = temperature

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """Send a single user turn to the model and return the text reply.

        The system prompt is passed via the top-level `system` parameter
        rather than as a message. `max_tokens` is mandatory in the Anthropic
        API, so DEFAULT_MAX_TOKENS is used when it is not provided.

        The request is streamed because the SDK requires streaming for
        requests that could take longer than 10 minutes (large max_tokens on
        big models). The streamed chunks are accumulated and returned as one
        string, so callers see no difference.
        """
        chunks = []
        with self._client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        return "".join(chunks)