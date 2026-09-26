import os
from typing import Optional




def _uses_completion_tokens(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


class LLMClient:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The `openai` package is required. Install with `pip install openai`."
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before running the agent."
            )

        self._client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self._new_token_param = _uses_completion_tokens(model_name)

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """Send a single user turn to the model and return the text reply.

        `max_tokens` is only included in the request when explicitly set, so we
        never serialize a `null` value (which the API rejects). For newer
        models the parameter is sent as `max_completion_tokens`.
        """
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Newer reasoning models only support the default temperature; omit it.
        if not self._new_token_param:
            kwargs["temperature"] = self.temperature

        if max_tokens is not None:
            if self._new_token_param:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content