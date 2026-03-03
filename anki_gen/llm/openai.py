from __future__ import annotations

import os

from anki_gen.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by the OpenAI Chat Completions API.

    Reads OPENAI_API_KEY from the environment. The model is
    configurable at construction time; defaults to gpt-4o.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
