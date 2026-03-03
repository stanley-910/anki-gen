from __future__ import annotations

import os

from anki_gen.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by the Anthropic Messages API.

    Reads ANTHROPIC_API_KEY from the environment. The model is
    configurable at construction time; defaults to claude-3-5-sonnet-latest.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc

        self._model = model
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def complete(self, prompt: str) -> str:
        import anthropic as _anthropic

        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        if isinstance(block, _anthropic.types.TextBlock):
            return block.text
        return ""
