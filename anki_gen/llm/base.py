from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract interface for an LLM backend.

    Concrete implementations must produce a UTF-8 string response
    given a plain-text prompt. All provider-specific concerns
    (auth, model selection, retry logic) are encapsulated here.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* to the model and return the raw text response."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier shown in CLI output."""
        ...
