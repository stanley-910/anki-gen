"""Tests for context-budget estimation helpers in anki_gen.generator."""

import sys
from unittest.mock import MagicMock

import pytest

from anki_gen.generator import (
    _CONTEXT_LIMITS,
    _CONTEXT_WARN_THRESHOLD,
    _check_context_budget,
    _context_limit_for,
    _estimate_tokens,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # min guard

    def test_four_chars_is_one_token(self):
        assert _estimate_tokens("abcd") == 1

    def test_proportional(self):
        assert _estimate_tokens("a" * 400) == 100

    def test_large_text(self):
        result = _estimate_tokens("word " * 10_000)  # 50_000 chars
        assert result == 12_500


class TestContextLimitFor:
    def test_gpt4o(self):
        assert _context_limit_for("openai/gpt-4o") == 128_000

    def test_gpt4o_mini(self):
        assert _context_limit_for("openai/gpt-4o-mini") == 128_000

    def test_gpt35(self):
        assert _context_limit_for("openai/gpt-3.5-turbo") == 16_385

    def test_claude(self):
        assert _context_limit_for("anthropic/claude-3-5-sonnet-latest") == 200_000

    def test_unknown_falls_back_to_default(self):
        assert (
            _context_limit_for("some/unknown-model-xyz") == _CONTEXT_LIMITS["_default"]
        )

    def test_case_insensitive(self):
        assert _context_limit_for("OpenAI/GPT-4O") == 128_000


class TestCheckContextBudget:
    def _make_provider(self, name: str) -> MagicMock:
        p = MagicMock()
        p.name = name
        return p

    def test_no_warning_below_threshold(self, capsys):
        provider = self._make_provider("openai/gpt-4o")
        # 128_000 limit * 0.75 threshold = 96_000 tokens → 384_000 chars
        # Use a prompt just under: 383_999 chars → 95_999 tokens
        prompt = "x" * (383_999)
        _check_context_budget(prompt, provider)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warning_at_threshold(self, capsys):
        provider = self._make_provider("openai/gpt-4o")
        # 96_000 tokens exactly → 384_000 chars (= 75% of 128k)
        prompt = "x" * 384_000
        _check_context_budget(prompt, provider)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "gpt-4o" in captured.err
        assert "128,000" in captured.err

    def test_warning_contains_percent(self, capsys):
        provider = self._make_provider("anthropic/claude-3-5-sonnet-latest")
        # 200_000 * 0.75 = 150_000 tokens → 600_000 chars (exactly at threshold)
        prompt = "x" * 600_000
        _check_context_budget(prompt, provider)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "%" in captured.err

    def test_warning_goes_to_stderr_not_stdout(self, capsys):
        provider = self._make_provider("openai/gpt-4o")
        prompt = "x" * 500_000  # well over threshold
        _check_context_budget(prompt, provider)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "WARNING" in captured.err
