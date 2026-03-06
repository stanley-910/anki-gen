"""Tests for deterministic concept chunking in anki_gen.generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from anki_gen.generator import (
    CHUNK_SIZE,
    chunk_concepts,
    generate_cards_for_chunk,
    generate_cards_from_concepts,
)
from anki_gen.models import BasicCard
from anki_gen.parser import ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(text: str = "source content") -> ParsedDocument:
    return ParsedDocument(
        source_path=Path("test.md"),
        title="Test",
        plain_text=text,
        concept_count=10,
    )


def _concepts(n: int) -> list[str]:
    return [f"concept_{i}" for i in range(n)]


def _provider_returning(cards_per_call: list[list[str]]) -> MagicMock:
    """Mock provider whose successive complete() calls return JSON arrays.

    Each element of *cards_per_call* is a list of concept labels for that
    call; the mock returns one basic card per label (front=label, back="def").
    """
    provider = MagicMock()
    provider.name = "openai/gpt-4o"
    responses = []
    for labels in cards_per_call:
        arr = [{"type": "basic", "front": lbl, "back": "definition"} for lbl in labels]
        responses.append(json.dumps(arr))
    provider.complete.side_effect = responses
    return provider


# ---------------------------------------------------------------------------
# chunk_concepts unit tests
# ---------------------------------------------------------------------------


class TestChunkConcepts:
    def test_empty_returns_empty(self):
        assert chunk_concepts([], None, 20) == []

    def test_below_chunk_size_single_chunk(self):
        concepts = _concepts(5)
        result = chunk_concepts(concepts, None, 20)
        assert len(result) == 1
        assert result[0][0] == concepts

    def test_exact_chunk_size_single_chunk(self):
        concepts = _concepts(20)
        result = chunk_concepts(concepts, None, 20)
        assert len(result) == 1
        assert result[0][0] == concepts

    def test_even_split(self):
        concepts = _concepts(40)
        result = chunk_concepts(concepts, None, 20)
        assert len(result) == 2
        assert result[0][0] == concepts[:20]
        assert result[1][0] == concepts[20:]

    def test_remainder_chunk(self):
        concepts = _concepts(45)
        result = chunk_concepts(concepts, None, 20)
        assert len(result) == 3
        assert len(result[0][0]) == 20
        assert len(result[1][0]) == 20
        assert len(result[2][0]) == 5

    def test_order_preserved(self):
        concepts = _concepts(25)
        result = chunk_concepts(concepts, None, 20)
        flat = [c for chunk, _ in result for c in chunk]
        assert flat == concepts

    def test_no_reversed_concepts(self):
        concepts = _concepts(25)
        result = chunk_concepts(concepts, None, 20)
        assert all(rev is None for _, rev in result)

    def test_reversed_concepts_scoped_to_chunk(self):
        concepts = _concepts(40)
        # Mark one concept from each chunk as reversed
        rev = {concepts[5], concepts[25]}
        result = chunk_concepts(concepts, rev, 20)
        chunk0_reversed = result[0][1]
        chunk1_reversed = result[1][1]
        assert chunk0_reversed == {concepts[5]}
        assert chunk1_reversed == {concepts[25]}

    def test_reversed_concepts_all_in_one_chunk(self):
        concepts = _concepts(40)
        rev = {concepts[3], concepts[7]}  # both in first chunk
        result = chunk_concepts(concepts, rev, 20)
        assert result[0][1] == rev
        assert result[1][1] is None  # second chunk has none → None

    def test_reversed_none_when_chunk_has_no_matches(self):
        concepts = _concepts(40)
        rev = {concepts[0]}  # only in first chunk
        result = chunk_concepts(concepts, rev, 20)
        assert result[1][1] is None

    def test_chunk_size_one(self):
        concepts = _concepts(3)
        result = chunk_concepts(concepts, None, 1)
        assert len(result) == 3
        assert [c for c, _ in result] == [[c] for c in concepts]


# ---------------------------------------------------------------------------
# generate_cards_from_concepts integration tests (mock provider)
# ---------------------------------------------------------------------------


class TestGenerateCardsFromConcepts:
    def test_single_chunk_calls_provider_once(self):
        concepts = _concepts(5)
        provider = _provider_returning([concepts])
        cards = generate_cards_from_concepts(_doc(), concepts, provider, chunk_size=20)
        assert provider.complete.call_count == 1
        assert len(cards) == 5

    def test_two_chunks_calls_provider_twice(self):
        concepts = _concepts(25)
        provider = _provider_returning([concepts[:20], concepts[20:]])
        cards = generate_cards_from_concepts(_doc(), concepts, provider, chunk_size=20)
        assert provider.complete.call_count == 2
        assert len(cards) == 25

    def test_three_chunks_remainder(self):
        concepts = _concepts(45)
        provider = _provider_returning(
            [
                concepts[:20],
                concepts[20:40],
                concepts[40:],
            ]
        )
        cards = generate_cards_from_concepts(_doc(), concepts, provider, chunk_size=20)
        assert provider.complete.call_count == 3
        assert len(cards) == 45

    def test_output_order_matches_input_order(self):
        concepts = _concepts(25)
        provider = _provider_returning([concepts[:20], concepts[20:]])
        cards = generate_cards_from_concepts(_doc(), concepts, provider, chunk_size=20)
        fronts = [c.front for c in cards]
        assert fronts == concepts

    def test_reversed_concepts_scoped_per_chunk(self):
        concepts = _concepts(25)
        rev = {concepts[3], concepts[22]}  # one per chunk
        provider = _provider_returning([concepts[:20], concepts[20:]])
        generate_cards_from_concepts(
            _doc(), concepts, provider, reversed_concepts=rev, chunk_size=20
        )
        call_args = [c.args[0] for c in provider.complete.call_args_list]
        # First chunk prompt must mention concept_3 as reversed
        assert "[REVERSED]" in call_args[0]
        assert concepts[3] in call_args[0]
        # Second chunk prompt must mention concept_22 as reversed
        assert "[REVERSED]" in call_args[1]
        assert concepts[22] in call_args[1]
        # Cross-contamination: reversed marker for concept_3 must NOT appear in chunk 2
        assert concepts[3] not in call_args[1]

    def test_notes_forwarded_to_every_chunk(self):
        concepts = _concepts(25)
        notes = ["focus on equations", "bidirectional where possible"]
        provider = _provider_returning([concepts[:20], concepts[20:]])
        generate_cards_from_concepts(
            _doc(), concepts, provider, notes=notes, chunk_size=20
        )
        for c in provider.complete.call_args_list:
            prompt = c.args[0]
            assert "focus on equations" in prompt
            assert "bidirectional where possible" in prompt

    def test_without_notes_caps_each_chunk(self):
        # Provider returns more cards than concepts in chunk — cap must apply.
        concepts = _concepts(3)
        extra = [
            {"type": "basic", "front": f"concept_{i}", "back": "def"} for i in range(5)
        ]
        provider = MagicMock()
        provider.name = "openai/gpt-4o"
        provider.complete.return_value = json.dumps(extra)
        cards = generate_cards_from_concepts(_doc(), concepts, provider, chunk_size=20)
        assert len(cards) == 3  # capped at num_concepts

    def test_with_notes_lifts_cap(self):
        # Provider returns more cards than concepts — with notes the cap is lifted.
        concepts = _concepts(3)
        extra = [
            {"type": "basic", "front": f"extra_{i}", "back": "def"} for i in range(5)
        ]
        provider = MagicMock()
        provider.name = "openai/gpt-4o"
        provider.complete.return_value = json.dumps(extra)
        cards = generate_cards_from_concepts(
            _doc(), concepts, provider, notes=["add more"], chunk_size=20
        )
        assert len(cards) == 5  # cap lifted

    def test_empty_concepts_returns_empty(self):
        provider = MagicMock()
        provider.name = "openai/gpt-4o"
        cards = generate_cards_from_concepts(_doc(), [], provider, chunk_size=20)
        assert cards == []
        provider.complete.assert_not_called()

    def test_single_chunk_identical_to_chunk_size_eq_total(self):
        """Regression: chunk_size >= len(concepts) must behave like old single call."""
        concepts = _concepts(10)
        provider = _provider_returning([concepts])
        cards_chunked = generate_cards_from_concepts(
            _doc(), concepts, provider, chunk_size=10
        )
        provider2 = _provider_returning([concepts])
        cards_large = generate_cards_from_concepts(
            _doc(), concepts, provider2, chunk_size=100
        )
        assert len(cards_chunked) == len(cards_large) == 10

    def test_default_chunk_size_is_module_constant(self):
        """chunk_size default must equal CHUNK_SIZE so callers get the same value."""
        import inspect

        sig = inspect.signature(generate_cards_from_concepts)
        assert sig.parameters["chunk_size"].default == CHUNK_SIZE
