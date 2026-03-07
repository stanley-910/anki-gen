from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from anki_gen.models import BasicCard, Card, DefinitionCard

_ANKICONNECT_URL = "http://127.0.0.1:8765"
_ANKICONNECT_VERSION = 6


class AnkiConnectError(RuntimeError):
    """Raised when AnkiConnect returns an error payload or is unreachable."""


def _invoke(action: str, **params: Any) -> Any:
    """
    Send a single request to the AnkiConnect HTTP API and return the result.

    Raises AnkiConnectError on any failure — network, API-level, or
    structural (malformed response envelope).
    """
    payload = json.dumps(
        {"action": action, "version": _ANKICONNECT_VERSION, "params": params}
    ).encode("utf-8")

    request = urllib.request.Request(
        _ANKICONNECT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            response = json.load(resp)
    except urllib.error.URLError as exc:
        raise AnkiConnectError(
            f"Cannot reach AnkiConnect at {_ANKICONNECT_URL}. "
            "Is Anki running with the AnkiConnect add-on installed?"
        ) from exc

    if (
        not isinstance(response, dict)
        or "result" not in response
        or "error" not in response
    ):
        raise AnkiConnectError(f"Unexpected AnkiConnect response: {response!r}")

    if response["error"] is not None:
        raise AnkiConnectError(f"AnkiConnect error: {response['error']}")

    return response["result"]


def _card_to_note_payload(card: Card, deck_name: str) -> dict[str, Any]:
    """Convert a Card object into the note dict expected by AnkiConnect's addNote."""
    if isinstance(card, BasicCard):
        model = "Basic (and reversed card)" if card.reversed else "Basic"
        tags = ["anki-gen"] + card.tags
        return {
            "deckName": deck_name,
            "modelName": model,
            "fields": {"Front": card.front, "Back": card.back},
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
            "tags": tags,
        }
    elif isinstance(card, DefinitionCard):
        tags = ["anki-gen", "definition"] + card.tags
        return {
            "deckName": deck_name,
            "modelName": "Basic",
            "fields": {"Front": card.term, "Back": card.definition},
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
            "tags": tags,
        }
    else:
        raise TypeError(f"Unknown card type: {type(card)}")


def check_connection() -> None:
    """Verify that AnkiConnect is reachable. Raises AnkiConnectError if not."""
    _invoke("version")


def ensure_deck(deck_name: str) -> None:
    """Create *deck_name* if it does not already exist (idempotent)."""
    _invoke("createDeck", deck=deck_name)


def deck_exists(deck_name: str) -> bool:
    """Return True if *deck_name* already exists in Anki."""
    names: list[str] = _invoke("deckNames")
    return deck_name in names


def get_deck_card_count(deck_name: str) -> int:
    """Return the number of cards currently in *deck_name*."""
    card_ids: list[int] = _invoke("findCards", query=f'deck:"{deck_name}"')
    return len(card_ids)


def push_cards(deck_name: str, cards: list[Card]) -> tuple[int, list[str]]:
    """
    Push *cards* into *deck_name* via AnkiConnect.

    Uses a canAddNotes preflight so duplicate detection is handled cleanly
    without relying on addNotes error behaviour (which varies by AnkiConnect
    version).  Duplicate scope is restricted to the target deck, so the same
    card can exist in multiple decks without being flagged.

    Returns (added, skipped_fronts) where skipped_fronts is the list of
    "Front" field values for every card that was skipped as a duplicate.
    """
    ensure_deck(deck_name)

    notes = [_card_to_note_payload(card, deck_name) for card in cards]

    # Pre-flight: which notes are not already in this deck?
    can_add: list[bool] = _invoke("canAddNotes", notes=notes)

    addable = [note for note, ok in zip(notes, can_add) if ok]
    skipped_fronts = [
        note["fields"]["Front"] for note, ok in zip(notes, can_add) if not ok
    ]

    if addable:
        _invoke("addNotes", notes=addable)

    return len(addable), skipped_fronts


def store_media_files(media_files: list[Path]) -> None:
    """
    Store image/media files in Anki's media collection via AnkiConnect.

    Each file is base64-encoded and uploaded with its basename as the
    filename so that Anki can resolve ``<img src="filename">`` tags in cards.
    Existing files with the same name are silently overwritten.
    """
    for path in media_files:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        _invoke("storeMediaFile", filename=path.name, data=data)
