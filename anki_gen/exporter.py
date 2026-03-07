from __future__ import annotations

from pathlib import Path

from anki_gen.models import BasicCard, Card, DefinitionCard

# ---------------------------------------------------------------------------
# Stable model IDs — generated once, hardcoded per Anki convention.
# These must be consistent across runs so that re-imports update rather
# than duplicate existing cards.
# ---------------------------------------------------------------------------
_BASIC_MODEL_ID = 1_607_392_319
_DEFINITION_MODEL_ID = 1_607_392_320
_BASIC_REVERSED_MODEL_ID = 1_607_392_321

# Structural CSS for code blocks — monospace font, scroll on overflow.
_CODE_CSS = """\
pre {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 0.875em;
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 0.8em 1em;
  border-radius: 6px;
  overflow-x: auto;
  white-space: pre;
  margin: 0.5em 0;
}
code {
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 0.875em;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.5em auto;
}
"""


def _build_css() -> str:
    """Combine base card CSS with code block rules."""
    base = (
        "body { font-family: Arial, sans-serif; font-size: 16px; }\n"
        ".card { padding: 1em; }\n"
        "hr#answer { margin: 1em 0; }\n"
    )
    return base + _CODE_CSS


def _make_basic_model():  # type: ignore[return]
    import genanki

    return genanki.Model(
        _BASIC_MODEL_ID,
        "anki-gen Basic",
        fields=[
            {"name": "Question"},
            {"name": "Answer"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Answer}}",
            }
        ],
        css=_build_css(),
    )


def _make_definition_model():  # type: ignore[return]
    import genanki

    return genanki.Model(
        _DEFINITION_MODEL_ID,
        "anki-gen Definition",
        fields=[
            {"name": "Term"},
            {"name": "Definition"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Term}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Definition}}",
            }
        ],
        css=_build_css(),
    )


def _make_basic_reversed_model():  # type: ignore[return]
    import genanki

    return genanki.Model(
        _BASIC_REVERSED_MODEL_ID,
        "anki-gen Basic (reversed)",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Back}}",
            },
            {
                "name": "Card 2",
                "qfmt": "{{Back}}",
                "afmt": "{{FrontSide}}<hr id='answer'>{{Front}}",
            },
        ],
        css=_build_css(),
    )


def _stable_deck_id(deck_name: str) -> int:
    """
    Derive a deterministic deck ID from the name so that re-running
    the tool with the same deck name always maps to the same Anki deck
    rather than creating a duplicate.
    """
    import hashlib

    digest = hashlib.sha256(deck_name.encode()).hexdigest()
    # Anki deck IDs are 64-bit integers; keep it in a safe positive range.
    return int(digest[:15], 16) % (2**31 - 1) + (1 << 30)


def _note_for_card(card: Card, basic_model, definition_model, basic_reversed_model):  # type: ignore[return]
    """
    Build a genanki.Note from a Card.
    Fields are already HTML (produced by _render_code in generator.py) —
    no further escaping is applied.
    """
    import genanki

    if isinstance(card, BasicCard):
        tags = ["anki-gen"] + card.tags
        if card.reversed:
            return genanki.Note(
                model=basic_reversed_model,
                fields=[card.front, card.back],
                tags=tags,
            )
        return genanki.Note(
            model=basic_model,
            fields=[card.front, card.back],
            tags=tags,
        )
    elif isinstance(card, DefinitionCard):
        tags = ["anki-gen", "definition"] + card.tags
        return genanki.Note(
            model=definition_model,
            fields=[card.term, card.definition],
            tags=tags,
        )
    else:
        raise TypeError(f"Unknown card type: {type(card)}")


def export_apkg(
    cards_by_deck: dict[str, list[Card]],
    output_path: Path,
    media_files: list[Path] | None = None,
) -> None:
    """
    Write an .apkg file containing one deck per entry in *cards_by_deck*.

    Args:
        cards_by_deck: Mapping of deck name → list of Card objects.
        output_path:   Destination .apkg file path.
        media_files:   Optional list of image/media file paths to bundle.
    """
    try:
        import genanki
    except ImportError as exc:
        raise ImportError(
            "genanki is required for .apkg export. Install it with: pip install genanki"
        ) from exc

    basic_model = _make_basic_model()
    definition_model = _make_definition_model()
    basic_reversed_model = _make_basic_reversed_model()

    decks: list = []
    for deck_name, cards in cards_by_deck.items():
        deck = genanki.Deck(_stable_deck_id(deck_name), deck_name)
        for card in cards:
            deck.add_note(
                _note_for_card(
                    card, basic_model, definition_model, basic_reversed_model
                )
            )
        decks.append(deck)

    package = genanki.Package(decks)
    if media_files:
        package.media_files = [str(p) for p in media_files]
    package.write_to_file(str(output_path))
