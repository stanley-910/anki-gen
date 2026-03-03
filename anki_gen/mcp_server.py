from __future__ import annotations

"""
anki-gen MCP server
===================
Exposes anki-gen's card-generation pipeline as MCP tools so that AI agents
running in Cursor, Claude Desktop, or any other MCP-compatible host can drive
the tool programmatically.

Tools
-----
- generate_cards  : run the LLM pipeline on file paths or raw Markdown content
- push_cards      : push a previously-generated card set to Anki via AnkiConnect
- export_cards    : write a previously-generated card set to a .apkg file
- list_decks      : list all decks currently known to the running Anki instance

Usage (stdio transport, compatible with Cursor / Claude Desktop)::

    anki-gen-mcp

The server reads OPENAI_API_KEY / ANTHROPIC_API_KEY from environment / .env.
The LLM provider is selected via the ANKI_GEN_PROVIDER environment variable
(defaults to "openai"). The model can be overridden via ANKI_GEN_MODEL.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="anki-gen",
    instructions=(
        "Generate Anki flashcards from Markdown notes using an LLM. "
        "Use generate_cards to create cards, then either push_cards to send "
        "them to a running Anki instance or export_cards to save an .apkg file."
    ),
)

# ---------------------------------------------------------------------------
# In-memory card store — keyed by a session handle returned by generate_cards
# ---------------------------------------------------------------------------
# Maps handle -> {"cards": list[Card], "deck_name": str}
_card_store: dict[str, dict[str, Any]] = {}
_handle_counter = 0


def _next_handle() -> str:
    global _handle_counter
    _handle_counter += 1
    return f"cards_{_handle_counter}"


# ---------------------------------------------------------------------------
# Provider factory (mirrors cli.py but reads from env)
# ---------------------------------------------------------------------------


def _build_provider():
    from anki_gen.llm.base import LLMProvider  # noqa: F401 (type hint only)

    provider_name = os.environ.get("ANKI_GEN_PROVIDER", "openai").lower()
    model = os.environ.get("ANKI_GEN_MODEL") or None

    if provider_name == "openai":
        from anki_gen.llm.openai import OpenAIProvider

        return OpenAIProvider(**({"model": model} if model else {}))
    elif provider_name == "anthropic":
        from anki_gen.llm.anthropic import AnthropicProvider

        return AnthropicProvider(**({"model": model} if model else {}))
    else:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            "Set ANKI_GEN_PROVIDER to 'openai' or 'anthropic'."
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Generate Anki flashcards from one or more Markdown files or from raw "
        "Markdown content. Returns a handle that can be passed to push_cards or "
        "export_cards.\n\n"
        "Provide either `file_paths` (list of absolute paths to .md files or "
        "directories) OR `content` (raw Markdown string) — not both.\n\n"
        "Optional `key_concepts` constrains generation to those specific concepts "
        "(two-phase flow). Optional `deck_name` sets the target deck (defaults to "
        "the document's H1 title). Optional `max_cards` caps the number of cards."
    )
)
def generate_cards(
    file_paths: list[str] | None = None,
    content: str | None = None,
    key_concepts: list[str] | None = None,
    deck_name: str | None = None,
    max_cards: int | None = None,
) -> dict[str, Any]:
    """Generate flashcards and return a session handle plus a summary."""
    from anki_gen.generator import generate_cards as gen_cards
    from anki_gen.generator import generate_cards_from_concepts
    from anki_gen.parser import ParsedDocument, collect_markdown_files, parse_file

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if file_paths and content:
        return {"error": "Provide either file_paths or content, not both."}
    if not file_paths and not content:
        return {"error": "Provide at least one of file_paths or content."}

    provider = _build_provider()

    all_cards = []
    resolved_deck = deck_name

    # ------------------------------------------------------------------
    # Source: raw Markdown content
    # ------------------------------------------------------------------
    if content:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            doc = parse_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not resolved_deck:
            resolved_deck = doc.title

        if key_concepts:
            cards = generate_cards_from_concepts(doc, key_concepts, provider)
        else:
            cards = gen_cards(doc, provider, max_cards=max_cards)

        all_cards.extend(cards)

    # ------------------------------------------------------------------
    # Source: file paths
    # ------------------------------------------------------------------
    else:
        raw_paths = [Path(p) for p in (file_paths or [])]
        try:
            md_files = collect_markdown_files(raw_paths)
        except ValueError as exc:
            return {"error": str(exc)}

        if not md_files:
            return {"error": "No .md files found at the provided paths."}

        for md_path in md_files:
            try:
                doc = parse_file(md_path)
            except Exception as exc:
                return {"error": f"Failed to parse {md_path}: {exc}"}

            if not resolved_deck:
                resolved_deck = deck_name or doc.title

            if key_concepts:
                cards = generate_cards_from_concepts(doc, key_concepts, provider)
            else:
                cards = gen_cards(doc, provider, max_cards=max_cards)

            all_cards.extend(cards)

    if not all_cards:
        return {"error": "No cards were generated."}

    # ------------------------------------------------------------------
    # Store and return handle
    # ------------------------------------------------------------------
    handle = _next_handle()
    _card_store[handle] = {
        "cards": all_cards,
        "deck_name": resolved_deck or "anki-gen",
    }

    # Build a compact summary
    type_counts: dict[str, int] = {}
    for card in all_cards:
        type_counts[card.type] = type_counts.get(card.type, 0) + 1

    return {
        "handle": handle,
        "deck_name": resolved_deck,
        "total_cards": len(all_cards),
        "by_type": type_counts,
    }


@mcp.tool(
    description=(
        "Push cards referenced by `handle` into a running Anki instance via "
        "AnkiConnect. Anki must be open with the AnkiConnect add-on installed.\n\n"
        "Optional `deck_name` overrides the deck name stored in the handle."
    )
)
def push_cards(
    handle: str,
    deck_name: str | None = None,
) -> dict[str, Any]:
    """Push generated cards to Anki via AnkiConnect."""
    from anki_gen import ankiconnect

    if handle not in _card_store:
        return {"error": f"Unknown handle '{handle}'. Run generate_cards first."}

    entry = _card_store[handle]
    target_deck = deck_name or entry["deck_name"]
    cards = entry["cards"]

    try:
        ankiconnect.check_connection()
    except ankiconnect.AnkiConnectError as exc:
        return {"error": str(exc)}

    try:
        added, skipped = ankiconnect.push_cards(target_deck, cards)
    except ankiconnect.AnkiConnectError as exc:
        return {"error": str(exc)}

    return {
        "deck_name": target_deck,
        "added": added,
        "skipped": skipped,
    }


@mcp.tool(
    description=(
        "Export cards referenced by `handle` to an Anki .apkg file.\n\n"
        "`output_path` must be an absolute path ending in .apkg. "
        "Optional `deck_name` overrides the deck name stored in the handle."
    )
)
def export_cards(
    handle: str,
    output_path: str,
    deck_name: str | None = None,
) -> dict[str, Any]:
    """Write generated cards to a .apkg file."""
    from anki_gen.exporter import export_apkg

    if handle not in _card_store:
        return {"error": f"Unknown handle '{handle}'. Run generate_cards first."}

    entry = _card_store[handle]
    target_deck = deck_name or entry["deck_name"]
    cards = entry["cards"]

    out = Path(output_path)
    if out.suffix.lower() != ".apkg":
        return {"error": f"output_path must end in .apkg, got: {output_path}"}

    try:
        export_apkg({target_deck: cards}, out)
    except Exception as exc:
        return {"error": f"Failed to write .apkg: {exc}"}

    return {
        "output_path": str(out.resolve()),
        "deck_name": target_deck,
        "total_cards": len(cards),
    }


@mcp.tool(
    description=(
        "List all deck names currently known to the running Anki instance. "
        "Requires Anki to be open with the AnkiConnect add-on installed."
    )
)
def list_decks() -> dict[str, Any]:
    """Return all deck names from the running Anki instance."""
    from anki_gen import ankiconnect

    try:
        ankiconnect.check_connection()
    except ankiconnect.AnkiConnectError as exc:
        return {"error": str(exc)}

    try:
        decks: list[str] = ankiconnect._invoke("deckNames")
    except ankiconnect.AnkiConnectError as exc:
        return {"error": str(exc)}

    return {"decks": sorted(decks)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
