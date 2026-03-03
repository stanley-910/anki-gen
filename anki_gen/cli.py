from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from anki_gen import ankiconnect
from anki_gen.exporter import export_apkg
from anki_gen.generator import (
    extract_concepts,
    generate_cards,
    generate_cards_from_concepts,
)
from anki_gen.models import BasicCard, Card, DefinitionCard
from anki_gen.parser import ParsedDocument, collect_markdown_files, parse_file

load_dotenv()


# ---------------------------------------------------------------------------
# Terminal color helpers
# ---------------------------------------------------------------------------


def _fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


_R = "\x1b[0m"
_BOLD = "\x1b[1m"

# Accent-derived color strings — populated by _apply_accent_colors()
_C_INDEX = ""  # dim accent   → [N/N]
_C_FILENAME = ""  # bright+bold  → filename / deck name
_C_ACCENT = ""  # full accent  → labels, dividers, summary header
_C_NUM = ""  # full+bold    → bare numbers
_C_TAG_VAL = ""  # mid-dim      → tag values


def _apply_accent_colors() -> None:
    """Recompute output color constants from the current _ANIM_R/G/B triple."""
    global _C_INDEX, _C_FILENAME, _C_ACCENT, _C_NUM, _C_TAG_VAL
    r, g, b = _ANIM_R, _ANIM_G, _ANIM_B
    _C_INDEX = _fg(r, g, b)
    _C_FILENAME = (
        _fg(
            min(255, int(r + (255 - r) * 0.4)),
            min(255, int(g + (255 - g) * 0.4)),
            min(255, int(b + (255 - b) * 0.4)),
        )
        + _BOLD
    )
    _C_ACCENT = _fg(r, g, b)
    _C_NUM = _fg(r, g, b) + _BOLD
    _C_TAG_VAL = _fg(int(r * 0.65), int(g * 0.65), int(b * 0.65))


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _build_provider(provider: str, model: str | None):
    if provider == "openai":
        from anki_gen.llm.openai import OpenAIProvider

        kwargs = {"model": model} if model else {}
        return OpenAIProvider(**kwargs)
    elif provider == "anthropic":
        from anki_gen.llm.anthropic import AnthropicProvider

        kwargs = {"model": model} if model else {}
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose 'openai' or 'anthropic'."
        )


# ---------------------------------------------------------------------------
# Verbose output helpers
# ---------------------------------------------------------------------------


def _print_thinking_concepts(concepts: list[str], provider_name: str) -> None:
    """Print the concept list the LLM identified (verbose thinking phase)."""
    print(f"  → {len(concepts)} concept(s) identified via {provider_name}:")
    for i, c in enumerate(concepts, 1):
        print(f"     {i:>2}. {c}")


def _print_summary(deck_name: str, cards: list[Card], verbose: bool) -> None:
    """Print a per-deck card-type breakdown after generation."""
    counts: Counter = Counter()
    for card in cards:
        if isinstance(card, BasicCard):
            counts["basic (reversed)" if card.reversed else "basic"] += 1
        elif isinstance(card, DefinitionCard):
            counts["definition"] += 1
        else:
            counts["other"] += 1

    if verbose:
        print(f"\n  {_C_ACCENT}Summary{_R}")
        print("  " + "─" * 30)
        for label, n in sorted(counts.items()):
            print(f"  {_C_ACCENT}  {label:<18}{_R}  {_C_NUM}{n:>4}{_R} card(s)")
        print("  " + "─" * 30)
        print(f"  {_C_ACCENT}  {'total':<18}{_R}  {_C_NUM}{len(cards):>4}{_R} card(s)")

        # Collect the actual unique tags from all cards
        all_tags: set[str] = {"anki-gen"}
        for card in cards:
            if isinstance(card, DefinitionCard):
                all_tags.add("definition")
            all_tags.update(card.tags)
        tag_str = ", ".join(f"{_C_TAG_VAL}{t}{_R}" for t in sorted(all_tags))
        print(f"  Tags: {tag_str}\n")
    else:
        print(
            f"  {_C_NUM}{len(cards)}{_R} card(s) generated for {_C_FILENAME}'{deck_name}'{_R}."
        )


# ---------------------------------------------------------------------------
# Loading animation
# ---------------------------------------------------------------------------

_ANIM_WIDTH = 8
_ANIM_TRAIL = _ANIM_WIDTH - 2  # = 6; trail 2 shorter than the bar
_ANIM_FPS = 0.04  # ~33 fps
_ANIM_PAUSE = 0.35  # hold at cycle end (thinking effect)
_ANIM_R, _ANIM_G, _ANIM_B = 64, 156, 255  # base blue
_ANIM_DOT_BRIGHT = 0.35  # mid-tone dots (inactive positions)
_ANIM_EDGE_BRIGHT = 0.40  # brightness at the very edge of the trail

_ANIM_COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (64, 156, 255),
    "indigo": (120, 80, 255),
    "orange": (255, 145, 30),
    "red": (238, 80, 80),
    "green": (60, 190, 120),
}

_apply_accent_colors()  # initialise with default blue


def _dot_char() -> str:
    r, g, b = (
        int(_ANIM_R * _ANIM_DOT_BRIGHT),
        int(_ANIM_G * _ANIM_DOT_BRIGHT),
        int(_ANIM_B * _ANIM_DOT_BRIGHT),
    )
    return f"\x1b[38;2;{r};{g};{b}m⬝\x1b[0m"


def _block_char(trail_pos: int) -> str:
    """Colored ■ — bright at centre of trail, darker toward the edges."""
    centre = (_ANIM_TRAIL - 1) / 2.0
    norm = abs(trail_pos - centre) / centre  # 0 = centre, 1 = edge
    brightness = _ANIM_EDGE_BRIGHT + (1.0 - _ANIM_EDGE_BRIGHT) * (1.0 - norm)
    r, g, b = (
        int(_ANIM_R * brightness),
        int(_ANIM_G * brightness),
        int(_ANIM_B * brightness),
    )
    return f"\x1b[38;2;{r};{g};{b}m■\x1b[0m"


def _render_anim_frame(offset: int) -> str:
    """Render one bar frame. *offset* is the left edge of the trail (may be out of range)."""
    dot = _dot_char()
    return "".join(
        _block_char(i - offset) if 0 <= i - offset < _ANIM_TRAIL else dot
        for i in range(_ANIM_WIDTH)
    )


def _animate_loading(message: str, stop: threading.Event) -> None:
    """
    Trail sweeps left→right, then right→left (ping-pong). After each full
    round trip the display holds on all-dots for _ANIM_PAUSE seconds to
    suggest thinking rather than plain loading.
    """
    period = 2 * (_ANIM_WIDTH + _ANIM_TRAIL)
    frame = 0

    sys.stdout.write("\x1b[?25l")  # hide cursor for the duration
    sys.stdout.flush()
    try:
        while not stop.is_set():
            t = frame % period
            if t < _ANIM_WIDTH + _ANIM_TRAIL:
                offset = -(_ANIM_TRAIL - 1) + t  # forward: -(trail-1) → width
            else:
                offset = (_ANIM_WIDTH - 1) - (
                    t - (_ANIM_WIDTH + _ANIM_TRAIL)
                )  # backward

            sys.stdout.write(f"\r  {message}  {_render_anim_frame(offset)}\x1b[0m")
            sys.stdout.flush()
            frame += 1

            if frame % period == 0:
                # End of round trip — hold on all-dots, sleep in small slices
                # so we stay responsive to the stop event.
                deadline = time.monotonic() + _ANIM_PAUSE
                while time.monotonic() < deadline:
                    if stop.is_set():
                        break
                    time.sleep(_ANIM_FPS)
            else:
                time.sleep(_ANIM_FPS)

        # End state: all dots (offset = width → nothing in range)
        sys.stdout.write(f"\r  {message}  {_render_anim_frame(_ANIM_WIDTH)}\x1b[0m\n")
        sys.stdout.flush()
    finally:
        sys.stdout.write("\x1b[?25h")  # always restore cursor
        sys.stdout.flush()


def _run_with_animation(message: str, fn, *args, **kwargs):
    """Run *fn* in a background thread while showing the loading animation."""
    result: list = []
    error: list = []
    stop = threading.Event()

    def _worker():
        try:
            result.append(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            stop.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        _animate_loading(message, stop)
    except KeyboardInterrupt:
        stop.set()
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()
        raise
    t.join()

    if error:
        raise error[0]
    return result[0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

# Short flags that are boolean (store_true) — can be combined as -vcp etc.
_BOOL_SHORTS = frozenset("vcp")


def _expand_argv(argv: list[str]) -> list[str]:
    """Expand combined bool short flags, e.g. -cv → -c -v."""
    result = []
    for arg in argv:
        if (
            len(arg) > 2
            and arg[0] == "-"
            and arg[1] != "-"
            and all(c in _BOOL_SHORTS for c in arg[1:])
        ):
            result.extend(f"-{c}" for c in arg[1:])
        else:
            result.append(arg)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anki-gen",
        description=(
            "Generate Anki flashcards from Markdown notes using an LLM.\n\n"
            "Pass one or more .md file paths and/or directory paths. "
            "Directories are traversed recursively for .md files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="One or more .md files or directories containing .md files.",
    )

    # Output / push
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help=(
            "Write an Anki-importable .apkg to this path. "
            "Defaults to <first-file-stem>.apkg in the current directory."
        ),
    )
    output_group.add_argument(
        "-p",
        "--push",
        action="store_true",
        help=(
            "Push cards directly into a running Anki instance via AnkiConnect "
            "(requires the AnkiConnect add-on and Anki to be open)."
        ),
    )

    parser.add_argument(
        "-d",
        "--deck",
        metavar="NAME",
        help=(
            "Deck name to use when --push is active, or a single deck name to "
            "bundle all cards into when writing an .apkg. "
            "Defaults to each file's title (one deck per file) when omitted."
        ),
    )

    # Confirm / review
    parser.add_argument(
        "-c",
        "--confirm",
        action="store_true",
        help=(
            "Two-phase mode: first extract a concept list from the LLM and "
            "open an interactive TUI to review, edit, or delete concepts "
            "before generating cards. Requires a TTY."
        ),
    )

    # LLM
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use for card generation (default: openai).",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL_ID",
        help=(
            "Override the model used by the selected provider "
            "(e.g. gpt-4o-mini, claude-3-haiku-20240307)."
        ),
    )

    # Card count
    parser.add_argument(
        "-n",
        "--max-cards",
        type=int,
        metavar="N",
        help=(
            "Maximum number of cards to generate per file. "
            "When omitted, the ceiling is derived automatically from the "
            "document's concept density."
        ),
    )

    parser.add_argument(
        "--tags",
        metavar="TAGS",
        default=None,
        help=(
            "Comma-separated tags applied to all generated cards "
            "(e.g. --tags math,calculus,linear-algebra). "
            "Spaces within a tag are replaced with hyphens."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress, concept lists, and a card-type summary.",
    )

    parser.add_argument(
        "--color",
        choices=["blue", "indigo", "orange", "red", "green"],
        default="blue",
        metavar="COLOR",
        help=(
            "Loading animation color: blue (default), indigo, orange, red, or green."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Tag application helper
# ---------------------------------------------------------------------------


def _with_tags(
    cards: list[Card],
    concepts: list[str] | None,
    concept_tags: dict[str, list[str]] | None,
    global_tags: list[str],
) -> list[Card]:
    """
    Return a new list of cards with per-concept and global tags applied.

    *concepts* and *concept_tags* are used together in the confirm path to
    assign per-concept tags positionally (card[i] ← concepts[i] tags).
    *global_tags* are merged onto every card regardless of path.
    """
    result: list[Card] = []
    for i, card in enumerate(cards):
        per_tags: list[str] = []
        if concepts and concept_tags and i < len(concepts):
            per_tags = concept_tags.get(concepts[i], [])
        all_tags = global_tags + per_tags
        if not all_tags:
            result.append(card)
        elif isinstance(card, BasicCard):
            result.append(
                BasicCard(
                    front=card.front,
                    back=card.back,
                    reversed=card.reversed,
                    tags=all_tags,
                )
            )
        elif isinstance(card, DefinitionCard):
            result.append(
                DefinitionCard(
                    term=card.term,
                    definition=card.definition,
                    tags=all_tags,
                )
            )
        else:
            result.append(card)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args(_expand_argv(sys.argv[1:]))

    global _ANIM_R, _ANIM_G, _ANIM_B
    _ANIM_R, _ANIM_G, _ANIM_B = _ANIM_COLORS[args.color]
    _apply_accent_colors()
    from anki_gen.confirm import set_accent as _set_accent

    _set_accent(_ANIM_R, _ANIM_G, _ANIM_B)

    # Parse --tags into a normalised list
    global_tags: list[str] = []
    if args.tags:
        global_tags = [
            t.strip().replace(" ", "-") for t in args.tags.split(",") if t.strip()
        ]

    # ------------------------------------------------------------------
    # Resolve input paths
    # ------------------------------------------------------------------
    raw_paths = [Path(p) for p in args.paths]
    try:
        md_files = collect_markdown_files(raw_paths)
    except ValueError as exc:
        parser.error(str(exc))

    if not md_files:
        parser.error("No .md files found at the provided paths.")

    if args.verbose:
        print(f"Found {len(md_files)} file(s) to process.")

    # ------------------------------------------------------------------
    # Guard: --confirm requires a real TTY
    # ------------------------------------------------------------------
    if args.confirm and not sys.stdin.isatty():
        parser.error("--confirm requires an interactive terminal (TTY).")

    # ------------------------------------------------------------------
    # Build provider
    # ------------------------------------------------------------------
    try:
        provider = _build_provider(args.provider, args.model)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Using provider: {provider.name}")

    # ------------------------------------------------------------------
    # Validate AnkiConnect before any LLM work
    # ------------------------------------------------------------------
    if args.push:
        try:
            ankiconnect.check_connection()
        except ankiconnect.AnkiConnectError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print("AnkiConnect: connected.")

    # ------------------------------------------------------------------
    # Process each file
    # ------------------------------------------------------------------
    cards_by_deck: dict[str, list[Card]] = {}

    try:
        for idx, md_path in enumerate(md_files, 1):
            try:
                doc: ParsedDocument = parse_file(md_path)
            except Exception as exc:
                print(f"warning: Failed to parse {md_path}: {exc}", file=sys.stderr)
                continue

            if not doc.plain_text.strip():
                print(
                    f"warning: {md_path.name} has no content, skipping.",
                    file=sys.stderr,
                )
                continue

            deck_name = args.deck or doc.title

            if args.confirm:
                # --------------------------------------------------------
                # Phase 1: extract concept list
                # --------------------------------------------------------
                print(
                    f"{_C_ACCENT}[{idx}/{len(md_files)}]{_R} "
                    f"{_C_FILENAME}{md_path.name}{_R} — "
                    f"extracting concepts "
                    f"({_C_FILENAME}{doc.concept_count} signals{_R})..."
                )
                try:
                    concepts = _run_with_animation(
                        "Extracting concepts",
                        extract_concepts,
                        doc,
                        provider,
                        max_cards=args.max_cards,
                    )
                except Exception as exc:
                    print(
                        f"warning: Concept extraction failed for {md_path}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # --------------------------------------------------------
                # Interactive TUI review
                # --------------------------------------------------------
                from anki_gen.confirm import review_concepts  # lazy import (needs TTY)

                print()
                try:
                    confirmed, reversed_concepts, concept_tags, notes = review_concepts(
                        doc.title, concepts
                    )
                except SystemExit:
                    print("Aborted.")
                    sys.exit(0)

                if not confirmed:
                    print(f"  No concepts confirmed for '{md_path.name}', skipping.")
                    continue

                print(f"  Confirmed {len(confirmed)} concept(s):")
                for i, c in enumerate(confirmed, 1):
                    rev_marker = f"  ⇄" if c in reversed_concepts else ""
                    tag_marker = (
                        f"  [{', '.join(concept_tags[c])}]" if c in concept_tags else ""
                    )
                    print(f"     {i:>2}. {c}{rev_marker}{tag_marker}")

                if notes and args.verbose:
                    print(f"  Notes ({len(notes)}):")
                    for note in notes:
                        print(f"       • {note}")

                # --------------------------------------------------------
                # Phase 2: generate from confirmed concepts
                # --------------------------------------------------------
                print()
                try:
                    cards = _run_with_animation(
                        "Generating cards",
                        generate_cards_from_concepts,
                        doc,
                        confirmed,
                        provider,
                        notes=notes or None,
                        reversed_concepts=reversed_concepts or None,
                    )
                except Exception as exc:
                    print(
                        f"warning: Card generation failed for {md_path}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # Apply per-concept tags and global tags
                cards = _with_tags(cards, confirmed, concept_tags, global_tags)

            else:
                # --------------------------------------------------------
                # Single-call path (no --confirm)
                # --------------------------------------------------------
                if args.verbose:
                    print(
                        f"{_C_ACCENT}[{idx}/{len(md_files)}]{_R} "
                        f"{_C_FILENAME}{md_path.name}{_R} — "
                        f"{_C_FILENAME}{doc.concept_count} concept signals{_R}"
                        f" → deck '{deck_name}'"
                    )

                try:
                    cards = _run_with_animation(
                        f"Generating cards via {provider.name}",
                        generate_cards,
                        doc,
                        provider,
                        max_cards=args.max_cards,
                    )
                except Exception as exc:
                    print(
                        f"warning: Card generation failed for {md_path}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                # Apply global tags (no per-concept tags in direct mode)
                if global_tags:
                    cards = _with_tags(cards, None, None, global_tags)

            _print_summary(deck_name, cards, verbose=args.verbose)

            if deck_name not in cards_by_deck:
                cards_by_deck[deck_name] = []
            cards_by_deck[deck_name].extend(cards)

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Abort if nothing was generated
    # ------------------------------------------------------------------
    if not cards_by_deck:
        print("No cards were generated.", file=sys.stderr)
        sys.exit(1)

    total_cards = sum(len(v) for v in cards_by_deck.values())

    # ------------------------------------------------------------------
    # Deliver: push via AnkiConnect or write .apkg
    # ------------------------------------------------------------------
    if args.push:
        success = True
        for deck_name, cards in cards_by_deck.items():
            try:
                is_new = not ankiconnect.deck_exists(deck_name)
                added, skipped_fronts = ankiconnect.push_cards(deck_name, cards)
                total = ankiconnect.get_deck_card_count(deck_name)
                if is_new:
                    print(
                        f"Created deck {_C_FILENAME}'{deck_name}'{_R} — "
                        f"{_C_NUM}{added}{_R} card(s) added."
                    )
                else:
                    msg = (
                        f"Pushed to {_C_FILENAME}'{deck_name}'{_R} — "
                        f"{_C_NUM}{added}{_R} added, "
                        f"{_C_NUM}{total}{_R} total."
                    )
                    if skipped_fronts:
                        msg += f" ({_C_NUM}{len(skipped_fronts)}{_R} duplicate(s) skipped.)"
                    print(msg)
                if skipped_fronts:
                    for front in skipped_fronts:
                        print(f"    {_C_TAG_VAL}↳ (duplicate) {front}{_R}")
            except ankiconnect.AnkiConnectError as exc:
                print(f"error pushing to '{deck_name}': {exc}", file=sys.stderr)
                success = False
        if not success:
            sys.exit(1)
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            stem = md_files[0].stem if len(md_files) == 1 else "cards"
            output_path = Path(stem + ".apkg")

        try:
            export_apkg(cards_by_deck, output_path)
        except Exception as exc:
            print(f"error: Failed to write .apkg: {exc}", file=sys.stderr)
            sys.exit(1)

        if len(cards_by_deck) > 1:
            print(
                f"Exported {_C_NUM}{total_cards}{_R} card(s) across "
                f"{_C_NUM}{len(cards_by_deck)}{_R} deck(s)"
                f" → {_C_FILENAME}{output_path}{_R}"
            )
        else:
            print(
                f"Exported {_C_NUM}{total_cards}{_R} card(s)"
                f" → {_C_FILENAME}{output_path}{_R}"
            )


if __name__ == "__main__":
    main()
