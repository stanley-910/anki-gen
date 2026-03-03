from __future__ import annotations

"""
Inline concept-review TUI (fzf-style).

Renders directly in the terminal at the current cursor position without
taking over the full screen.  Uses raw termios input so arrow keys, inline
editing, and wrapping navigation all work without any extra dependencies.

Public API: review_concepts(title, concepts)
  → tuple[list[str], set[str], dict[str, list[str]], list[str]]
  Returns (confirmed_concepts, reversed_concepts, concept_tags, notes) where:
    - confirmed_concepts: non-deleted concept strings
    - reversed_concepts:  set of confirmed concept strings marked for reversal
    - concept_tags:       dict mapping concept string → per-concept tag list
    - notes:              non-deleted note strings (general LLM instructions)
"""

import select
import shutil
import sys
import termios
import tty
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_R = "\x1b[0m"  # reset
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_ITALIC = "\x1b[3m"
_STRIKE = "\x1b[9m"
_REV = "\x1b[7m"  # reverse video  (used for edit cursor)

_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_ERASE_LINE = "\r\x1b[K"
_ERASE_DOWN = "\x1b[J"


def _fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def _bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


_C_HEADER = _fg(100, 180, 255) + _BOLD
_C_CURSOR = _fg(255, 255, 255) + _BOLD + _bg(0, 95, 135)
_C_DELETED = _fg(110, 110, 110) + _STRIKE + _DIM
_C_EDITED = _fg(170, 221, 255) + _ITALIC
_C_HINT = _fg(110, 110, 110) + _ITALIC
_C_BULLET_DEL = _fg(110, 110, 110)

# Reversal indicator — soft green, fixed semantic colour
_C_REVERSED = _fg(80, 215, 135) + _BOLD

# Inline tag display — teal, fixed semantic colour
_C_TAG_INLINE = _fg(75, 195, 195)

# Note items get a warm amber style to visually distinguish them as
# instructions rather than study content.
_C_NOTE_NORMAL = _fg(255, 195, 60) + _ITALIC
_C_NOTE_CURSOR = _fg(255, 255, 255) + _BOLD + _bg(90, 55, 0)
_C_DIVIDER = _fg(70, 70, 70)

# Reversal symbol (UNO reverse style — left/right arrows)
_REV_SYMBOL = "⇄"


def set_accent(r: int, g: int, b: int) -> None:
    """Recalculate accent-dependent colour strings from a base RGB triple."""
    global _C_HEADER, _C_CURSOR
    rh = min(255, int(r + (255 - r) * 0.4))
    gh = min(255, int(g + (255 - g) * 0.4))
    bh = min(255, int(b + (255 - b) * 0.4))
    _C_HEADER = _fg(rh, gh, bh) + _BOLD
    rb, gb, bb = int(r * 0.4), int(g * 0.4), int(b * 0.4)
    _C_CURSOR = _fg(255, 255, 255) + _BOLD + _bg(rb, gb, bb)


_BULLET = "●"
_NOTE_BULLET = "◆"
_BULL_DEL = "✗"
_PREFIX_W = 4  # len("  ● ")


# ---------------------------------------------------------------------------
# Tag normalisation
# ---------------------------------------------------------------------------


def _normalize_tag(t: str) -> str:
    """Strip surrounding whitespace and replace internal spaces with hyphens."""
    return t.strip().replace(" ", "-")


# ---------------------------------------------------------------------------
# Raw-mode context manager
# ---------------------------------------------------------------------------


@contextmanager
def _raw_mode():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    # Disable echo + canonical; keep output processing (ONLCR) intact
    new[3] &= ~(termios.ECHO | termios.ICANON)
    new[6][termios.VMIN] = 1  # type: ignore[index]
    new[6][termios.VTIME] = 0  # type: ignore[index]
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> str:
    """Read one logical keypress, including multi-byte escape sequences."""
    try:
        ch = sys.stdin.read(1)
    except KeyboardInterrupt:
        return "\x03"
    if ch != "\x1b":
        return ch
    # Peek for the rest of an escape sequence with a short timeout
    if not select.select([sys.stdin], [], [], 0.05)[0]:
        return "\x1b"
    ch2 = sys.stdin.read(1)
    if ch2 == "O":
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            return "\x1bO"
        ch3 = sys.stdin.read(1)
        return "\x1bO" + ch3
    if ch2 != "[":
        return "\x1b" + ch2
    seq = ""
    while True:
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            break
        c = sys.stdin.read(1)
        seq += c
        if c.isalpha() or c == "~":
            break
    return "\x1b[" + seq


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class _Item:
    __slots__ = ("text", "deleted", "edited", "kind", "reversed", "tags")

    def __init__(self, text: str, kind: str = "concept") -> None:
        self.text = text
        self.deleted = False
        self.edited = False
        self.kind = kind  # "concept" or "note"
        self.reversed = False
        self.tags: list[str] = []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _render_item(
    item: _Item,
    global_idx: int,
    cursor: int,
    edit_mode: str,  # "none" | "concept" | "tags"
    edit_buf: str,
    edit_pos: int,
    width: int,
) -> str:
    is_cur = global_idx == cursor
    is_note = item.kind == "note"
    bullet = _NOTE_BULLET if is_note else _BULLET

    # ── Editing concept (inline text edit) ──────────────────────────────────
    if edit_mode == "concept" and is_cur:
        before = edit_buf[:edit_pos]
        at = edit_buf[edit_pos] if edit_pos < len(edit_buf) else " "
        after = edit_buf[edit_pos + 1 :] if edit_pos < len(edit_buf) else ""
        text_color = _C_NOTE_NORMAL if is_note else _C_EDITED
        return f"  {text_color}{bullet} {before}{_REV}{at}{_R}{text_color}{after}{_R}"

    # ── Editing tags (tag field to the right; concept is read-only) ─────────
    if edit_mode == "tags" and is_cur and not is_note:
        before = edit_buf[:edit_pos]
        at = edit_buf[edit_pos] if edit_pos < len(edit_buf) else " "
        after = edit_buf[edit_pos + 1 :] if edit_pos < len(edit_buf) else ""
        concept_part = f"  {bullet} {item.text}"
        # Keep reversal indicator visible while editing tags
        if item.reversed:
            concept_part += f"  {_C_REVERSED}{_REV_SYMBOL}{_R}"
        tag_field = f"{before}{_REV}{at}{_R}{_C_TAG_INLINE}{after}"
        return f"{concept_part}   {_C_TAG_INLINE}[{tag_field}]{_R}"

    # ── Deleted item ─────────────────────────────────────────────────────────
    if item.deleted:
        raw = f"  {_BULL_DEL} {item.text}"
        if is_cur:
            # Dim reverse-video so cursor position is always visible on
            # deleted items — prevents accidental blind re-toggles.
            pad = max(0, width - len(raw))
            return f"{_DIM}{_REV}{raw}{' ' * pad}{_R}"
        return f"  {_C_BULLET_DEL}{_BULL_DEL}{_R} {_C_DELETED}{item.text}{_R}"

    # ── Build the visual (ANSI-free) plain content for padding calculation ───
    raw_line = f"  {bullet} {item.text}"
    vis_suffix = ""
    if not is_note and item.reversed:
        vis_suffix += f"  {_REV_SYMBOL}"
    if item.tags:
        vis_suffix += f"  [{', '.join(item.tags)}]"

    # ── Current cursor (navigation mode) ────────────────────────────────────
    if is_cur:
        full_plain = raw_line + vis_suffix
        pad = max(0, width - len(full_plain))
        c_cur = _C_NOTE_CURSOR if is_note else _C_CURSOR
        return f"{c_cur}{full_plain}{' ' * pad}{_R}"

    # ── Normal (non-cursor, non-deleted) ─────────────────────────────────────
    if is_note:
        base = f"  {bullet} {_C_NOTE_NORMAL}{item.text}{_R}"
        # Notes don't get reversal or concept tags
        return base

    result = f"  {bullet} "
    if item.edited:
        result += f"{_C_EDITED}{item.text}{_R}"
    else:
        result += item.text
    if item.reversed:
        result += f"  {_C_REVERSED}{_REV_SYMBOL}{_R}"
    if item.tags:
        tag_str = ", ".join(item.tags)
        result += f"  {_C_TAG_INLINE}[{tag_str}]{_R}"
    return result


def _draw(
    concept_items: list[_Item],
    note_items: list[_Item],
    cursor: int,
    edit_mode: str,
    edit_buf: str,
    edit_pos: int,
    title: str,
    prev_total: int,
) -> int:
    """
    Draw the full TUI. Returns the number of lines rendered.

    prev_total is the line count from the previous render (0 on first draw).
    On subsequent draws we move up prev_total lines, erase to end of screen,
    then write the new frame — this correctly handles variable-height renders
    caused by adding/removing concept or note items.
    """
    nc = len(concept_items)
    nn = len(note_items)
    has_notes = nn > 0
    w = _term_width()

    kept = sum(1 for it in concept_items if not it.deleted)
    header = f"{_C_HEADER}  Anki Concept Review  ({kept}/{nc} selected){_R}"

    editing = edit_mode != "none"
    hints = (
        f"{_C_HINT}  editing — Enter save   Esc cancel{_R}"
        if editing
        else (
            f"{_C_HINT}  j/k nav   d del   e edit   r rev   R rev-all   t tags   "
            f"a/n add   Enter confirm   q quit{_R}"
        )
    )

    lines: list[str] = [header]
    for i, item in enumerate(concept_items):
        lines.append(_render_item(item, i, cursor, edit_mode, edit_buf, edit_pos, w))

    if has_notes:
        lines.append(f"  {_C_DIVIDER}── notes ──{_R}")
        for i, item in enumerate(note_items):
            lines.append(
                _render_item(item, nc + i, cursor, edit_mode, edit_buf, edit_pos, w)
            )

    lines.append(hints)

    out: list[str] = []
    if prev_total > 0:
        out.append(f"\x1b[{prev_total}A{_ERASE_DOWN}")
    for line in lines:
        out.append(line + "\n")

    sys.stdout.write("".join(out))
    sys.stdout.flush()
    return len(lines)


def _erase(total_lines: int) -> None:
    sys.stdout.write(f"\x1b[{total_lines}A{_ERASE_DOWN}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def review_concepts(
    title: str, concepts: list[str]
) -> tuple[list[str], set[str], dict[str, list[str]], list[str]]:
    """
    Open the inline TUI. Returns (confirmed_concepts, reversed_concepts,
    concept_tags, notes) where:
      - confirmed_concepts: non-deleted concept strings
      - reversed_concepts:  set of confirmed concept strings marked for reversal
      - concept_tags:       dict mapping concept string → per-concept tag list
      - notes:              non-deleted note strings (general instructions for the LLM)

    Raises SystemExit(0) if the user aborts with q / Ctrl-C.

    Key bindings (navigation mode):
      j / ↓     move down
      k / ↑     move up
      d         toggle delete on current item (advances cursor)
      e         edit current item inline
      r         toggle reversal on current concept (advances cursor)
      R         toggle-all reversal on non-deleted concepts
      t         edit tags for current concept (separate field, read-only concept)
      a         add a new concept (opens inline editor)
      n         add a new note / instruction (opens inline editor)
      Enter     confirm and return
      q         quit (abort)
    """
    concept_items: list[_Item] = [_Item(c, "concept") for c in concepts]
    note_items: list[_Item] = []
    cursor = 0
    edit_mode = "none"  # "none" | "concept" | "tags"
    edit_buf = ""
    edit_pos = 0
    prev_total = 0

    def _nc() -> int:
        return len(concept_items)

    def _nn() -> int:
        return len(note_items)

    def _total() -> int:
        return _nc() + _nn()

    def _item_at(c: int) -> tuple[list[_Item], int]:
        nc = _nc()
        if c < nc:
            return concept_items, c
        return note_items, c - nc

    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    result = None
    try:
        with _raw_mode():
            prev_total = _draw(
                concept_items,
                note_items,
                cursor,
                edit_mode,
                edit_buf,
                edit_pos,
                title,
                prev_total,
            )

            while result is None:
                key = _read_key()

                if edit_mode != "none":
                    items, idx = _item_at(cursor)
                    item = items[idx]

                    if key in ("\r", "\n"):
                        if edit_mode == "concept":
                            text = edit_buf.strip()
                            if text:
                                item.text = text
                                item.edited = True
                            else:
                                # Empty save — discard the item
                                items.pop(idx)
                                tot = _total()
                                cursor = min(cursor, tot - 1) if tot > 0 else 0
                        elif edit_mode == "tags":
                            item.tags = [
                                _normalize_tag(t)
                                for t in edit_buf.split(",")
                                if _normalize_tag(t)
                            ]
                        edit_mode = "none"

                    elif key == "\x1b":
                        if edit_mode == "concept":
                            # Cancel — if item was freshly added (no text yet), remove it
                            if not item.text:
                                items.pop(idx)
                                tot = _total()
                                cursor = min(cursor, tot - 1) if tot > 0 else 0
                        # For tags: just cancel, original tags unchanged
                        edit_mode = "none"

                    elif key == "\x1b[C":  # →
                        edit_pos = min(edit_pos + 1, len(edit_buf))

                    elif key == "\x1b[D":  # ←
                        edit_pos = max(edit_pos - 1, 0)

                    elif key in ("\x1b[H", "\x01"):  # Home / Ctrl-A
                        edit_pos = 0

                    elif key in ("\x1b[F", "\x05"):  # End / Ctrl-E
                        edit_pos = len(edit_buf)

                    elif key in (
                        "\x17",
                        "\x1b\x7f",
                    ):  # Ctrl-W / Option+Backspace — kill word back
                        pos = edit_pos
                        while pos > 0 and edit_buf[pos - 1] == " ":
                            pos -= 1
                        while pos > 0 and edit_buf[pos - 1] != " ":
                            pos -= 1
                        edit_buf = edit_buf[:pos] + edit_buf[edit_pos:]
                        edit_pos = pos

                    elif key in ("\x7f", "\x08"):  # Backspace
                        if edit_pos > 0:
                            edit_buf = edit_buf[: edit_pos - 1] + edit_buf[edit_pos:]
                            edit_pos -= 1

                    elif key == "\x1b[3~":  # Delete
                        edit_buf = edit_buf[:edit_pos] + edit_buf[edit_pos + 1 :]

                    elif len(key) == 1 and ord(key) >= 32:  # printable
                        edit_buf = edit_buf[:edit_pos] + key + edit_buf[edit_pos:]
                        edit_pos += 1

                else:
                    tot = _total()

                    if key in ("j", "\x1b[B", "\x1bOB"):  # down (wrap)
                        if tot > 0:
                            cursor = (cursor + 1) % tot

                    elif key in ("k", "\x1b[A", "\x1bOA"):  # up (wrap)
                        if tot > 0:
                            cursor = (cursor - 1) % tot

                    elif key == "d":  # toggle delete + advance
                        if tot > 0:
                            items, idx = _item_at(cursor)
                            items[idx].deleted = not items[idx].deleted
                            cursor = (cursor + 1) % tot

                    elif key == "r":  # toggle reversal on current concept + advance
                        if tot > 0:
                            items, idx = _item_at(cursor)
                            if items is concept_items:
                                items[idx].reversed = not items[idx].reversed
                            cursor = (cursor + 1) % tot

                    elif key == "R":  # toggle-all reversal on non-deleted concepts
                        non_deleted = [it for it in concept_items if not it.deleted]
                        if non_deleted:
                            all_rev = all(it.reversed for it in non_deleted)
                            for it in non_deleted:
                                it.reversed = not all_rev

                    elif key == "t":  # edit tags on current concept
                        if tot > 0:
                            items, idx = _item_at(cursor)
                            if items is concept_items:
                                edit_mode = "tags"
                                edit_buf = ", ".join(items[idx].tags)
                                edit_pos = len(edit_buf)

                    elif key == "e":  # start inline edit
                        if tot > 0:
                            items, idx = _item_at(cursor)
                            edit_mode = "concept"
                            edit_buf = items[idx].text
                            edit_pos = len(edit_buf)

                    elif key == "a":  # add new concept
                        concept_items.append(_Item("", "concept"))
                        cursor = _nc() - 1
                        edit_mode = "concept"
                        edit_buf = ""
                        edit_pos = 0

                    elif key == "n":  # add new note/instruction
                        note_items.append(_Item("", "note"))
                        cursor = _nc() + _nn() - 1
                        edit_mode = "concept"
                        edit_buf = ""
                        edit_pos = 0

                    elif key in ("\r", "\n"):  # confirm
                        result = "confirm"

                    elif key in ("q", "\x03"):  # quit / Ctrl-C
                        result = "quit"

                if result is None:
                    prev_total = _draw(
                        concept_items,
                        note_items,
                        cursor,
                        edit_mode,
                        edit_buf,
                        edit_pos,
                        title,
                        prev_total,
                    )

    finally:
        _erase(prev_total)
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()

    if result == "quit":
        raise SystemExit(0)

    confirmed_concepts = [it.text for it in concept_items if not it.deleted]
    reversed_concepts: set[str] = {
        it.text for it in concept_items if not it.deleted and it.reversed
    }
    concept_tags: dict[str, list[str]] = {
        it.text: it.tags for it in concept_items if not it.deleted and it.tags
    }
    confirmed_notes = [it.text for it in note_items if not it.deleted]
    return confirmed_concepts, reversed_concepts, concept_tags, confirmed_notes
