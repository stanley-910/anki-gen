from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt


@dataclass
class ParsedDocument:
    """Structured representation of a Markdown document."""

    source_path: Path
    title: str
    plain_text: str
    concept_count: int


def _count_concepts(tokens: list) -> int:
    """
    Heuristically count the number of distinct concepts in a document.

    Concepts are identified as:
    - Level 1-3 headings (each heading signals a new topic)
    - Bold/strong spans (authors typically bold key terms)
    - Items in bullet/ordered lists

    Headings are weighted more heavily than inline emphasis.
    """
    concept_score = 0
    in_heading = False
    heading_level = 0

    for token in tokens:
        if token.type == "heading_open":
            in_heading = True
            heading_level = int(token.tag[1])  # h1 → 1, h2 → 2, etc.
        elif token.type == "heading_close":
            in_heading = False
            # H1 = 3 pts, H2 = 2 pts, H3+ = 1 pt
            concept_score += max(1, 4 - heading_level)
        elif token.type == "strong_open":
            concept_score += 1
        elif token.type == "list_item_open":
            concept_score += 1

    return concept_score


def _derive_max_cards(concept_count: int) -> int:
    """
    Derive a sensible card ceiling from the concept score.

    Formula: floor(concept_count * 0.6) clamped to [5, 80].
    The 0.6 multiplier prevents one-to-one card explosion on large
    documents while ensuring meaningful coverage on small ones.
    A log-based floor guarantees at least a handful of cards even
    for very sparse notes.
    """
    if concept_count == 0:
        return 10  # fallback for empty / plain prose

    raw = math.floor(concept_count * 0.6)
    log_floor = max(5, math.ceil(math.log2(concept_count + 1) * 2))
    return max(log_floor, min(raw, 80))


def _extract_title(tokens: list, fallback: str) -> str:
    """Return the text of the first H1, or fall back to the filename stem."""
    capture_next_inline = False
    for token in tokens:
        if token.type == "heading_open" and token.tag == "h1":
            capture_next_inline = True
        elif capture_next_inline and token.type == "inline":
            return token.content.strip()
    return fallback


def _tokens_to_plain_text(tokens: list) -> str:
    """
    Reconstruct readable plain text from markdown-it token stream,
    preserving heading hierarchy as context signals for the LLM.
    """
    lines: list[str] = []
    current_heading_level = 0

    for token in tokens:
        if token.type == "heading_open":
            current_heading_level = int(token.tag[1])
        elif token.type == "inline" and token.content:
            prefix = "#" * current_heading_level + " " if current_heading_level else ""
            lines.append(f"{prefix}{token.content}")
            current_heading_level = 0
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            pass  # list items are handled via their inline children
        elif token.type == "fence":
            # Preserve code blocks — they may contain important syntax
            lines.append(f"```\n{token.content.strip()}\n```")

    return "\n".join(line for line in lines if line.strip())


def parse_file(path: Path) -> ParsedDocument:
    """Parse a single Markdown file into a ParsedDocument."""
    md = MarkdownIt()
    source = path.read_text(encoding="utf-8")
    tokens = md.parse(source)

    title = _extract_title(tokens, fallback=path.stem)
    plain_text = _tokens_to_plain_text(tokens)
    concept_count = _count_concepts(tokens)

    return ParsedDocument(
        source_path=path,
        title=title,
        plain_text=plain_text,
        concept_count=concept_count,
    )


def collect_markdown_files(paths: list[Path]) -> list[Path]:
    """
    Resolve a mixed list of file and directory paths into a flat list
    of .md files. Directories are traversed non-recursively by default;
    nested subdirectories are included via rglob.
    """
    result: list[Path] = []
    seen: set[Path] = set()

    for p in paths:
        if p.is_dir():
            candidates = sorted(p.rglob("*.md"))
        elif p.suffix.lower() == ".md":
            candidates = [p]
        else:
            raise ValueError(
                f"Unsupported path: {p}. Expected a .md file or a directory."
            )

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(candidate)

    return result
