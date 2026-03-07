from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt


@dataclass
class ImageRef:
    """A single image reference extracted from a Markdown document."""

    filename: str  # basename only, e.g. "diagram.webp"
    alt_text: str  # caption / alt text, or "" if absent
    width: int | None  # explicit width in pixels, or None
    height: int | None  # explicit height in pixels, or None
    resolved_path: Path | None  # absolute path to the file, or None if not found


@dataclass
class ParsedDocument:
    """Structured representation of a Markdown document."""

    source_path: Path
    title: str
    plain_text: str
    concept_count: int
    images: list[ImageRef] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

# Common Obsidian / Markdown vault attachment folder names, searched in order.
_IMAGE_SEARCH_DIRS: tuple[str, ...] = (
    ".",
    "attachments",
    "assets",
    "_resources",
    "images",
)

# Obsidian wikilink image: ![[filename|alt|size]]  (pipes and size optional)
_OBSIDIAN_IMAGE_RE = re.compile(r"!\[\[([^\]\|]+?)(?:\|([^\]\|]*))?(?:\|([^\]]*))?\]\]")

# Standard Markdown image: ![alt](path)
_STANDARD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)")

# A string that looks like an Obsidian size spec: "386", "386x14", "x14"
_SIZE_RE = re.compile(r"^\d*(x\d+)?$")


def _is_size_field(s: str) -> bool:
    """Return True if *s* looks like an Obsidian dimension spec (not alt text)."""
    s = s.strip()
    return bool(_SIZE_RE.match(s)) and bool(s)


def _parse_obsidian_size(s: str) -> tuple[int | None, int | None]:
    """Parse "386", "386x14", or "x14" → (width | None, height | None)."""
    s = s.strip().lower()
    if "x" in s:
        left, right = s.split("x", 1)
        w = int(left) if left.isdigit() else None
        h = int(right) if right.isdigit() else None
        return w, h
    return (int(s) if s.isdigit() else None), None


def _resolve_image(filename: str, doc_dir: Path) -> Path | None:
    """
    Search for *filename* in the document directory and common attachment
    subdirectories. Returns the first absolute path found, or None.
    """
    for subdir in _IMAGE_SEARCH_DIRS:
        candidate = (doc_dir / subdir / filename).resolve()
        if candidate.is_file():
            return candidate
    return None


def _image_ref_to_text(ref: ImageRef) -> str:
    """Render an ImageRef as a plain-text marker for the LLM context."""
    size_part = ""
    if ref.width is not None and ref.height is not None:
        size_part = f", {ref.width}x{ref.height}px"
    elif ref.width is not None:
        size_part = f", {ref.width}px wide"
    if ref.alt_text:
        return f'[Image: "{ref.alt_text}" ({ref.filename}{size_part})]'
    return f"[Image: {ref.filename}{size_part}]"


def _substitute_images(source: str, doc_dir: Path) -> tuple[str, list[ImageRef]]:
    """
    Find all image references in *source* (both Obsidian wikilink and standard
    Markdown formats), replace each with a plain-text LLM marker, and return
    the modified source together with a list of ImageRef objects.

    The substitution preserves the in-document position of each reference so
    the LLM can use proximity to infer which concept each image belongs to.
    """
    refs: list[ImageRef] = []
    replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

    # --- Obsidian wikilink images: ![[filename|alt|size]] ---
    for m in _OBSIDIAN_IMAGE_RE.finditer(source):
        raw_filename = m.group(1).strip()
        field2 = (m.group(2) or "").strip()
        field3 = (m.group(3) or "").strip()

        filename = Path(raw_filename).name  # ensure only basename

        # Disambiguate: field2 may be alt text or a size spec (if only 2 fields)
        if field3:
            # Three fields: filename | alt | size
            alt_text = field2
            w, h = _parse_obsidian_size(field3) if field3 else (None, None)
        elif field2 and _is_size_field(field2):
            # Two fields: filename | size (no alt)
            alt_text = ""
            w, h = _parse_obsidian_size(field2)
        else:
            # Two fields: filename | alt  (or just one field)
            alt_text = field2
            w, h = None, None

        resolved = _resolve_image(filename, doc_dir)
        ref = ImageRef(
            filename=filename,
            alt_text=alt_text,
            width=w,
            height=h,
            resolved_path=resolved,
        )
        refs.append(ref)
        replacements.append((m.start(), m.end(), _image_ref_to_text(ref)))

    # --- Standard Markdown images: ![alt](path) ---
    for m in _STANDARD_IMAGE_RE.finditer(source):
        # Skip if already covered by an Obsidian match (shouldn't overlap, but guard)
        start, end = m.start(), m.end()
        if any(s <= start < e for s, e, _ in replacements):
            continue
        alt_text = m.group(1).strip()
        raw_path = m.group(2).strip()
        filename = Path(raw_path).name
        resolved = _resolve_image(filename, doc_dir)
        ref = ImageRef(
            filename=filename,
            alt_text=alt_text,
            width=None,
            height=None,
            resolved_path=resolved,
        )
        refs.append(ref)
        replacements.append((start, end, _image_ref_to_text(ref)))

    if not replacements:
        return source, refs

    # Apply replacements in reverse order so offsets stay valid
    replacements.sort(key=lambda x: x[0], reverse=True)
    result = source
    for start, end, text in replacements:
        result = result[:start] + text + result[end:]

    return result, refs


def _strip_images(source: str) -> str:
    """Remove Markdown/Obsidian image syntax entirely."""
    source = _OBSIDIAN_IMAGE_RE.sub("", source)
    source = _STANDARD_IMAGE_RE.sub("", source)
    return source


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


# Matches a YAML frontmatter block at the very start of a file.
_FRONTMATTER_RE = re.compile(
    r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL
)


def _extract_frontmatter_tags(source: str) -> tuple[str, list[str]]:
    """Strip YAML frontmatter and return (body, tags).

    Supports these frontmatter tag formats:
        tags: [tag1, tag2]
        tags:
          - tag1
          - tag2
        tags: tag1, tag2

    Returns (source_without_frontmatter, normalised_tag_list).
    Tags are lower-cased and spaces replaced with hyphens.
    If no frontmatter or no tags field, returns (source, []).
    """
    m = _FRONTMATTER_RE.match(source)
    if not m:
        return source, []

    body = source[m.end() :]
    yaml_block = m.group(1)

    try:
        import yaml  # pyyaml — available as transitive dep

        data: Any = yaml.safe_load(yaml_block)
    except Exception:
        return body, []

    if not isinstance(data, dict):
        return body, []

    raw = data.get("tags")
    if raw is None:
        return body, []

    # Normalise to list[str]
    if isinstance(raw, list):
        tag_list = [str(t) for t in raw]
    elif isinstance(raw, str):
        tag_list = [t for t in (t.strip() for t in raw.split(",")) if t]
    else:
        tag_list = [str(raw)]

    normalised = [t.strip().replace(" ", "-").lower() for t in tag_list if t.strip()]
    return body, normalised


def parse_file(path: Path, images_enabled: bool = True) -> ParsedDocument:
    """Parse a single Markdown file into a ParsedDocument."""
    md = MarkdownIt()
    raw_source = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter and collect any declared tags.
    source, source_tags = _extract_frontmatter_tags(raw_source)

    # Replace image syntax with plain-text markers before tokenizing so that
    # the markers appear at the correct position in plain_text (preserving
    # proximity information for the LLM).
    if images_enabled:
        normalized_source, images = _substitute_images(source, path.parent)
    else:
        normalized_source, images = _strip_images(source), []

    tokens = md.parse(normalized_source)

    title = _extract_title(tokens, fallback=path.stem)
    plain_text = _tokens_to_plain_text(tokens)
    concept_count = _count_concepts(tokens)

    return ParsedDocument(
        source_path=path,
        title=title,
        plain_text=plain_text,
        concept_count=concept_count,
        images=images,
        source_tags=source_tags,
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
