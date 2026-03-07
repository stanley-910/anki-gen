from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from anki_gen.latex import convert_latex_to_mathjax, promote_sole_inline_to_display
from anki_gen.models import BasicCard, Card, DefinitionCard
from anki_gen.parser import ImageRef, ParsedDocument, _derive_max_cards

if TYPE_CHECKING:
    from anki_gen.llm.base import LLMProvider

# ---------------------------------------------------------------------------
# Context-window budget estimation
# ---------------------------------------------------------------------------

# Known input-token limits keyed by lowercase substrings of provider.name.
# First matching key wins; "_default" is the fallback.
_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-3.5": 16_385,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "claude": 200_000,
    "_default": 128_000,
}

# Warn when estimated prompt tokens exceed this fraction of the context window.
_CONTEXT_WARN_THRESHOLD = 0.75

# Maximum number of concepts sent to the LLM in a single Phase 2 call.
# Keeping batches small ensures the model stays focused and follows per-card
# instructions (e.g. the reverse-card rule) reliably across the whole deck.
CHUNK_SIZE = 20

# Shared signal written by _check_context_budget and read by the CLI animation
# loop to drive the context-fill progress bar.  Index 0 holds the most recent
# prompt-to-context-window ratio (0.0–1.0).  Plain list so writes are atomic
# under the GIL; safe for the single-background-thread CLI usage pattern.
_CTX_RATIO_SIGNAL: list[float] = [0.0]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (works for English prose)."""
    return max(1, len(text) // 4)


def _context_limit_for(provider_name: str) -> int:
    """Return the input context-token limit for a provider/model name string."""
    lower = provider_name.lower()
    for key, limit in _CONTEXT_LIMITS.items():
        if key == "_default":
            continue
        if key in lower:
            return limit
    return _CONTEXT_LIMITS["_default"]


def _check_context_budget(prompt: str, provider: "LLMProvider") -> None:
    """Update the shared context-ratio signal and warn when usage is high.

    Always writes to ``_CTX_RATIO_SIGNAL`` so the CLI animation bar reflects
    the actual context % regardless of whether the threshold is exceeded.
    Does not raise — generation proceeds either way.
    """
    estimated = _estimate_tokens(prompt)
    limit = _context_limit_for(provider.name)
    ratio = estimated / limit
    _CTX_RATIO_SIGNAL[0] = ratio
    if ratio >= _CONTEXT_WARN_THRESHOLD:
        pct = int(ratio * 100)
        print(
            f"[anki-gen] WARNING: prompt is ~{estimated:,} tokens"
            f" ({pct}% of {provider.name}'s ~{limit:,}-token context window).\n"
            "          Card quality may degrade. Consider splitting your notes"
            " into smaller files or using --confirm with fewer concepts.",
            file=sys.stderr,
        )


_IMAGE_INSTRUCTION = """\
When the source notes reference an image with a marker like
  [Image: "caption" (filename.ext)]  or  [Image: filename.ext]
YOU MUST include the image in a card field using an HTML img tag:
  <img src="filename.ext" alt="caption">
Use the exact filename (including extension) as the src value — Anki resolves
filenames directly from its media folder. Place each image in the card whose
concept is discussed nearest to that image marker in the source. Include ALL
images from the source notes — do not omit any image marker.\
"""

_MATHJAX_INSTRUCTION = """\
For any mathematical notation use Anki's MathJax format exclusively.
This output is embedded in JSON, so backslashes MUST be doubled:
  - Inline equations  (was $...$)  : \\\\( equation \\\\)
  - Display equations (was $$...$$): \\\\[ equation \\\\]
CRITICAL: use \\\\[...\\\\] whenever: (a) the source had $$...$$, or (b) the
equation is the primary content of a field or would naturally sit on its own line.
Never use \\\\(...\\\\) for display equations.
Do NOT use $...$ or $$...$$ notation.
Do NOT write \\( or \\[ with a single backslash — JSON requires \\\\( and \\\\[.\
"""

_CODE_INSTRUCTION = """\
Every piece of code must be explicitly marked — no exceptions:
  - Any identifier, keyword, operator, type, or syntax fragment that appears
    inside a prose sentence must be wrapped in backticks: `if let`, `Some(x)`,
    `None`, `match`, `Vec<T>`, `->`, etc.
  - Multi-line examples, or any snippet containing braces / operators / full
    expressions, must use a fenced block with a language tag:
    ```rust
    match foo { Some(v) => v, None => default }
    ```
  - Complex inline examples (e.g. a full `if let` expression) must also use a
    fenced block rather than being embedded in a sentence.
  - When a code block contains a diagram, ASCII art, or any indented structure,
    reproduce every character exactly — including ALL leading spaces on the VERY
    FIRST line. Do NOT trim or normalise the first line or any other line.
    Example: if the source has
    ```
           G - H    lanes_branch
          /
    A - B - C - D   main
    ```
    the card MUST preserve the 7 spaces before "G" on the first line, exactly
    as they appear. Stripping them breaks the visual alignment of the diagram.
Do NOT write raw HTML tags (<pre>, <code>, <span>, etc.) in any field.\
"""

_SRS_INSTRUCTION = """\
Optimise every card for spaced-repetition (SRS):
  - One atomic fact or relationship per card — no padding or over-explanation.
  - NEVER make information less concise than the source. Do not expand a terse
    bullet point into a prose paragraph.
  - If the source expresses a concept as a list of bullets, the card field may
    (and should) preserve that structure as an HTML list:
      <ul><li>first point</li><li>second point</li></ul>
  - Add context only when the source is genuinely ambiguous without it.
  - CRITICAL: preserve all source emphasis exactly — **bold** stays **bold**, *italic* stays *italic*. Never strip or flatten it.\
"""

_REVERSED_INSTRUCTION = """\
Reversed card rule: "front" is the forward question; "back" is the reverse
question whose answer is the content of "front". Both sides must be standalone
questions testing different facts — answering "front" and "back" must give
different answers.
Good: front = "What technique uses \\\\( \\\\theta = (X^TX)^{-1}X^Ty \\\\)?",
      back  = "What is the closed-form solution for linear regression?"
      front answer = normal equation / linear regression; back answer = the formula ✓
Bad:  front = "Toward what potential does the membrane return during repolarization?",
      back  = "What does the membrane potential return toward during the repolarization phase?"
      → both answers = EK / -90 mV — paraphrase, not a reversal ✗\
"""

# ---------------------------------------------------------------------------
# Code rendering  (plain pre/code — no syntax highlighting)
# ---------------------------------------------------------------------------

# Compiled once at module level for performance.
_FENCED_RE = re.compile(r"```([^\n]*)\n([\s\S]*?)```")

# Matches patterns that _render_inline_segment should handle specially rather
# than passing raw to html.escape():
#   1. Inline backtick code   →  wrapped in <code>…</code>
#   2. Markdown **bold**      →  <strong>…</strong>
#   3. Markdown *italic*      →  <em>…</em>
#      Note: __ and _ variants are intentionally excluded — `_` is too common
#      in code identifiers (e.g. __init__) to be a safe italic delimiter.
#      Bold must come before italic in the alternation so that ** is not
#      consumed by two successive * matches.
#   4. <img> tags             →  verbatim (hook for future image support)
#   5. Whitelisted HTML structural/formatting tags  →  verbatim, so that
#      LLM-generated <ul><li>…</li></ul> lists and <pre><code>…</code></pre>
#      blocks reach Anki as real HTML instead of escaped literal text.
#      The *content* between matched tag boundaries still flows through
#      html.escape(), so code with <, >, & is handled correctly.
#
# Named groups are used so that the dispatch in _render_inline_segment can
# identify which alternative matched without re-examining the raw string.
_COMBINED_INLINE_RE = re.compile(
    r"`(?P<backtick>[^`\n]+)`"
    r"|\*\*(?P<bold>[^*\n]+)\*\*"
    r"|\*(?P<em>[^*\n]+)\*"
    r"|<img\b[^>]*?/?>"
    r"|</?(?:ul|ol|li|br|p|pre|code|strong|em|b|i|h[1-6]|hr)\b[^>]*?/?>",
    re.IGNORECASE,
)


def _render_fenced(lang: str, code: str) -> str:
    """Wrap a fenced code block in <pre><code> with HTML-escaped content.

    The opening <code> and closing </code> tags are placed on their own lines
    so that the first line of content (including any leading spaces) is never
    run together with the tag.  This is important for ASCII-art diagrams where
    leading whitespace on the first line is visually significant.
    """
    escaped = html.escape(code.strip("\n"))
    tag = f' class="language-{html.escape(lang)}"' if lang else ""
    return f"<pre><code{tag}>\n{escaped}\n</code></pre>"


def _render_inline_segment(text: str) -> str:
    """
    Process a non-fenced text segment:
    1. Inline backtick code  → <code> (content is html.escape()-d)
    2. Markdown **bold**     → <strong> (content is html.escape()-d)
    3. Markdown *italic*     → <em> (content is html.escape()-d)
    4. <img …> tags          → preserved as-is (for image-support feature)
    5. Whitelisted HTML tags → preserved verbatim
    6. Text inside LLM-generated <code>…</code> → preserved verbatim
       (it is already HTML-escaped by the LLM; re-escaping would double-encode
       entities like &lt; → &amp;lt;)
    7. Everything else       → html.escape()

    A ``code_depth`` counter tracks nesting inside LLM-generated <code> blocks.
    When depth > 0, plain-text segments between regex matches are emitted
    verbatim instead of being passed through html.escape().
    """
    result: list[str] = []
    last = 0
    code_depth = 0
    for m in _COMBINED_INLINE_RE.finditer(text):
        before = text[last : m.start()]
        if before:
            result.append(before if code_depth > 0 else html.escape(before))
        raw = m.group(0)
        raw_lower = raw.lower()
        if m.group("backtick") is not None:
            # Markdown inline code: escape the content ourselves.
            result.append(f"<code>{html.escape(m.group('backtick'))}</code>")
        elif m.group("bold") is not None:
            # Markdown **bold**: escape the content, wrap in <strong>.
            result.append(f"<strong>{html.escape(m.group('bold'))}</strong>")
        elif m.group("em") is not None:
            # Markdown *italic*: escape the content, wrap in <em>.
            result.append(f"<em>{html.escape(m.group('em'))}</em>")
        elif raw_lower.startswith("<code"):
            # Opening <code> or <code class="…"> from LLM-generated HTML.
            code_depth += 1
            result.append(raw)
        elif raw_lower.startswith("</code"):
            # Closing </code> from LLM-generated HTML.
            code_depth = max(0, code_depth - 1)
            result.append(raw)
        else:
            result.append(
                raw
            )  # <img>, <ul>, <li>, <strong>, <em>, etc. — preserve verbatim
        last = m.end()
    tail = text[last:]
    if tail:
        result.append(tail if code_depth > 0 else html.escape(tail))
    return "".join(result)


def _render_code(text: str) -> str:
    """
    Multi-pass HTML renderer applied to every LLM card field.

    Pass order:
    1. Fenced code blocks (```lang\\n…```) → <pre><code> with html.escape
    2. Inline backtick code (`…`)          → <code> span
    3. <img …> tags                        → preserved verbatim
    4. Remaining plain text                → html.escape()

    MathJax delimiters (\\(…\\) / \\[…\\]) contain no HTML-special chars
    and pass through html.escape unchanged.
    """
    parts: list[str] = []
    last = 0
    for m in _FENCED_RE.finditer(text):
        before = text[last : m.start()]
        if before:
            parts.append(_render_inline_segment(before))
        lang = m.group(1).strip()
        code = m.group(2)
        parts.append(_render_fenced(lang, code))
        last = m.end()
    remaining = text[last:]
    if remaining:
        parts.append(_render_inline_segment(remaining))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Phase 1 prompt: concept extraction
# ---------------------------------------------------------------------------

_CONCEPT_PROMPT_TEMPLATE = """\
You are an expert educator analysing study notes.

## Task
Read the notes below and identify up to {max_concepts} distinct concepts, facts,
or relationships that would make strong Anki flashcards.

Return ONLY a JSON array of short concept strings (one idea per string, max ~10 words each).
No prose before or after the array.

Example output:
["TCP three-way handshake", "CAP theorem — consistency vs availability trade-off", ...]

## Notes (title: {title})

{content}
"""


def extract_concepts(
    doc: ParsedDocument,
    provider: "LLMProvider",
    max_cards: int | None = None,
) -> list[str]:
    """
    Phase 1 — ask the LLM for a concept list it would turn into cards.
    Returns a list of short concept strings capped at the same ceiling
    used for card generation so the two phases stay in sync.
    """
    effective_max = (
        max_cards if max_cards is not None else _derive_max_cards(doc.concept_count)
    )
    prompt = _CONCEPT_PROMPT_TEMPLATE.format(
        max_concepts=effective_max,
        title=doc.title,
        content=doc.plain_text,
    )
    _check_context_budget(prompt, provider)
    raw = provider.complete(prompt)
    clean = _extract_json_array(raw)
    try:
        items = json.loads(_repair_json(clean))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON for concept list: {exc}\n\nRaw:\n{raw}"
        ) from exc
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON array, got {type(items).__name__}")
    concepts = [str(c).strip() for c in items if str(c).strip()]
    return concepts[:effective_max]


# ---------------------------------------------------------------------------
# Phase 2 prompt: cards from confirmed concepts
# ---------------------------------------------------------------------------

_CARDS_FROM_CONCEPTS_TEMPLATE = """\
You are an expert educator creating Anki flashcards from study notes.

## Task
{task_count_instruction}

Confirmed concepts:
{concept_list}

For each concept choose the most appropriate card type:
- Use "definition" when the concept is a term, keyword, or named idea that has
  a specific meaning worth memorising directly.
- Use "basic" for processes, relationships, reasons, comparisons, or anything
  better expressed as a question and answer.
- Use "basic_reversed" when a concept has two genuinely distinct testable
  directions (e.g. name↔formula, cause↔effect, term↔mechanism). A concept
  marked [REVERSED] should use "basic_reversed" — but only when a valid
  reversal exists. If the concept is a bare fact, isolated property, or
  inherently one-directional (e.g. "resting potential is -70 mV"), use "basic"
  instead. A forced reversal that paraphrases the front is worse than no reversal.
  {reversed_instruction}

- {mathjax_instruction}
- {code_instruction}
- {srs_instruction}
- {image_instruction}

Rules:
- Exactly one card per concept — do NOT generate multiple cards for the same concept.
- Cover every concept in the list — do not skip any.
- Each card must be self-contained (no pronouns requiring outside context).
- CRITICAL: every piece of information on every card must come directly and
  exclusively from the source notes below. Do NOT add facts, definitions,
  explanations, or context that are not explicitly stated in the source.
- CRITICAL: if a concept description contains explicit layout instructions
  (e.g. "put the diagram on the front", "include X on the front of the card"),
  follow them exactly. The "front" field may contain a diagram, code block, or
  other context material before or after the question — it is not restricted to
  a single question sentence.
- {mathjax_instruction}
- {code_instruction}
- {srs_instruction}
- {image_instruction}
{notes_section}## Output format
Return ONLY a JSON array{output_count_instruction}. No prose before or after.
Each element is one of:

  {{"type": "basic", "front": "<question>", "back": "<answer>"}}
  {{"type": "basic_reversed", "front": "<question — forward direction>", "back": "<question — reverse direction>"}}
  {{"type": "definition", "term": "<term>", "definition": "<definition>"}}

## Source notes (title: {title})

{content}
"""


# ---------------------------------------------------------------------------
# Single-call prompt (no --confirm)
# ---------------------------------------------------------------------------

_CARDS_DIRECT_TEMPLATE = """\
You are an expert educator creating Anki flashcards from study notes.

## Task
Analyse the notes below and generate up to {max_cards} high-quality flashcards.
Produce a mix of:
- **basic** cards: a question on the front, a concise answer on the back.
- **basic_reversed** cards: use when a concept has two genuinely distinct
  testable directions (e.g. name↔formula, cause↔effect, term↔mechanism).
  Only use when a valid reversal exists — if the concept is a bare fact or
  inherently one-directional, use "basic" instead.
  {reversed_instruction}
- **definition** cards: a term on the front, its definition on the back.

Prioritise the most important concepts, relationships, and facts.
Do NOT generate cards for trivial or obvious statements.
Each card must be self-contained — no pronouns that require context to resolve.
CRITICAL: every piece of information on every card must come directly and
exclusively from the source notes below. Do NOT add facts, definitions,
explanations, or context that are not explicitly stated in the source.
If the source notes are empty or contain no substantive content, return an
empty JSON array: [].
- {mathjax_instruction}
- {code_instruction}
- {srs_instruction}
- {image_instruction}

## Output format
Return ONLY a JSON array. No prose before or after. Each element is one of:

  {{"type": "basic", "front": "<question>", "back": "<answer>"}}
  {{"type": "basic_reversed", "front": "<question — forward direction>", "back": "<question — reverse direction>"}}
  {{"type": "definition", "term": "<term>", "definition": "<definition>"}}

## Notes (title: {title})

{content}
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_json_array(raw: str) -> str:
    """Strip markdown fences then isolate the first JSON array span.

    Only matches fenced blocks tagged as ``json`` (or untagged — no language
    word) by requiring the fence to be followed immediately by optional
    whitespace then a newline.  This prevents accidentally matching code
    blocks whose language tag is something like ``rust`` or ``python``.
    """
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]+)\n\s*```", raw)
    if fenced:
        return fenced.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _repair_json_backslashes(text: str) -> str:
    """
    Fix unescaped backslashes that the LLM sometimes emits inside JSON strings.

    The LLM occasionally outputs MathJax delimiters like \\( \\) \\[ \\] with a
    single backslash even though JSON requires them to be doubled (\\\\( etc.).

    Strategy: use a negative lookbehind so that a backslash already preceded by
    another backslash (i.e. part of a valid \\\\  escape sequence) is left alone,
    and only lone backslashes before non-JSON-escape characters are doubled.

    Valid JSON escape sequences: \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
    """
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)


def _repair_json_literal_newlines(text: str) -> str:
    """
    Replace literal newline/carriage-return characters that appear inside JSON
    string values with their JSON escape sequences (\\n / \\r).

    The LLM sometimes emits fenced code blocks inside JSON string values using
    actual newline characters rather than the required \\n escape sequences,
    producing "Unterminated string" parse errors.

    Strategy: walk through the text character-by-character tracking whether we
    are inside a JSON string (handling \\\\ and \\" escapes so we don't mistake
    an escaped quote for a string boundary).  Only replace newlines found
    inside a string context.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                # Consume the escape sequence verbatim (two chars)
                result.append(ch)
                i += 1
                if i < len(text):
                    result.append(text[i])
                    i += 1
            elif ch == '"':
                in_string = False
                result.append(ch)
                i += 1
            elif ch == "\n":
                result.append("\\n")
                i += 1
            elif ch == "\r":
                result.append("\\r")
                i += 1
            else:
                result.append(ch)
                i += 1
        else:
            if ch == '"':
                in_string = True
            result.append(ch)
            i += 1
    return "".join(result)


def _repair_json(text: str) -> str:
    """Apply all JSON repair passes in order."""
    text = _repair_json_literal_newlines(text)
    text = _repair_json_backslashes(text)
    return text


def _parse_cards(raw_json: str) -> list[Card]:
    """
    Deserialise a JSON array of card dicts into typed Card objects.
    _render_code() is applied to every field so the caller receives
    HTML-ready content with syntax-highlighted code blocks.
    """
    repaired = _repair_json(raw_json)
    try:
        items = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\n\nRaw output:\n{raw_json}"
        ) from exc

    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array, got {type(items).__name__}")

    cards: list[Card] = []
    for item in items:
        card_type = item.get("type")
        if card_type == "basic":
            cards.append(
                BasicCard(
                    front=_render_code(item["front"]),
                    back=_render_code(item["back"]),
                )
            )
        elif card_type == "basic_reversed":
            cards.append(
                BasicCard(
                    front=_render_code(item["front"]),
                    back=_render_code(item["back"]),
                    reversed=True,
                )
            )
        elif card_type == "definition":
            cards.append(
                DefinitionCard(
                    term=_render_code(item["term"]),
                    definition=_render_code(item["definition"]),
                )
            )
        # Silently skip unrecognised types

    return cards


def _apply_mathjax(cards: list[Card]) -> list[Card]:
    """
    Run the LaTeX → MathJax conversion over every text field on every card.
    This is the non-negotiable enforcement layer — runs regardless of what
    the LLM produced.

    Safe to run after _render_code() because MathJax chars (\\, (, )) are
    not HTML-special and pass through html.escape unchanged.
    """
    result: list[Card] = []
    for card in cards:
        if isinstance(card, BasicCard):
            result.append(
                BasicCard(
                    front=promote_sole_inline_to_display(
                        convert_latex_to_mathjax(card.front)
                    ),
                    back=promote_sole_inline_to_display(
                        convert_latex_to_mathjax(card.back)
                    ),
                    reversed=card.reversed,
                    tags=card.tags,
                )
            )
        elif isinstance(card, DefinitionCard):
            result.append(
                DefinitionCard(
                    term=promote_sole_inline_to_display(
                        convert_latex_to_mathjax(card.term)
                    ),
                    definition=promote_sole_inline_to_display(
                        convert_latex_to_mathjax(card.definition)
                    ),
                    tags=card.tags,
                )
            )
        else:
            result.append(card)
    return result


# ---------------------------------------------------------------------------
# Image injection post-processor
# ---------------------------------------------------------------------------

_IMG_SRC_RE = re.compile(r'src=["\']([^"\']+)["\']')


def _append_image_to_card(card: Card, img_tag: str) -> Card:
    wrapped = f"<br>{img_tag}<br>"
    if isinstance(card, BasicCard):
        return BasicCard(
            front=card.front,
            back=card.back + wrapped,
            reversed=card.reversed,
            tags=card.tags,
        )
    if isinstance(card, DefinitionCard):
        return DefinitionCard(
            term=card.term,
            definition=card.definition + wrapped,
            tags=card.tags,
        )
    return card


def _with_card_fields(card: Card, fields: list[str]) -> Card:
    if isinstance(card, BasicCard):
        return BasicCard(
            front=fields[0],
            back=fields[1],
            reversed=card.reversed,
            tags=card.tags,
        )
    if isinstance(card, DefinitionCard):
        return DefinitionCard(term=fields[0], definition=fields[1], tags=card.tags)
    return card


def _remove_image_from_cards(cards: list[Card], filename: str) -> list[Card]:
    pattern = re.compile(
        r'(?:<br>)?\s*<img\b(?=[^>]*\bsrc=["\']%s["\'])[^>]*?/?>\s*(?:<br>)?'
        % re.escape(filename),
        re.IGNORECASE,
    )
    result: list[Card] = []
    for card in cards:
        if isinstance(card, BasicCard):
            fields = [card.front, card.back]
        elif isinstance(card, DefinitionCard):
            fields = [card.term, card.definition]
        else:
            result.append(card)
            continue
        cleaned = [pattern.sub("", field) for field in fields]
        result.append(_with_card_fields(card, cleaned))
    return result


def _build_img_tag(ref: ImageRef) -> str:
    alt = html.escape(ref.alt_text) if ref.alt_text else ""
    return (
        f'<img src="{ref.filename}" alt="{alt}">'
        if alt
        else f'<img src="{ref.filename}">'
    )


def _images_already_placed(cards: Sequence[Card]) -> set[str]:
    """Return the set of image filenames already referenced in any card field."""
    placed: set[str] = set()
    for card in cards:
        if isinstance(card, BasicCard):
            fields = [card.front, card.back]
        elif isinstance(card, DefinitionCard):
            fields = [card.term, card.definition]
        else:
            continue
        for f in fields:
            for m in _IMG_SRC_RE.finditer(f):
                placed.add(m.group(1))
    return placed


def _best_card_for_image(ref: ImageRef, cards: Sequence[Card], plain_text: str) -> int:
    """Return the index of the card most contextually relevant to *ref*.

    Score = (window overlap) + 2 × (filename overlap).

    The filename bonus is weighted 2× because it is a direct authorial signal
    about the image's subject (e.g. "derivative-log-fun.webp" strongly implies
    the image belongs with the "Derivative of logistic function" card even when
    the surrounding text happens to discuss other topics).

    Falls back to index 0 if the filename is not found in *plain_text*.
    """
    _STOPWORDS = {"a", "an", "the", "of", "in", "is", "and", "or", "to", "for", "fun"}

    marker_pos = plain_text.find(ref.filename)
    if marker_pos == -1:
        return 0

    start = max(0, marker_pos - 300)
    end = min(len(plain_text), marker_pos + 300)
    window = plain_text[start:end].lower()
    window_words = set(re.findall(r"\w+", window))

    # Words encoded in the filename (split on hyphens, underscores, dots)
    stem = Path(ref.filename).stem  # drop extension
    filename_words = set(re.findall(r"\w+", stem.lower())) - _STOPWORDS

    best_idx = 0
    best_score = -1
    for i, card in enumerate(cards):
        if isinstance(card, BasicCard):
            raw = card.front + " " + card.back
        elif isinstance(card, DefinitionCard):
            raw = card.term + " " + card.definition
        else:
            continue
        # Strip HTML tags before tokenising
        text = re.sub(r"<[^>]+>", " ", raw).lower()
        card_words = set(re.findall(r"\w+", text))
        score = len(card_words & window_words) + 4 * len(card_words & filename_words)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _image_near_any_concept(
    ref: ImageRef, plain_text: str, confirmed_concepts: list[str]
) -> bool:
    """Return True if the image marker's context window overlaps with any confirmed concept.

    Tokenises the 600-char window around the image marker and checks for at
    least one shared word with any confirmed concept name (excluding stopwords).
    """
    _STOPWORDS = {"a", "an", "the", "of", "in", "is", "are", "and", "or", "to", "for"}

    marker_pos = plain_text.find(ref.filename)
    if marker_pos == -1:
        # No marker found — don't block injection; let caller decide.
        return True

    start = max(0, marker_pos - 300)
    end = min(len(plain_text), marker_pos + 300)
    window_words = set(re.findall(r"\w+", plain_text[start:end].lower())) - _STOPWORDS

    for concept in confirmed_concepts:
        concept_words = set(re.findall(r"\w+", concept.lower())) - _STOPWORDS
        if concept_words & window_words:
            return True
    return False


def inject_missed_images(
    cards: list[Card] | Sequence[Card],
    doc: ParsedDocument,
    confirmed_concepts: list[str] | None = None,
    concept_order: list[str] | None = None,
    manual_image_assignments: dict[str, list[str]] | None = None,
    excluded_images: set[str] | None = None,
) -> list[Card]:
    """Ensure every image in *doc* appears in at least one generated card.

    For each :class:`~anki_gen.parser.ImageRef` whose filename is not already
    referenced by a ``src`` attribute in any card field, finds the most
    contextually relevant card via text proximity in ``doc.plain_text`` and
    appends ``<img src="filename" alt="alt_text">`` to its ``back`` /
    ``definition`` field.

    When *confirmed_concepts* is provided (``--confirm`` path), an image is
    only injected if its context window in ``doc.plain_text`` shares at least
    one word with a confirmed concept name.  Images that sit exclusively near
    rejected concepts are left out — respecting the user's deletion choice.

    This is a deterministic safety net: it fires only for images the LLM
    omitted despite the prompt instruction.
    """
    if not doc.images or not cards:
        return list(cards)

    refs_by_name = {ref.filename: ref for ref in doc.images}
    result = list(cards)
    excluded = set(excluded_images or ())

    if excluded:
        for filename in excluded:
            result = _remove_image_from_cards(result, filename)

    if manual_image_assignments:
        concept_index = {concept: i for i, concept in enumerate(concept_order or [])}
        for concept, image_names in manual_image_assignments.items():
            idx = concept_index.get(concept)
            if idx is None or idx >= len(result):
                continue
            for image_name in image_names:
                ref = refs_by_name.get(image_name)
                if ref is None:
                    continue
                result = _remove_image_from_cards(result, image_name)
                result[idx] = _append_image_to_card(result[idx], _build_img_tag(ref))

    placed = _images_already_placed(result)
    missed = [
        ref
        for ref in doc.images
        if ref.filename not in placed and ref.filename not in excluded
    ]
    if not missed:
        return result

    # When the caller supplies confirmed concepts, skip images whose context
    # window doesn't overlap with any of them (i.e. the image is near a
    # rejected concept and the user intentionally excluded it).
    if confirmed_concepts is not None:
        missed = [
            ref
            for ref in missed
            if _image_near_any_concept(ref, doc.plain_text, confirmed_concepts)
        ]
    if not missed:
        return result

    for ref in missed:
        idx = _best_card_for_image(ref, result, doc.plain_text)
        result[idx] = _append_image_to_card(result[idx], _build_img_tag(ref))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_notes_section(notes: list[str] | None) -> str:
    """
    Format the user's free-text instructions as a prompt section.
    Returns an empty string when there are no notes so the template
    placeholder disappears without leaving extra whitespace.
    """
    if not notes:
        return ""
    items = "\n".join(f"- {n}" for n in notes)
    return f"## Additional instructions\n{items}\n\n"


def _format_concept_list(
    concepts: list[str], reversed_concepts: set[str] | None
) -> str:
    """
    Format the confirmed concept list for the Phase 2 prompt.

    Concepts that the user explicitly marked for reversal are annotated with
    [REVERSED] so the LLM knows to use the basic_reversed card type for them.
    """
    lines: list[str] = []
    for c in concepts:
        if reversed_concepts and c in reversed_concepts:
            lines.append(f"- {c}  [REVERSED]")
        else:
            lines.append(f"- {c}")
    return "\n".join(lines)


def chunk_concepts(
    concepts: list[str],
    reversed_concepts: set[str] | None,
    chunk_size: int,
) -> list[tuple[list[str], set[str] | None]]:
    """Split *concepts* into sequential batches of *chunk_size*.

    Each batch is paired with the subset of *reversed_concepts* that belongs
    to it, so every chunk is self-contained and the caller never needs to
    filter the reversed set itself.

    Returns a list of ``(concept_chunk, reversed_chunk)`` tuples preserving
    the original concept order.  When *concepts* is empty the result is empty.
    """
    if not concepts:
        return []
    chunks: list[tuple[list[str], set[str] | None]] = []
    for i in range(0, len(concepts), chunk_size):
        batch = concepts[i : i + chunk_size]
        if reversed_concepts:
            rev_batch: set[str] | None = reversed_concepts & set(batch)
            rev_batch = rev_batch if rev_batch else None
        else:
            rev_batch = None
        chunks.append((batch, rev_batch))
    return chunks


def generate_cards_from_concepts(
    doc: ParsedDocument,
    concepts: list[str],
    provider: "LLMProvider",
    notes: list[str] | None = None,
    reversed_concepts: set[str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    images_enabled: bool = True,
) -> list[Card]:
    """Phase 2 — generate exactly one card per confirmed concept.

    Concepts are processed in sequential batches of *chunk_size* (default
    ``CHUNK_SIZE``).  Keeping batches small prevents the model from losing
    focus on per-card rules (e.g. the reverse-card constraint) when the deck
    is large.  Each batch receives the *full* source document so the LLM
    always has complete context, but only needs to juggle a small number of
    concepts at once.

    When *concepts* fits within a single chunk the call is functionally
    identical to the previous single-shot behaviour — no extra API calls.

    *notes* is forwarded to every chunk call because notes are typically
    style instructions that apply globally (e.g. "make everything
    bidirectional").

    *reversed_concepts* is filtered per chunk so each call only sees the
    reversed markers that are relevant to its concept subset.
    """
    all_cards: list[Card] = []

    for batch, batch_reversed in chunk_concepts(
        concepts, reversed_concepts, chunk_size
    ):
        all_cards.extend(
            generate_cards_for_chunk(
                doc,
                batch,
                provider,
                notes,
                batch_reversed,
                images_enabled=images_enabled,
            )
        )

    return all_cards


def generate_cards_for_chunk(
    doc: ParsedDocument,
    concepts: list[str],
    provider: "LLMProvider",
    notes: list[str] | None,
    reversed_concepts: set[str] | None,
    images_enabled: bool = True,
) -> list[Card]:
    """Single-chunk Phase 2 call — internal workhorse for generate_cards_from_concepts."""
    concept_list = _format_concept_list(concepts, reversed_concepts)
    num_concepts = len(concepts)
    has_notes = bool(notes)

    if has_notes:
        task_count_instruction = (
            f"Generate ONE flashcard for each of the {num_concepts} confirmed"
            f" concepts below, plus any additional cards requested in the"
            f" 'Additional instructions' section."
        )
        output_count_instruction = f" with at least {num_concepts} elements"
    else:
        task_count_instruction = (
            f"Generate exactly ONE flashcard for each of the {num_concepts}"
            f" confirmed concepts below.\n"
            f"The output array must contain exactly {num_concepts} cards"
            f" — one per concept, no more."
        )
        output_count_instruction = f" of exactly {num_concepts} elements"

    prompt = _CARDS_FROM_CONCEPTS_TEMPLATE.format(
        concept_list=concept_list,
        task_count_instruction=task_count_instruction,
        output_count_instruction=output_count_instruction,
        reversed_instruction=_REVERSED_INSTRUCTION,
        mathjax_instruction=_MATHJAX_INSTRUCTION,
        code_instruction=_CODE_INSTRUCTION,
        srs_instruction=_SRS_INSTRUCTION,
        image_instruction=_IMAGE_INSTRUCTION
        if images_enabled
        else "Do not include images.",
        notes_section=_format_notes_section(notes),
        title=doc.title,
        content=doc.plain_text,
    )

    _check_context_budget(prompt, provider)
    raw = provider.complete(prompt)
    cards = _parse_cards(_extract_json_array(raw))
    if not has_notes:
        cards = cards[:num_concepts]
    return _apply_mathjax(cards)


def generate_cards(
    doc: ParsedDocument,
    provider: "LLMProvider",
    max_cards: int | None = None,
    images_enabled: bool = True,
) -> list[Card]:
    """
    Single-call path (no --confirm). Generates cards directly from the document.

    MathJax enforcement is applied as a post-processing pass on the output.
    """
    effective_max = (
        max_cards if max_cards is not None else _derive_max_cards(doc.concept_count)
    )

    prompt = _CARDS_DIRECT_TEMPLATE.format(
        max_cards=effective_max,
        reversed_instruction=_REVERSED_INSTRUCTION,
        mathjax_instruction=_MATHJAX_INSTRUCTION,
        code_instruction=_CODE_INSTRUCTION,
        srs_instruction=_SRS_INSTRUCTION,
        image_instruction=_IMAGE_INSTRUCTION
        if images_enabled
        else "Do not include images.",
        title=doc.title,
        content=doc.plain_text,
    )

    _check_context_budget(prompt, provider)
    raw = provider.complete(prompt)
    cards = _parse_cards(_extract_json_array(raw))
    cards = _apply_mathjax(cards)
    return cards[:effective_max]
