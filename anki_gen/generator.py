from __future__ import annotations

import html
import json
import re
from typing import TYPE_CHECKING

from anki_gen.latex import convert_latex_to_mathjax
from anki_gen.models import BasicCard, Card, DefinitionCard
from anki_gen.parser import ParsedDocument, _derive_max_cards

if TYPE_CHECKING:
    from anki_gen.llm.base import LLMProvider

# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_MATHJAX_INSTRUCTION = """\
For any mathematical notation use Anki's MathJax format exclusively.
This output is embedded in JSON, so backslashes MUST be doubled:
  - Inline equations  (was $...$)  : \\\\( equation \\\\)
  - Display equations (was $$...$$): \\\\[ equation \\\\]
CRITICAL: equations that appeared as $$...$$ in the source MUST use \\\\[...\\\\],
never \\\\(...\\\\). Using inline delimiters for display equations is WRONG.
CRITICAL: if an equation is the primary content of a card field, or would
naturally sit on its own line, use \\\\[...\\\\] even if the source did not
explicitly mark it as a display equation.
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
  - Add context only when the source is genuinely ambiguous without it.\
"""

# ---------------------------------------------------------------------------
# Code rendering  (plain pre/code — no syntax highlighting)
# ---------------------------------------------------------------------------

# Compiled once at module level for performance.
_FENCED_RE = re.compile(r"```([^\n]*)\n([\s\S]*?)```")

# Matches patterns that _render_inline_segment should preserve rather than
# html-escape:
#   1. Inline backtick code  →  wrapped in <code>…</code>
#   2. <img> tags            →  verbatim (hook for future image support)
#   3. Whitelisted HTML structural/formatting tags  →  verbatim, so that
#      LLM-generated <ul><li>…</li></ul> lists and <pre><code>…</code></pre>
#      blocks reach Anki as real HTML instead of escaped literal text.
#      The *content* between matched tag boundaries still flows through
#      html.escape(), so code with <, >, & is handled correctly.
_COMBINED_INLINE_RE = re.compile(
    r"`([^`\n]+)`"
    r"|<img\b[^>]*?/?>"
    r"|</?(?:ul|ol|li|br|p|pre|code|strong|em|b|i|h[1-6]|hr)\b[^>]*?/?>",
    re.IGNORECASE,
)


def _render_fenced(lang: str, code: str) -> str:
    """Wrap a fenced code block in <pre><code> with HTML-escaped content."""
    escaped = html.escape(code.rstrip("\n"))
    tag = f' class="language-{html.escape(lang)}"' if lang else ""
    return f"<pre><code{tag}>{escaped}</code></pre>"


def _render_inline_segment(text: str) -> str:
    """
    Process a non-fenced text segment:
    1. Inline backtick code  → <code>
    2. <img …> tags          → preserved as-is (for image-support feature)
    3. Everything else       → html.escape()
    """
    result: list[str] = []
    last = 0
    for m in _COMBINED_INLINE_RE.finditer(text):
        result.append(html.escape(text[last : m.start()]))
        raw = m.group(0)
        if raw.startswith("`"):
            result.append(f"<code>{html.escape(m.group(1))}</code>")
        else:
            result.append(raw)  # preserve <img> verbatim
        last = m.end()
    result.append(html.escape(text[last:]))
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
Generate exactly ONE flashcard for each of the {num_concepts} confirmed concepts below.
The output array must contain exactly {num_concepts} cards — one per concept, no more.

Confirmed concepts:
{concept_list}

For each concept choose the most appropriate card type:
- Use "definition" when the concept is a term, keyword, or named idea that has
  a specific meaning worth memorising directly.
- Use "basic" for processes, relationships, reasons, comparisons, or anything
  better expressed as a question and answer.
- Use "basic_reversed" when the additional instructions ask for bidirectional
  testing, or when the relationship is inherently symmetric (e.g. a named
  formula, a term with a symbol, a concept with a specific expression).
  CRITICAL: both "front" and "back" must be phrased as questions or active
  prompts — NOT as a statement and its answer. Anki will show each field
  alone as the question in separate review sessions, so each side must
  hide what the other reveals.
  Good: front = "What is the closed-form solution for linear regression?",
        back  = "What technique uses the formula \\\\( \\\\theta = (X^TX)^{{-1}}X^Ty \\\\)?"
  Bad:  front = "The closed-form solution for linear regression is...",
        back  = "\\\\( \\\\theta = (X^TX)^{{-1}}X^Ty \\\\)"

Rules:
- Exactly one card per concept — do NOT generate multiple cards for the same concept.
- Cover every concept in the list — do not skip any.
- Concepts marked [REVERSED] MUST use the "basic_reversed" type — no other type is allowed for them.
- Each card must be self-contained (no pronouns requiring outside context).
- {mathjax_instruction}
- {code_instruction}
- {srs_instruction}
{notes_section}## Output format
Return ONLY a JSON array of exactly {num_concepts} elements. No prose before or after.
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
- **basic_reversed** cards: use when the relationship is inherently symmetric
  (e.g. a named formula, a term with a symbol). CRITICAL: both "front" and
  "back" must be phrased as questions or active prompts — NOT a statement and
  its answer. Anki shows each field alone as the question in separate sessions,
  so each side must hide what the other reveals.
  Good: front = "What is the closed-form solution for linear regression?",
        back  = "What technique uses θ = (XᵀX)⁻¹Xᵀy?"
  Bad:  front = "The closed-form solution for linear regression is...",
        back  = "θ = (XᵀX)⁻¹Xᵀy"
- **definition** cards: a term on the front, its definition on the back.

Prioritise the most important concepts, relationships, and facts.
Do NOT generate cards for trivial or obvious statements.
Each card must be self-contained — no pronouns that require context to resolve.
- {mathjax_instruction}
- {code_instruction}
- {srs_instruction}

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
                    front=convert_latex_to_mathjax(card.front),
                    back=convert_latex_to_mathjax(card.back),
                    reversed=card.reversed,
                    tags=card.tags,
                )
            )
        elif isinstance(card, DefinitionCard):
            result.append(
                DefinitionCard(
                    term=convert_latex_to_mathjax(card.term),
                    definition=convert_latex_to_mathjax(card.definition),
                    tags=card.tags,
                )
            )
        else:
            result.append(card)
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


def generate_cards_from_concepts(
    doc: ParsedDocument,
    concepts: list[str],
    provider: "LLMProvider",
    notes: list[str] | None = None,
    reversed_concepts: set[str] | None = None,
) -> list[Card]:
    """
    Phase 2 — generate exactly one card per confirmed concept.

    The prompt instructs the LLM to produce one card per concept and to choose
    the best card type (basic vs definition) for each. The output is capped at
    len(concepts) as a hard safety net, then MathJax notation is enforced.

    *notes* is an optional list of free-text instructions that the user entered
    in the confirm TUI (via the 'n' key).  When provided they are injected into
    the prompt as an "Additional instructions" section so the LLM can apply
    them while building cards (e.g. "make cards testable forwards and backwards"
    or "focus only on the equations").

    *reversed_concepts* is an optional set of concept strings that the user
    explicitly marked for reversal in the TUI (via the 'r' / 'R' keys). These
    are annotated in the prompt so the LLM uses the basic_reversed card type.
    """
    concept_list = _format_concept_list(concepts, reversed_concepts)
    num_concepts = len(concepts)

    prompt = _CARDS_FROM_CONCEPTS_TEMPLATE.format(
        concept_list=concept_list,
        num_concepts=num_concepts,
        mathjax_instruction=_MATHJAX_INSTRUCTION,
        code_instruction=_CODE_INSTRUCTION,
        srs_instruction=_SRS_INSTRUCTION,
        notes_section=_format_notes_section(notes),
        title=doc.title,
        content=doc.plain_text,
    )

    raw = provider.complete(prompt)
    cards = _parse_cards(_extract_json_array(raw))
    # Hard cap: never return more cards than there are concepts
    cards = cards[:num_concepts]
    return _apply_mathjax(cards)


def generate_cards(
    doc: ParsedDocument,
    provider: "LLMProvider",
    max_cards: int | None = None,
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
        mathjax_instruction=_MATHJAX_INSTRUCTION,
        code_instruction=_CODE_INSTRUCTION,
        srs_instruction=_SRS_INSTRUCTION,
        title=doc.title,
        content=doc.plain_text,
    )

    raw = provider.complete(prompt)
    cards = _parse_cards(_extract_json_array(raw))
    cards = _apply_mathjax(cards)
    return cards[:effective_max]
