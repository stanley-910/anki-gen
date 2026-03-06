from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Anki MathJax rules (from Anki documentation):
#   Inline equations  : \( ... \)
#   Display equations : \[ ... \]
#
# Markdown convention we convert FROM:
#   Inline equations  : $...$
#   Display equations : $$...$$
#
# ORDER MATTERS: display ($$) must be processed before inline ($) to avoid
# the inline pattern consuming the outer dollars of a display block.
# ---------------------------------------------------------------------------

# Matches $$...$$ including across newlines (display equations)
_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

# Matches $...$ but NOT $$...$$ — uses a negative look-around to exclude
# strings that start/end with a second dollar sign.
_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", re.DOTALL)

# Matches a field whose entire content is a single \(...\) block (with optional
# surrounding whitespace).  These should render as display equations.
_SOLE_INLINE_RE = re.compile(r"^\s*\\\((.*?)\\\)\s*$", re.DOTALL)


def convert_latex_to_mathjax(text: str) -> str:
    """
    Convert Markdown LaTeX delimiters to Anki's MathJax delimiters.

    Transformations applied (display before inline, non-negotiable):
      $$...$$ → \\[...\\]
      $...$   → \\(...\\)

    Idempotent on text that already uses \\(...\\) / \\[...\\] notation.
    """
    # Step 1: display equations
    text = _DISPLAY_RE.sub(lambda m: r"\[" + m.group(1) + r"\]", text)
    # Step 2: inline equations
    text = _INLINE_RE.sub(lambda m: r"\(" + m.group(1) + r"\)", text)
    return text


def promote_sole_inline_to_display(text: str) -> str:
    """Promote a field whose entire content is a single \\(...\\) to \\[...\\].

    When the model generates a card whose back (or any field) is nothing but
    one equation, it should render as a display block.  The LLM frequently
    emits \\(...\\) even in this case; this deterministic pass corrects it.
    """
    m = _SOLE_INLINE_RE.match(text)
    if m:
        return r"\[" + m.group(1) + r"\]"
    return text


def apply_mathjax_to_card_fields(**fields: str) -> dict[str, str]:
    """
    Convenience wrapper: apply convert_latex_to_mathjax to every field value
    in a dict and return the transformed dict.

    Usage:
        new_fields = apply_mathjax_to_card_fields(front=card.front, back=card.back)
    """
    return {key: convert_latex_to_mathjax(value) for key, value in fields.items()}
