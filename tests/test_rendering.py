"""Tests for the HTML rendering pipeline in anki_gen.generator.

Covers _render_fenced, _render_inline_segment, and _render_code.
"""

import pytest
from anki_gen.generator import _render_code, _render_fenced, _render_inline_segment


# ---------------------------------------------------------------------------
# _render_fenced
# ---------------------------------------------------------------------------


class TestRenderFenced:
    def test_basic_no_lang(self):
        out = _render_fenced("", "x = 1")
        assert out == "<pre><code>\nx = 1\n</code></pre>"

    def test_lang_tag(self):
        out = _render_fenced("python", "x = 1")
        assert 'class="language-python"' in out
        assert out.startswith("<pre><code")

    def test_html_chars_escaped(self):
        out = _render_fenced("", "a < b && c > d")
        assert "&lt;" in out
        assert "&amp;" in out
        assert "&gt;" in out
        assert "<" not in out.replace("<pre>", "").replace("<code>", "").replace(
            "</code>", ""
        ).replace("</pre>", "")

    def test_leading_spaces_preserved(self):
        # ASCII-art diagrams depend on leading whitespace on the first line.
        code = "       G - H    branch\n      /\nA - B - C   main"
        out = _render_fenced("", code)
        # First line must start with 7 spaces (not trimmed).
        assert "\n       G - H" in out

    def test_newline_after_code_tag(self):
        # Content must not start on the same line as <code>.
        out = _render_fenced("", "x = 1")
        assert "<code>\n" in out
        assert "\n</code>" in out

    def test_strips_surrounding_blank_lines_only(self):
        # strip("\n") removes leading/trailing newlines but not spaces.
        out = _render_fenced("", "\n  indented\n")
        assert "\n  indented\n" in out

    def test_lang_html_escaped(self):
        out = _render_fenced("<script>", "x")
        assert "&lt;script&gt;" in out
        assert "<script>" not in out


# ---------------------------------------------------------------------------
# _render_inline_segment — plain text escaping
# ---------------------------------------------------------------------------


class TestInlineEscaping:
    def test_angle_brackets(self):
        assert _render_inline_segment("a < b > c") == "a &lt; b &gt; c"

    def test_ampersand(self):
        assert _render_inline_segment("a & b") == "a &amp; b"

    def test_plain_text_unchanged(self):
        assert _render_inline_segment("hello world") == "hello world"

    def test_empty_string(self):
        assert _render_inline_segment("") == ""


# ---------------------------------------------------------------------------
# _render_inline_segment — backtick inline code
# ---------------------------------------------------------------------------


class TestInlineBacktickCode:
    def test_basic(self):
        assert _render_inline_segment("`code`") == "<code>code</code>"

    def test_html_chars_in_code_escaped(self):
        out = _render_inline_segment("`a < b`")
        assert out == "<code>a &lt; b</code>"

    def test_backtick_in_prose(self):
        out = _render_inline_segment("Use `git commit` often")
        assert out == "Use <code>git commit</code> often"

    def test_no_multiline_backtick(self):
        # Backtick span must not cross a newline.
        out = _render_inline_segment("`foo\nbar`")
        assert "<code>" not in out


# ---------------------------------------------------------------------------
# _render_inline_segment — markdown bold and italic
# ---------------------------------------------------------------------------


class TestInlineBold:
    def test_basic(self):
        assert _render_inline_segment("**bold**") == "<strong>bold</strong>"

    def test_in_prose(self):
        out = _render_inline_segment("This is **important** text")
        assert out == "This is <strong>important</strong> text"

    def test_html_chars_escaped(self):
        out = _render_inline_segment("**a < b**")
        assert out == "<strong>a &lt; b</strong>"

    def test_no_cross_newline(self):
        out = _render_inline_segment("**foo\nbar**")
        assert "<strong>" not in out

    def test_not_split_into_two_em(self):
        # **bold** must produce <strong>, not two <em> wrapping "bold".
        out = _render_inline_segment("**bold**")
        assert "<em>" not in out
        assert "<strong>bold</strong>" in out


class TestInlineItalic:
    def test_basic(self):
        assert _render_inline_segment("*italic*") == "<em>italic</em>"

    def test_in_prose(self):
        out = _render_inline_segment("This is *emphasised* text")
        assert out == "This is <em>emphasised</em> text"

    def test_html_chars_escaped(self):
        out = _render_inline_segment("*a > b*")
        assert out == "<em>a &gt; b</em>"

    def test_no_cross_newline(self):
        out = _render_inline_segment("*foo\nbar*")
        assert "<em>" not in out


class TestInlineBoldAndItalic:
    def test_both_in_same_string(self):
        out = _render_inline_segment("**bold** and *italic*")
        assert "<strong>bold</strong>" in out
        assert "<em>italic</em>" in out

    def test_bold_before_italic_ordering(self):
        # Ensure bold is matched before italic when ** appears first.
        out = _render_inline_segment("**b** *i*")
        assert out == "<strong>b</strong> <em>i</em>"


# ---------------------------------------------------------------------------
# _render_inline_segment — HTML tag pass-through
# ---------------------------------------------------------------------------


class TestHtmlPassThrough:
    def test_ul_li(self):
        inp = "<ul><li>item</li></ul>"
        out = _render_inline_segment(inp)
        assert out == inp

    def test_strong_from_llm(self):
        inp = "<strong>hi</strong>"
        assert _render_inline_segment(inp) == inp

    def test_em_from_llm(self):
        inp = "<em>hi</em>"
        assert _render_inline_segment(inp) == inp

    def test_br(self):
        assert _render_inline_segment("line<br>next") == "line<br>next"

    def test_plain_text_between_tags_escaped(self):
        out = _render_inline_segment("<ul><li>a < b</li></ul>")
        assert "&lt;" in out
        assert "<ul>" in out and "</ul>" in out


# ---------------------------------------------------------------------------
# _render_inline_segment — double-escaping fix (LLM <code> with entities)
# ---------------------------------------------------------------------------


class TestNoDoubleEscaping:
    def test_pre_escaped_entities_in_code(self):
        # LLM writes &lt; inside <code>; we must NOT re-escape the &.
        inp = "<code>git checkout -b &lt;branch&gt;</code>"
        out = _render_inline_segment(inp)
        assert "&amp;lt;" not in out
        assert "&lt;branch&gt;" in out

    def test_full_list_with_code(self):
        inp = "<ul><li><code>git checkout -b &lt;branch&gt;</code></li></ul>"
        out = _render_inline_segment(inp)
        assert "&amp;lt;" not in out
        assert "&lt;branch&gt;" in out

    def test_multiple_code_blocks(self):
        inp = "<code>a &lt; b</code> and <code>c &gt; d</code>"
        out = _render_inline_segment(inp)
        assert "&amp;lt;" not in out
        assert "&amp;gt;" not in out
        assert "&lt; b" in out
        assert "&gt; d" in out

    def test_depth_guard_orphan_close(self):
        # A stray </code> must not push depth negative; text after is escaped normally.
        out = _render_inline_segment("</code>a & b")
        assert "&amp;" in out  # '&' in plain text is escaped

    def test_code_with_class_attribute(self):
        inp = '<code class="language-python">x &lt; y</code>'
        out = _render_inline_segment(inp)
        assert "&amp;lt;" not in out
        assert "&lt; y" in out


# ---------------------------------------------------------------------------
# _render_code — integration (fenced + inline together)
# ---------------------------------------------------------------------------


class TestRenderCode:
    def test_fenced_block(self):
        inp = "```python\nx = 1\n```"
        out = _render_code(inp)
        assert "<pre><code" in out
        assert "x = 1" in out

    def test_inline_only(self):
        out = _render_code("Use `x` and **y**")
        assert "<code>x</code>" in out
        assert "<strong>y</strong>" in out

    def test_fenced_then_inline(self):
        inp = "```\ncode\n```\nUse *this*"
        out = _render_code(inp)
        assert "<pre><code>" in out
        assert "<em>this</em>" in out

    def test_plain_text_escaped(self):
        out = _render_code("a < b")
        assert "a &lt; b" in out

    def test_no_double_escape_end_to_end(self):
        inp = "Run <code>git push &lt;remote&gt;</code> to push."
        out = _render_code(inp)
        assert "&amp;lt;" not in out
        assert "&lt;remote&gt;" in out

    def test_fenced_code_html_escaped(self):
        inp = "```\na < b\n```"
        out = _render_code(inp)
        assert "&lt;" in out
