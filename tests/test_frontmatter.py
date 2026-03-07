"""Tests for YAML frontmatter tag extraction in parser.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anki_gen.parser import _extract_frontmatter_tags, parse_file


# ---------------------------------------------------------------------------
# _extract_frontmatter_tags
# ---------------------------------------------------------------------------


class TestExtractFrontmatterTags:
    def test_no_frontmatter_returns_source_unchanged(self):
        src = "# Hello\n\nsome content"
        body, tags = _extract_frontmatter_tags(src)
        assert body == src
        assert tags == []

    def test_frontmatter_without_tags_key(self):
        src = "---\ntitle: My Note\n---\n# Hello"
        body, tags = _extract_frontmatter_tags(src)
        assert body == "# Hello"
        assert tags == []

    def test_inline_yaml_list(self):
        src = "---\ntags: [math, linear-algebra]\n---\n# Hello"
        body, tags = _extract_frontmatter_tags(src)
        assert tags == ["math", "linear-algebra"]
        assert body == "# Hello"

    def test_block_yaml_list(self):
        src = textwrap.dedent("""\
            ---
            tags:
              - math
              - calculus
            ---
            # Hello
        """)
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["math", "calculus"]

    def test_comma_separated_string(self):
        src = "---\ntags: math, calculus, linear algebra\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["math", "calculus", "linear-algebra"]

    def test_spaces_replaced_with_hyphens(self):
        src = "---\ntags: [my tag, another one]\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["my-tag", "another-one"]

    def test_tags_lowercased(self):
        src = "---\ntags: [Math, CALCULUS]\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["math", "calculus"]

    def test_empty_tags_list(self):
        src = "---\ntags: []\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == []

    def test_frontmatter_not_at_start_ignored(self):
        src = "some text\n---\ntags: [math]\n---\nmore text"
        body, tags = _extract_frontmatter_tags(src)
        assert tags == []
        assert body == src

    def test_single_tag_string(self):
        src = "---\ntags: math\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["math"]

    def test_body_stripped_of_frontmatter(self):
        src = "---\ntags: [a]\n---\n# Title\n\nParagraph."
        body, _ = _extract_frontmatter_tags(src)
        assert body == "# Title\n\nParagraph."

    def test_invalid_yaml_returns_empty_tags(self):
        src = "---\ntags: [unclosed\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == []

    def test_whitespace_only_tags_skipped(self):
        # Comma-separated string with blank entries
        src = "---\ntags: ' , math , '\n---\nbody"
        _, tags = _extract_frontmatter_tags(src)
        assert tags == ["math"]


# ---------------------------------------------------------------------------
# parse_file integration — source_tags field
# ---------------------------------------------------------------------------


class TestParseFileSourceTags:
    def test_source_tags_populated(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("---\ntags: [ml, stats]\n---\n# My Note\n\nContent here.")
        doc = parse_file(f)
        assert doc.source_tags == ["ml", "stats"]

    def test_source_tags_empty_when_no_frontmatter(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# My Note\n\nContent here.")
        doc = parse_file(f)
        assert doc.source_tags == []

    def test_title_still_extracted_with_frontmatter(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("---\ntags: [a]\n---\n# Real Title\n\nContent.")
        doc = parse_file(f)
        assert doc.title == "Real Title"

    def test_plain_text_excludes_frontmatter(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("---\ntags: [a]\nauthor: Bob\n---\n# Title\n\nActual content.")
        doc = parse_file(f)
        assert "author" not in doc.plain_text
        assert "Bob" not in doc.plain_text
        assert "Actual content" in doc.plain_text
