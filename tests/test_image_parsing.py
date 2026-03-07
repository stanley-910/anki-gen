"""Tests for image reference parsing and substitution in parser.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anki_gen.parser import (
    ImageRef,
    _image_ref_to_text,
    _is_size_field,
    _parse_obsidian_size,
    _resolve_image,
    _substitute_images,
    parse_file,
)


# ---------------------------------------------------------------------------
# _is_size_field
# ---------------------------------------------------------------------------


class TestIsSizeField:
    def test_pure_integer(self):
        assert _is_size_field("386") is True

    def test_width_x_height(self):
        assert _is_size_field("386x14") is True

    def test_height_only(self):
        assert _is_size_field("x14") is True

    def test_text_is_not_size(self):
        assert _is_size_field("figure subtitle") is False

    def test_empty_string_is_not_size(self):
        assert _is_size_field("") is False

    def test_mixed_text_and_digits(self):
        assert _is_size_field("figure1") is False


# ---------------------------------------------------------------------------
# _parse_obsidian_size
# ---------------------------------------------------------------------------


class TestParseObsidianSize:
    def test_width_only(self):
        assert _parse_obsidian_size("386") == (386, None)

    def test_width_x_height(self):
        assert _parse_obsidian_size("386x14") == (386, 14)

    def test_height_only(self):
        w, h = _parse_obsidian_size("x14")
        assert w is None
        assert h == 14

    def test_case_insensitive(self):
        assert _parse_obsidian_size("200X100") == (200, 100)


# ---------------------------------------------------------------------------
# _image_ref_to_text
# ---------------------------------------------------------------------------


class TestImageRefToText:
    def test_filename_and_alt(self):
        ref = ImageRef("diagram.webp", "figure subtitle", None, None, None)
        assert _image_ref_to_text(ref) == '[Image: "figure subtitle" (diagram.webp)]'

    def test_filename_only(self):
        ref = ImageRef("diagram.webp", "", None, None, None)
        assert _image_ref_to_text(ref) == "[Image: diagram.webp]"

    def test_width_and_height(self):
        ref = ImageRef("img.png", "chart", 386, 14, None)
        assert _image_ref_to_text(ref) == '[Image: "chart" (img.png, 386x14px)]'

    def test_width_only(self):
        ref = ImageRef("img.png", "chart", 200, None, None)
        assert _image_ref_to_text(ref) == '[Image: "chart" (img.png, 200px wide)]'

    def test_filename_with_size_no_alt(self):
        ref = ImageRef("img.png", "", 400, 300, None)
        assert _image_ref_to_text(ref) == "[Image: img.png, 400x300px]"


# ---------------------------------------------------------------------------
# _substitute_images — Obsidian format
# ---------------------------------------------------------------------------


class TestSubstituteObsidianImages:
    def _sub(self, source: str) -> tuple[str, list[ImageRef]]:
        return _substitute_images(source, Path("/tmp"))

    def test_filename_only(self):
        src, refs = self._sub("![[image.webp]]")
        assert refs[0].filename == "image.webp"
        assert refs[0].alt_text == ""
        assert refs[0].width is None
        assert refs[0].height is None
        assert src == "[Image: image.webp]"

    def test_filename_and_alt(self):
        src, refs = self._sub("![[image.webp|figure subtitle]]")
        assert refs[0].alt_text == "figure subtitle"
        assert refs[0].width is None
        assert src == '[Image: "figure subtitle" (image.webp)]'

    def test_filename_and_width_only(self):
        src, refs = self._sub("![[image.webp|386]]")
        assert refs[0].alt_text == ""
        assert refs[0].width == 386
        assert refs[0].height is None
        assert src == "[Image: image.webp, 386px wide]"

    def test_filename_alt_and_width(self):
        src, refs = self._sub("![[image.webp|caption|386]]")
        assert refs[0].alt_text == "caption"
        assert refs[0].width == 386
        assert refs[0].height is None
        assert src == '[Image: "caption" (image.webp, 386px wide)]'

    def test_full_obsidian_syntax(self):
        """Matches the example from the feature spec: ![[image.webp|figure subtitle|386x14]]"""
        src, refs = self._sub("![[image.webp|figure subtitle|386x14]]")
        assert refs[0].filename == "image.webp"
        assert refs[0].alt_text == "figure subtitle"
        assert refs[0].width == 386
        assert refs[0].height == 14
        assert src == '[Image: "figure subtitle" (image.webp, 386x14px)]'

    def test_width_x_height_no_alt(self):
        src, refs = self._sub("![[image.webp|386x14]]")
        assert refs[0].alt_text == ""
        assert refs[0].width == 386
        assert refs[0].height == 14
        assert src == "[Image: image.webp, 386x14px]"

    def test_inline_in_text(self):
        src, refs = self._sub("Before ![[img.png|caption]] after.")
        assert "[Image:" in src
        assert src.startswith("Before ")
        assert src.endswith(" after.")
        assert len(refs) == 1

    def test_multiple_images(self):
        src, refs = self._sub("![[a.png]] and ![[b.png|B caption]]")
        assert len(refs) == 2
        filenames = {r.filename for r in refs}
        assert filenames == {"a.png", "b.png"}

    def test_basename_only_from_path(self):
        """Obsidian filenames may carry sub-paths; only the basename matters."""
        src, refs = self._sub("![[subfolder/image.webp]]")
        assert refs[0].filename == "image.webp"


# ---------------------------------------------------------------------------
# _substitute_images — Standard Markdown format
# ---------------------------------------------------------------------------


class TestSubstituteStandardImages:
    def _sub(self, source: str) -> tuple[str, list[ImageRef]]:
        return _substitute_images(source, Path("/tmp"))

    def test_alt_and_path(self):
        src, refs = self._sub("![alt text](path/to/image.png)")
        assert refs[0].filename == "image.png"
        assert refs[0].alt_text == "alt text"
        assert refs[0].width is None
        assert src == '[Image: "alt text" (image.png)]'

    def test_no_alt(self):
        src, refs = self._sub("![](image.png)")
        assert refs[0].filename == "image.png"
        assert refs[0].alt_text == ""
        assert src == "[Image: image.png]"

    def test_inline_in_text(self):
        src, refs = self._sub("See ![diagram](fig.svg) for details.")
        assert src == 'See [Image: "diagram" (fig.svg)] for details.'
        assert len(refs) == 1

    def test_basename_extracted_from_path(self):
        _, refs = self._sub("![](assets/images/chart.webp)")
        assert refs[0].filename == "chart.webp"


# ---------------------------------------------------------------------------
# _substitute_images — mixed formats
# ---------------------------------------------------------------------------


class TestSubstituteMixed:
    def test_obsidian_and_standard_in_same_source(self):
        source = "![[a.webp|first]] and ![second](b.png)"
        src, refs = _substitute_images(source, Path("/tmp"))
        assert len(refs) == 2
        filenames = {r.filename for r in refs}
        assert filenames == {"a.webp", "b.png"}

    def test_no_images_returns_source_unchanged(self):
        source = "Just plain text, no images here."
        src, refs = _substitute_images(source, Path("/tmp"))
        assert src == source
        assert refs == []


# ---------------------------------------------------------------------------
# _resolve_image
# ---------------------------------------------------------------------------


class TestResolveImage:
    def test_found_in_same_dir(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG")
        assert _resolve_image("photo.png", tmp_path) == img.resolve()

    def test_found_in_attachments_subdir(self, tmp_path):
        subdir = tmp_path / "attachments"
        subdir.mkdir()
        img = subdir / "diagram.svg"
        img.write_bytes(b"<svg/>")
        result = _resolve_image("diagram.svg", tmp_path)
        assert result == img.resolve()

    def test_found_in_assets_subdir(self, tmp_path):
        subdir = tmp_path / "assets"
        subdir.mkdir()
        img = subdir / "chart.webp"
        img.write_bytes(b"RIFF")
        assert _resolve_image("chart.webp", tmp_path) == img.resolve()

    def test_not_found_returns_none(self, tmp_path):
        assert _resolve_image("missing.png", tmp_path) is None

    def test_same_dir_preferred_over_subdir(self, tmp_path):
        """File in the doc dir takes priority over a file in attachments/."""
        (tmp_path / "img.png").write_bytes(b"A")
        sub = tmp_path / "attachments"
        sub.mkdir()
        (sub / "img.png").write_bytes(b"B")
        result = _resolve_image("img.png", tmp_path)
        assert result == (tmp_path / "img.png").resolve()


# ---------------------------------------------------------------------------
# parse_file integration
# ---------------------------------------------------------------------------


class TestParseFileImages:
    def _write_md(self, tmp_path: Path, content: str) -> Path:
        md = tmp_path / "notes.md"
        md.write_text(content, encoding="utf-8")
        return md

    def test_obsidian_image_appears_in_plain_text(self, tmp_path):
        md = self._write_md(
            tmp_path,
            "# Title\n\nSome concept.\n\n![[diagram.png|Activation function]]\n",
        )
        doc = parse_file(md)
        assert "[Image:" in doc.plain_text
        assert "diagram.png" in doc.plain_text

    def test_standard_image_appears_in_plain_text(self, tmp_path):
        md = self._write_md(tmp_path, "# Title\n\n![alt](fig.png)\n")
        doc = parse_file(md)
        assert "[Image:" in doc.plain_text
        assert "fig.png" in doc.plain_text

    def test_images_list_populated(self, tmp_path):
        md = self._write_md(
            tmp_path, "# Title\n\n![[a.webp|caption|100x50]]\n\n![b](b.png)\n"
        )
        doc = parse_file(md)
        assert len(doc.images) == 2
        filenames = {r.filename for r in doc.images}
        assert filenames == {"a.webp", "b.png"}

    def test_resolved_path_set_when_file_exists(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG")
        md = self._write_md(tmp_path, "# Title\n\n![[photo.png]]\n")
        doc = parse_file(md)
        assert doc.images[0].resolved_path == img.resolve()

    def test_resolved_path_none_when_missing(self, tmp_path):
        md = self._write_md(tmp_path, "# Title\n\n![[ghost.png]]\n")
        doc = parse_file(md)
        assert doc.images[0].resolved_path is None

    def test_image_marker_position_preserved(self, tmp_path):
        """Image marker appears between the surrounding text lines."""
        md = self._write_md(
            tmp_path,
            textwrap.dedent("""\
                # Title

                Concept A is important.

                ![[fig.png|Figure A]]

                Concept B follows.
            """),
        )
        doc = parse_file(md)
        idx_a = doc.plain_text.find("Concept A")
        idx_img = doc.plain_text.find("[Image:")
        idx_b = doc.plain_text.find("Concept B")
        assert idx_a < idx_img < idx_b

    def test_no_images_gives_empty_list(self, tmp_path):
        md = self._write_md(tmp_path, "# Title\n\nJust text.\n")
        doc = parse_file(md)
        assert doc.images == []

    def test_images_disabled_removes_markers_and_refs(self, tmp_path):
        md = self._write_md(
            tmp_path,
            "# Title\n\nBefore\n\n![[fig.png|Caption]]\n\nAfter\n",
        )
        doc = parse_file(md, images_enabled=False)
        assert doc.images == []
        assert "[Image:" not in doc.plain_text
        assert "fig.png" not in doc.plain_text
        assert "Before" in doc.plain_text
        assert "After" in doc.plain_text
