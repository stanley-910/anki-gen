"""Tests for image-attribution state helpers in anki_gen.confirm."""

from anki_gen.confirm import (
    _ImageDraft,
    _exclude_image_globally,
    _handle_image_menu_key,
    _ordered_unique_image_names,
    _render_image_panel,
    _serialize_image_assignments,
    _toggle_image_for_concept,
)


class TestOrderedUniqueImageNames:
    def test_preserves_order_and_deduplicates(self):
        assert _ordered_unique_image_names(["a.png", "b.png", "a.png"]) == [
            "a.png",
            "b.png",
        ]


class TestToggleImageForConcept:
    def test_assigns_image_to_concept(self):
        assignments: dict[str, set[str]] = {}
        excluded: set[str] = set()
        _toggle_image_for_concept(assignments, excluded, "Concept A", "a.png")
        assert assignments == {"Concept A": {"a.png"}}
        assert excluded == set()

    def test_reassigns_image_to_new_concept(self):
        assignments = {"Concept A": {"a.png"}}
        excluded: set[str] = set()
        _toggle_image_for_concept(assignments, excluded, "Concept B", "a.png")
        assert assignments == {"Concept B": {"a.png"}}

    def test_toggling_selected_image_removes_it(self):
        assignments = {"Concept A": {"a.png"}}
        excluded: set[str] = set()
        _toggle_image_for_concept(assignments, excluded, "Concept A", "a.png")
        assert assignments == {}

    def test_selecting_image_clears_exclusion(self):
        assignments: dict[str, set[str]] = {}
        excluded = {"a.png"}
        _toggle_image_for_concept(assignments, excluded, "Concept A", "a.png")
        assert assignments == {"Concept A": {"a.png"}}
        assert excluded == set()


class TestExcludeImageGlobally:
    def test_exclusion_removes_existing_assignment(self):
        assignments = {"Concept A": {"a.png"}, "Concept B": {"b.png"}}
        excluded: set[str] = set()
        _exclude_image_globally(assignments, excluded, "a.png")
        assert assignments == {"Concept B": {"b.png"}}
        assert excluded == {"a.png"}


class TestSerializeImageAssignments:
    def test_respects_image_order(self):
        assignments = {"Concept A": {"b.png", "a.png"}}
        result = _serialize_image_assignments(assignments, ["a.png", "b.png"])
        assert result == {"Concept A": ["a.png", "b.png"]}


class TestRenderImagePanel:
    def test_visible_without_active_draft(self):
        lines = _render_image_panel(["a.png"], None, 30)
        assert any("a.png" in line for line in lines)
        assert any("'i' edit image attribution" in line for line in lines)

    def test_active_draft_shows_target_concept(self):
        draft = _ImageDraft(target_concept="Concept A")
        lines = _render_image_panel(["a.png"], draft, 30)
        assert any("Concept A" in line for line in lines)

    def test_assigned_owner_shows_when_it_fits(self):
        lines = _render_image_panel(
            ["photo.png"],
            None,
            40,
            image_assignments={"Concept A": {"photo.png"}},
        )
        assert any("-> Concept A" in line for line in lines)

    def test_assigned_owner_clips_when_it_would_wrap(self):
        lines = _render_image_panel(
            ["photo.png"],
            None,
            26,
            image_assignments={"Very long concept name": {"photo.png"}},
        )
        assert any("->" in line for line in lines[1:])
        assert any("..." in line for line in lines[1:])

    def test_saved_assignments_show_outside_edit_mode(self):
        lines = _render_image_panel(
            ["photo.png"],
            None,
            40,
            image_assignments={"Concept A": {"photo.png"}},
        )
        assert any("Concept A" in line for line in lines)


class TestHandleImageMenuKey:
    def test_ctrl_c_quits_from_image_menu(self):
        draft = _ImageDraft(target_concept="Concept A")
        assert _handle_image_menu_key("\x03", ["a.png"], draft) == "quit"

    def test_q_quits_from_image_menu(self):
        draft = _ImageDraft(target_concept="Concept A")
        assert _handle_image_menu_key("q", ["a.png"], draft) == "quit"
