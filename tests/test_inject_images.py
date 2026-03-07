"""Tests for inject_missed_images() in generator.py."""

from __future__ import annotations

from pathlib import Path

from anki_gen.generator import (
    _best_card_for_image,
    _image_near_any_concept,
    _images_already_placed,
    inject_missed_images,
)
from anki_gen.models import BasicCard, Card, DefinitionCard
from anki_gen.parser import ImageRef, ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(plain_text: str = "", images: list[ImageRef] | None = None) -> ParsedDocument:
    return ParsedDocument(
        source_path=Path("/tmp/test.md"),
        title="Test",
        plain_text=plain_text,
        concept_count=1,
        images=images or [],
    )


def _img(filename: str, alt: str = "") -> ImageRef:
    return ImageRef(
        filename=filename,
        alt_text=alt,
        width=None,
        height=None,
        resolved_path=None,
    )


def _basic(front: str = "Q", back: str = "A") -> BasicCard:
    return BasicCard(front=front, back=back)


def _defn(term: str = "Term", defn: str = "Def") -> DefinitionCard:
    return DefinitionCard(term=term, definition=defn)


def _answer(card: Card) -> str:
    return card.back if isinstance(card, BasicCard) else card.definition


# ---------------------------------------------------------------------------
# _images_already_placed
# ---------------------------------------------------------------------------


class TestImagesAlreadyPlaced:
    def test_empty_cards(self):
        assert _images_already_placed([]) == set()

    def test_basic_card_back_has_image(self):
        card = _basic(back='Answer <img src="diagram.png">')
        assert _images_already_placed([card]) == {"diagram.png"}

    def test_basic_card_front_has_image(self):
        card = _basic(front='<img src="fig.webp"> Q')
        assert _images_already_placed([card]) == {"fig.webp"}

    def test_definition_card_definition_has_image(self):
        card = _defn(defn='Text <img src="chart.svg" alt="chart">')
        assert _images_already_placed([card]) == {"chart.svg"}

    def test_multiple_cards_multiple_images(self):
        cards = [
            _basic(back='<img src="a.png">'),
            _defn(defn='<img src="b.png">'),
        ]
        assert _images_already_placed(cards) == {"a.png", "b.png"}

    def test_no_images(self):
        assert _images_already_placed([_basic(), _defn()]) == set()

    def test_single_quotes_in_src(self):
        card = _basic(back="<img src='x.jpg'>")
        assert _images_already_placed([card]) == {"x.jpg"}


# ---------------------------------------------------------------------------
# _best_card_for_image
# ---------------------------------------------------------------------------


class TestBestCardForImage:
    def test_returns_zero_when_filename_not_in_text(self):
        cards = [_basic("question", "answer")]
        ref = _img("missing.png")
        assert _best_card_for_image(ref, cards, "some unrelated text") == 0

    def test_picks_card_with_overlapping_words(self):
        # plain_text: image marker surrounded by words that appear in card 1
        plain_text = (
            "sigmoid saturated derivative [Image: deriv-plot.png] gradient vanishes"
        )
        cards = [
            _basic("What is relu?", "relu replaces sigmoid"),
            _basic("Why does gradient vanish?", "sigmoid saturated derivative is tiny"),
        ]
        ref = _img("deriv-plot.png")
        idx = _best_card_for_image(ref, cards, plain_text)
        assert idx == 1  # card 1 has more overlap with the window words

    def test_ignores_html_tags_in_card_text(self):
        plain_text = "logistic function [Image: logistic.png] probability output"
        cards = [
            _basic(
                front="What is logistic <b>function</b>?",
                back="<p>probability output</p>",
            ),
        ]
        ref = _img("logistic.png")
        assert _best_card_for_image(ref, cards, plain_text) == 0


# ---------------------------------------------------------------------------
# inject_missed_images
# ---------------------------------------------------------------------------


class TestInjectMissedImages:
    def test_no_images_returns_cards_unchanged(self):
        cards = [_basic()]
        result = inject_missed_images(cards, _doc())
        assert result == cards

    def test_empty_cards_returns_empty(self):
        doc = _doc(images=[_img("x.png")])
        assert inject_missed_images([], doc) == []

    def test_image_already_placed_not_duplicated(self):
        card = _basic(back='A <img src="x.png">')
        doc = _doc(plain_text="x.png", images=[_img("x.png")])
        result = inject_missed_images([card], doc)
        assert _answer(result[0]).count("<img") == 1

    def test_missed_image_appended_to_basic_back(self):
        card = _basic(front="Q", back="A")
        doc = _doc(plain_text="[Image: chart.png]", images=[_img("chart.png")])
        result = inject_missed_images([card], doc)
        assert isinstance(result[0], BasicCard)
        assert '<br><img src="chart.png"><br>' in result[0].back

    def test_missed_image_with_alt_text(self):
        card = _basic()
        doc = _doc(
            plain_text="[Image: fig.webp]",
            images=[_img("fig.webp", alt="logistic curve")],
        )
        result = inject_missed_images([card], doc)
        assert 'alt="logistic curve"' in _answer(result[0])

    def test_missed_image_appended_to_definition_definition(self):
        card = _defn(term="Vanishing gradient", defn="Gradients shrink")
        doc = _doc(plain_text="[Image: grad.png]", images=[_img("grad.png")])
        result = inject_missed_images([card], doc)
        assert isinstance(result[0], DefinitionCard)
        assert '<br><img src="grad.png"><br>' in result[0].definition

    def test_injected_image_wrapped_with_line_breaks(self):
        card = _basic(front="Q", back="A")
        doc = _doc(plain_text="[Image: chart.png]", images=[_img("chart.png")])
        result = inject_missed_images([card], doc)
        assert _answer(result[0]).endswith('<br><img src="chart.png"><br>')

    def test_multiple_missed_images_injected(self):
        cards = [_basic("Q1", "A1"), _basic("Q2", "A2")]
        doc = _doc(
            plain_text="[Image: a.png] Q1 text   Q2 text [Image: b.png]",
            images=[_img("a.png"), _img("b.png")],
        )
        result = inject_missed_images(cards, doc)
        all_html = " ".join(
            (c.back if isinstance(c, BasicCard) else c.definition) for c in result
        )
        assert "a.png" in all_html
        assert "b.png" in all_html

    def test_card_tags_preserved(self):
        card = BasicCard(front="Q", back="A", tags=["ml", "neural-nets"])
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images([card], doc)
        assert result[0].tags == ["ml", "neural-nets"]

    def test_definition_card_tags_preserved(self):
        card = DefinitionCard(term="T", definition="D", tags=["tag1"])
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images([card], doc)
        assert result[0].tags == ["tag1"]

    def test_basic_reversed_preserved(self):
        card = BasicCard(front="Q", back="A", reversed=True)
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images([card], doc)
        assert isinstance(result[0], BasicCard)
        assert result[0].reversed is True

    def test_alt_text_html_escaped(self):
        card = _basic()
        doc = _doc(
            plain_text="[Image: x.png]",
            images=[_img("x.png", alt='<script>alert("xss")</script>')],
        )
        result = inject_missed_images([card], doc)
        assert "<script>" not in _answer(result[0])
        assert "&lt;script&gt;" in _answer(result[0])

    def test_proximity_picks_nearest_card(self):
        # Image marker appears right after text that matches card 0's content
        plain_text = (
            "The logistic function is saturated at large z values. "
            "[Image: saturation.png] "
            "ReLU avoids this problem entirely."
        )
        cards = [
            _basic(
                front="What is logistic function saturation?",
                back="Output saturated at large z values",
            ),
            _basic(
                front="What is ReLU?",
                back="Rectified Linear Unit avoids saturation",
            ),
        ]
        doc = _doc(plain_text=plain_text, images=[_img("saturation.png")])
        result = inject_missed_images(cards, doc)
        # saturation.png is closer to card 0's content
        assert '<img src="saturation.png">' in _answer(result[0])
        assert '<img src="saturation.png">' not in _answer(result[1])

    def test_manual_assignment_overrides_auto_placement(self):
        cards = [
            _basic(front="Logistic?", back="Saturation"),
            _basic(front="ReLU?", back="Rectified linear unit"),
        ]
        doc = _doc(
            plain_text="logistic saturation [Image: sat.png] relu",
            images=[_img("sat.png")],
        )
        result = inject_missed_images(
            cards,
            doc,
            confirmed_concepts=["Logistic saturation", "ReLU activation"],
            concept_order=["Logistic saturation", "ReLU activation"],
            manual_image_assignments={"ReLU activation": ["sat.png"]},
        )
        assert '<img src="sat.png">' not in _answer(result[0])
        assert '<img src="sat.png">' in _answer(result[1])

    def test_excluded_image_not_injected(self):
        cards = [_basic()]
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images(cards, doc, excluded_images={"x.png"})
        assert '<img src="x.png">' not in _answer(result[0])

    def test_manual_assignment_removes_prior_auto_image(self):
        cards = [
            _basic(front="Q1", back='A1 <img src="x.png">'),
            _basic(front="Q2", back="A2"),
        ]
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images(
            cards,
            doc,
            concept_order=["C1", "C2"],
            manual_image_assignments={"C2": ["x.png"]},
        )
        assert '<img src="x.png">' not in _answer(result[0])
        assert '<img src="x.png">' in _answer(result[1])

    def test_excluded_image_removed_from_existing_card(self):
        cards = [_basic(back='A<br><img src="x.png"><br>')]
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images(cards, doc, excluded_images={"x.png"})
        assert '<img src="x.png">' not in _answer(result[0])

    def test_excluding_image_removes_surrounding_breaks(self):
        cards = [_basic(back='A<br><img src="x.png"><br>B')]
        doc = _doc(plain_text="[Image: x.png]", images=[_img("x.png")])
        result = inject_missed_images(cards, doc, excluded_images={"x.png"})
        assert _answer(result[0]) == "AB"


# ---------------------------------------------------------------------------
# _image_near_any_concept
# ---------------------------------------------------------------------------


class TestImageNearAnyConcept:
    def test_concept_word_in_window_returns_true(self):
        plain_text = "logistic function saturated [Image: log-sat.png] more text"
        ref = _img("log-sat.png")
        assert (
            _image_near_any_concept(ref, plain_text, ["Logistic function saturation"])
            is True
        )

    def test_unrelated_concept_returns_false(self):
        plain_text = "logistic function saturated [Image: log-sat.png] more text"
        ref = _img("log-sat.png")
        assert _image_near_any_concept(ref, plain_text, ["ReLU activation"]) is False

    def test_filename_not_in_text_returns_true(self):
        # If the marker isn't found we don't block injection (safe default)
        ref = _img("mystery.png")
        assert _image_near_any_concept(ref, "unrelated text", ["some concept"]) is True

    def test_multiple_concepts_any_match_returns_true(self):
        plain_text = "vanishing gradient problem [Image: grad.png]"
        ref = _img("grad.png")
        concepts = ["ReLU activation", "Vanishing gradient problem"]
        assert _image_near_any_concept(ref, plain_text, concepts) is True

    def test_stopwords_ignored(self):
        # "the" and "of" are stopwords and must not trigger a match on their own
        plain_text = "the study of [Image: x.png]"
        ref = _img("x.png")
        assert _image_near_any_concept(ref, plain_text, ["the of"]) is False


# ---------------------------------------------------------------------------
# inject_missed_images — confirmed_concepts filtering
# ---------------------------------------------------------------------------


class TestInjectMissedImagesConceptFilter:
    def test_image_near_confirmed_concept_is_injected(self):
        plain_text = "logistic saturation [Image: sat.png] more text"
        cards = [_basic("What is saturation?", "Logistic output near 0 or 1")]
        doc = _doc(plain_text=plain_text, images=[_img("sat.png")])
        result = inject_missed_images(
            cards, doc, confirmed_concepts=["Logistic function saturation"]
        )
        assert '<img src="sat.png">' in _answer(result[0])

    def test_image_near_rejected_concept_not_injected(self):
        # Image is near "logistic saturation" text; confirmed list only has "ReLU activation".
        # Window words: logistic, function, saturated, sat, png, more, text, here
        # Concept words: relu, activation — no overlap → image skipped.
        plain_text = "logistic function saturated [Image: sat.png] more text here"
        cards = [_basic("What is ReLU?", "Avoids saturation")]
        doc = _doc(plain_text=plain_text, images=[_img("sat.png")])
        result = inject_missed_images(
            cards, doc, confirmed_concepts=["ReLU activation"]
        )
        assert _answer(result[0]) == "Avoids saturation"  # no img injected

    def test_image_near_rejected_concept_truly_unrelated(self):
        # Image marker is surrounded ONLY by words from a rejected concept.
        plain_text = "sigmoid derivative vanishes [Image: deriv.png] end"
        cards = [_basic("What is batch normalization?", "Normalises layer inputs")]
        doc = _doc(plain_text=plain_text, images=[_img("deriv.png")])
        result = inject_missed_images(
            cards, doc, confirmed_concepts=["Batch normalization"]
        )
        # "sigmoid", "derivative", "vanishes" have no overlap with "Batch normalization"
        assert '<img src="deriv.png">' not in _answer(result[0])

    def test_no_confirmed_concepts_still_injects(self):
        # confirmed_concepts=None means direct path — inject unconditionally
        plain_text = "[Image: x.png]"
        cards = [_basic()]
        doc = _doc(plain_text=plain_text, images=[_img("x.png")])
        result = inject_missed_images(cards, doc, confirmed_concepts=None)
        assert '<img src="x.png">' in _answer(result[0])

    def test_empty_confirmed_concepts_list_blocks_all(self):
        # Empty list → no confirmed concepts → no image can match → nothing injected
        plain_text = "logistic saturation [Image: sat.png]"
        cards = [_basic()]
        doc = _doc(plain_text=plain_text, images=[_img("sat.png")])
        result = inject_missed_images(cards, doc, confirmed_concepts=[])
        assert '<img src="sat.png">' not in _answer(result[0])
