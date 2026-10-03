"""Unit tests for CoT segmentation.

The load-bearing property is losslessness: if ``render()`` cannot reproduce
the original trace, then a "perturbed" CoT differs from the original in ways
the experiment never accounted for, and every label downstream is suspect.
"""

from __future__ import annotations

import pytest

from stepwise.extractor import (
    ExtractedCoT,
    extract_steps,
    splice_step,
    splice_steps,
)

from .conftest import MARKED_COT, NUMBERED_COT, PARAGRAPH_COT, PROSE_COT


# --------------------------------------------------------------------------
# Round-trip invariants
# --------------------------------------------------------------------------


def test_render_without_overrides_reproduces_the_source(any_cot):
    extracted = extract_steps(any_cot)
    assert extracted.render() == any_cot


def test_step_offsets_point_at_the_step_text(any_cot):
    extracted = extract_steps(any_cot)
    for step in extracted:
        assert any_cot[step.start : step.end] == step.text


def test_steps_are_numbered_from_one_without_gaps(any_cot):
    extracted = extract_steps(any_cot)
    assert [s.index for s in extracted] == list(range(1, len(extracted) + 1))


def test_concatenating_rendered_steps_rebuilds_the_trace(any_cot):
    extracted = extract_steps(any_cot)
    assert extracted.preamble + "".join(s.rendered for s in extracted) == any_cot


# --------------------------------------------------------------------------
# Marked formats
# --------------------------------------------------------------------------


def test_explicit_step_markers_are_detected():
    extracted = extract_steps(MARKED_COT)
    assert extracted.style == "marked"
    assert len(extracted) == 3
    assert extracted.step(2).text == "In May she sold half as many, so 48 / 2 = 24."
    assert extracted.step(2).marker == "Step 2: "


def test_marker_is_not_part_of_the_step_text():
    for step in extract_steps(MARKED_COT):
        assert not step.text.lower().startswith("step")


def test_numbered_list_markers_are_detected():
    extracted = extract_steps(NUMBERED_COT)
    assert extracted.style == "marked"
    assert len(extracted) == 3
    assert extracted.step(1).marker == "1. "
    assert extracted.step(3).text == "So she earned 50 * 0.2 = $10."


def test_inline_step_markers_on_one_line():
    cot = "Step 1: find the total. Step 2: divide by two. Step 3: report it."
    extracted = extract_steps(cot)
    assert extracted.style == "marked"
    assert [s.text for s in extracted] == [
        "find the total.",
        "divide by two.",
        "report it.",
    ]


def test_multiline_marked_step_keeps_its_continuation():
    cot = "Step 1: Set up the equation.\n    2x + 3 = 11\nStep 2: Solve for x, giving x = 4."
    extracted = extract_steps(cot)
    assert len(extracted) == 2
    assert "2x + 3 = 11" in extracted.step(1).text
    assert extracted.render() == cot


def test_bullet_markers_are_detected():
    cot = "- The train leaves at noon.\n- It travels for three hours.\n- So it arrives at 3pm."
    extracted = extract_steps(cot)
    assert extracted.style == "marked"
    assert len(extracted) == 3


def test_text_before_the_first_marker_becomes_preamble():
    cot = "Let me work through this.\nStep 1: Count the apples.\nStep 2: Subtract the eaten ones."
    extracted = extract_steps(cot)
    assert extracted.preamble == "Let me work through this.\n"
    assert len(extracted) == 2
    assert extracted.render() == cot


# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------


def test_prose_is_split_on_sentence_boundaries():
    extracted = extract_steps(PROSE_COT)
    assert extracted.style == "prose"
    assert len(extracted) == 4
    assert extracted.step(1).text == "Natalia sold 48 clips in April."


def test_prose_step_text_has_no_leading_whitespace():
    for step in extract_steps(PROSE_COT):
        assert step.text == step.text.lstrip()


def test_decimals_do_not_split_a_sentence():
    extracted = extract_steps("The rate is 3.5 per hour. Over 4 hours that is 14.0 units.")
    assert len(extracted) == 2


def test_abbreviations_do_not_split_a_sentence():
    extracted = extract_steps(
        "We use the standard rate, i.e. 12 per hour, for the whole shift. "
        "That covers the full period."
    )
    assert len(extracted) == 2


def test_initials_do_not_split_a_sentence():
    extracted = extract_steps("J. Smith owned the shop for years. He sold it in 2019.")
    assert len(extracted) == 2


def test_paragraph_breaks_end_a_step():
    extracted = extract_steps(PARAGRAPH_COT)
    assert len(extracted) == 3
    assert extracted.step(1).trailing == "\n\n"


def test_short_fragments_merge_into_the_previous_step():
    extracted = extract_steps("We start with 10. Add 5. That gives fifteen apples in total.")
    assert len(extracted) == 2
    assert extracted.step(1).text == "We start with 10. Add 5."


def test_merging_can_be_disabled():
    extracted = extract_steps(
        "We start with 10. Add 5. That gives fifteen apples in total.", min_step_chars=0
    )
    assert len(extracted) == 3


def test_a_single_sentence_yields_one_step():
    extracted = extract_steps("There is only one sentence here and nothing else")
    assert extracted.style == "single"
    assert len(extracted) == 1


@pytest.mark.parametrize("cot", ["", "   ", "\n\n", None])
def test_empty_input_yields_no_steps(cot):
    extracted = extract_steps(cot)
    assert len(extracted) == 0
    assert extracted.style == "empty"
    assert extracted.render() == (cot or "")


# --------------------------------------------------------------------------
# Splicing
# --------------------------------------------------------------------------


def test_splice_replaces_only_the_target_step():
    spliced = splice_step(MARKED_COT, 2, "In May she sold twice as many, so 48 * 2 = 96.")
    assert "48 * 2 = 96" in spliced
    assert "Natalia sold 48 clips in April." in spliced
    assert "Altogether she sold 48 + 24 = 72 clips." in spliced
    assert "48 / 2 = 24" not in spliced


def test_splice_preserves_markers_and_line_structure():
    spliced = splice_step(MARKED_COT, 2, "REPLACED")
    assert spliced.splitlines()[1] == "Step 2: REPLACED"
    assert len(spliced.splitlines()) == len(MARKED_COT.splitlines())


def test_splice_preserves_paragraph_spacing():
    spliced = splice_step(PARAGRAPH_COT, 2, "REPLACED")
    assert "\n\nREPLACED\n\n" in spliced


def test_splicing_the_original_text_is_a_no_op(any_cot):
    extracted = extract_steps(any_cot)
    for step in extracted:
        assert splice_step(extracted, step.index, step.text) == any_cot


@pytest.mark.parametrize("index", [1, 2, 3])
def test_every_step_index_is_spliceable(index):
    spliced = splice_step(MARKED_COT, index, "REPLACED")
    assert "REPLACED" in spliced
    assert spliced.count("REPLACED") == 1


def test_splice_accepts_a_pre_extracted_cot():
    extracted = extract_steps(MARKED_COT)
    assert splice_step(extracted, 1, "X") == splice_step(MARKED_COT, 1, "X")


def test_splice_rejects_an_out_of_range_index():
    with pytest.raises(IndexError):
        splice_step(MARKED_COT, 99, "REPLACED")


def test_splice_rejects_a_zero_index_since_steps_are_one_based():
    with pytest.raises(IndexError):
        splice_step(MARKED_COT, 0, "REPLACED")


def test_multi_step_splice():
    spliced = splice_steps(MARKED_COT, {1: "A", 3: "C"})
    assert spliced == "Step 1: A\nStep 2: In May she sold half as many, so 48 / 2 = 24.\nStep 3: C"


def test_splicing_multiline_replacement_keeps_surrounding_steps():
    spliced = splice_step(MARKED_COT, 2, "Line one.\nLine two.")
    assert "Step 3: Altogether" in spliced
    assert "Step 1: Natalia" in spliced


def test_step_lookup_raises_for_unknown_index():
    with pytest.raises(IndexError):
        extract_steps(MARKED_COT).step(7)


def test_to_dict_shape():
    payload = extract_steps(MARKED_COT).to_dict()
    assert payload["style"] == "marked"
    assert payload["steps"][0] == {
        "index": 1,
        "text": "Natalia sold 48 clips in April.",
        "marker": "Step 1:",
    }


def test_extracted_cot_is_iterable_and_sized():
    extracted = extract_steps(MARKED_COT)
    assert isinstance(extracted, ExtractedCoT)
    assert len(list(extracted)) == len(extracted) == 3
