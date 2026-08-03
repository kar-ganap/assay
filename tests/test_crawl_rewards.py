"""The Phase 0.1 graders and their two deliberately different extractors.

``R_binary`` (the ladder) is **lenient on shape, binary on correctness** — it takes the last integer
anywhere. ``R_format`` (ablation B) requires a strict ``<answer>N</answer>``. That they differ is not
an inconsistency; it is the entire point of having grader variants over one task set.

Why lenient, decided 2026-07-27 on measured evidence: strict-tag `parse_fail_rate` rose
**monotonically as pass rate fell**, in both task families independently, because harder problems
make the model reason out loud and the longer it reasons the less reliably it closes the tag. A
strict grader therefore filters *hard* problems rather than *badly formatted* ones — excluding the
exact difficulty band the screen exists to find.
"""

from __future__ import annotations

import pytest

from assay.crawl import rewards
from assay.crawl.rewards import Outcome

# --------------------------------------------------------------------------------------
# R_binary's extractor: last integer anywhere
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("<answer>42</answer>", "42"),  # the nominal shape
        ("<a>7</a>", "7"),  # truncated tag — 10.8% of observed completions
        ("(answer) 7", "7"),  # paraphrased tag — 9.4%
        ("<7>", "7"),  # bare angle brackets — 6.1%
        ("45 * 8 = 360", "360"),  # restates the problem, answer last — 5.2%
        ("After analyzing the string, the letter g appears 5 times", "5"),
        ("8", "8"),  # bare integer — the original failure mode
        ("<answer>  7\n</answer>", "7"),
        ("<answer>-3</answer>", "-3"),
    ],
)
def test_recovers_the_answer_from_every_observed_shape(completion: str, expected: str) -> None:
    """These nine shapes are taken from real Llama-3.2-1B output, not invented."""
    assert rewards.extract_final_integer(completion) == expected


def test_takes_the_last_integer_when_several_appear() -> None:
    assert rewards.extract_final_integer("<answer>1</answer> wait, <answer>2</answer>") == "2"
    assert rewards.extract_final_integer("125 + 200 = 325") == "325"


def test_accepts_thousands_separators() -> None:
    """The model emits ``1,125`` for 4-digit results; a comma-grouped integer is the same integer."""
    assert rewards.extract_final_integer("<answer>1,125</answer>") == "1125"
    assert rewards.extract_final_integer("the total is 9,801") == "9801"


def test_no_integer_at_all_extracts_nothing() -> None:
    """Under the lenient extractor, PARSE_FAIL means a genuine non-answer."""
    assert rewards.extract_final_integer("<answer>forty-two</answer>") is None
    assert rewards.extract_final_integer("I am not sure.") is None
    assert rewards.extract_final_integer("") is None


# --------------------------------------------------------------------------------------
# R_format's extractor: strict tag. Ablation B needs the shape to be a real constraint.
# --------------------------------------------------------------------------------------


def test_tagged_extractor_requires_the_full_tag() -> None:
    assert rewards.extract_tagged_answer("<answer>42</answer>") == "42"
    assert rewards.extract_tagged_answer("<answer>1,125</answer>") == "1125"
    assert rewards.extract_tagged_answer("<answer>  7\n</answer>") == "7"


@pytest.mark.parametrize(
    "completion", ["8", "<a>7</a>", "(answer) 7", "<7>", "<answer>42", "answer>7", "45 * 8 = 360"]
)
def test_tagged_extractor_rejects_near_misses(completion: str) -> None:
    """The near-miss shapes that made strict grading unusable for R_binary."""
    assert rewards.extract_tagged_answer(completion) is None


def test_tagged_extractor_rejects_malformed_comma_grouping() -> None:
    assert rewards.extract_tagged_answer("<answer>1,2,3</answer>") is None
    assert rewards.extract_tagged_answer("<answer>12,34</answer>") is None


def test_the_two_extractors_genuinely_differ() -> None:
    """If these ever agree everywhere, the grader variants have collapsed into one."""
    completion = "45 * 8 = 360"
    assert rewards.extract_final_integer(completion) == "360"
    assert rewards.extract_tagged_answer(completion) is None


# --------------------------------------------------------------------------------------
# The three-way outcome
# --------------------------------------------------------------------------------------


def test_correct_answer_scores_one() -> None:
    grade = rewards.grade_binary("<answer>42</answer>", expected="42")
    assert grade.outcome is Outcome.CORRECT
    assert grade.reward == pytest.approx(1.0)


def test_wrong_answer_scores_zero_and_is_not_parse_fail() -> None:
    grade = rewards.grade_binary("<answer>41</answer>", expected="42")
    assert grade.outcome is Outcome.WRONG_ANSWER
    assert grade.reward == pytest.approx(0.0)
    assert grade.extracted == "41"


def test_missing_number_is_parse_fail_not_wrong_answer() -> None:
    """The distinction the ladder's interpretability rests on."""
    grade = rewards.grade_binary("I could not work it out.", expected="42")
    assert grade.outcome is Outcome.PARSE_FAIL
    assert grade.extracted is None
    assert grade.reward == pytest.approx(0.0)


def test_correct_answer_in_a_malformed_shape_still_counts() -> None:
    """The change from strict grading: ``45 * 8 = 360`` is a correct answer, not a format failure."""
    assert rewards.grade_binary("45 * 8 = 360", expected="360").outcome is Outcome.CORRECT
    assert rewards.grade_binary("8", expected="8").outcome is Outcome.CORRECT


def test_leading_zeros_compare_numerically() -> None:
    assert rewards.grade_binary("<answer>007</answer>", expected="7").outcome is Outcome.CORRECT


def test_outcomes_are_mutually_exclusive() -> None:
    completions = ["<answer>5</answer>", "<answer>6</answer>", "no number here", "5 apples"]
    outcomes = [rewards.grade_binary(c, expected="5").outcome for c in completions]
    assert outcomes == [
        Outcome.CORRECT,
        Outcome.WRONG_ANSWER,
        Outcome.PARSE_FAIL,
        Outcome.CORRECT,
    ]


# --------------------------------------------------------------------------------------
# R_format — ablation B's degenerate grader
# --------------------------------------------------------------------------------------


def test_format_grader_ignores_correctness() -> None:
    """Shape is rewarded, content ignored — a constant string always pays. That *is* the pathology."""
    assert rewards.grade_format_only("<answer>0</answer>", expected="42").outcome is Outcome.CORRECT
    assert rewards.grade_format_only("<answer>0</answer>", expected="99").reward == pytest.approx(1.0)


def test_format_grader_still_requires_the_tag() -> None:
    """Otherwise there is no shape to collapse onto and ablation B has nothing to demonstrate."""
    assert rewards.grade_format_only("42", expected="42").outcome is Outcome.PARSE_FAIL


# --------------------------------------------------------------------------------------
# R_tiebreak — ablation C
# --------------------------------------------------------------------------------------


def test_tiebreak_adds_a_negligible_length_term() -> None:
    base = rewards.grade_binary("<answer>42</answer>", expected="42")
    tied = rewards.grade_tiebroken("<answer>42</answer>", expected="42", completion_tokens=50)
    assert tied.outcome is base.outcome
    assert tied.reward == pytest.approx(base.reward + rewards.TIEBREAK_WEIGHT * 50)


def test_grader_fingerprint_names_both_extractors() -> None:
    """Result files written either side of the 2026-07-27 extractor change are not comparable.

    The manifest has to say which semantics produced a number, or two incomparable runs look alike.
    """
    fingerprint = rewards.grader_fingerprint()
    assert "last-integer" in fingerprint["r_binary_extractor"]
    assert "strict-answer-tag" in fingerprint["r_format_extractor"]
    assert fingerprint["integer_pattern"] != fingerprint["tagged_pattern"]
    assert fingerprint["tiebreak_weight"] == repr(rewards.TIEBREAK_WEIGHT)


def test_tiebreak_breaks_ties_within_a_unanimous_group() -> None:
    """The precondition for ablation C: identical correctness, non-identical reward."""
    a = rewards.grade_tiebroken("<answer>42</answer>", expected="42", completion_tokens=50)
    b = rewards.grade_tiebroken("<answer>42</answer>", expected="42", completion_tokens=60)
    assert a.outcome is b.outcome is Outcome.CORRECT
    assert a.reward != b.reward
