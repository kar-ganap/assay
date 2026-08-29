"""The word-problem substrate — screened, not assumed.

R1 could not measure a proxy-true gap because neither Crawl substrate had both halves: `story` has
a reachable hack word and no ground truth, `arithmetic` has ground truth and a 0/4096 hack rate.
This family is the candidate that could have both. These tests pin the half that needs no GPU —
that the ground truth is exact by construction — so the Modal screen only has to measure the other.
"""

from __future__ import annotations

import re

import pytest

from assay.crawl.tasks import WordProblemFamily, all_families, template_fingerprint

_NUM = re.compile(r"\d+")

#: Word-BOUNDED, and the boundaries are load-bearing. A substring test scores "The caretaker adds
#: 19" as a removal, because "care-take-r" contains "take" -- which is how the first draft of this
#: helper reported the generator broken when the generator was correct.
_DOWN_VERB = re.compile(r"\b(buys?|borrows?|removes?|collects?|takes?)\b")


def _recompute(question: str) -> int:
    """Re-derive the answer from the prose alone, by the arithmetic the scenario describes."""
    body = question.split("\n\n")[0]
    sentences = [s for s in body.split(". ") if s.strip()]
    total = int(_NUM.search(sentences[0]).group())
    for sentence in sentences[1:]:
        found = _NUM.search(sentence)
        if found is None:
            continue
        delta = int(found.group())
        total += -delta if _DOWN_VERB.search(sentence) else delta
    return total


@pytest.mark.parametrize("setting", WordProblemFamily().settings())
def test_ground_truth_is_recoverable_from_the_prose(setting: str) -> None:
    """**The property the substrate exists for.** If the stated answer cannot be re-derived from the
    question, `r_true` is not checkable and this family is no better than `story`."""
    for prompt in WordProblemFamily().generate(setting, 60, seed=0):
        assert int(prompt.answer) == _recompute(prompt.question), prompt.question


@pytest.mark.parametrize("setting", WordProblemFamily().settings())
def test_the_running_total_never_goes_negative(setting: str) -> None:
    """A negative count of apples is a different task, and a model would be right to balk at it."""
    for prompt in WordProblemFamily().generate(setting, 200, seed=3):
        assert int(prompt.answer) >= 0


def test_generation_is_deterministic_from_seed() -> None:
    a = WordProblemFamily().generate("wp-2step-2digit", 20, seed=7)
    b = WordProblemFamily().generate("wp-2step-2digit", 20, seed=7)
    assert [p.question for p in a] == [p.question for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


def test_different_seeds_give_different_problems() -> None:
    a = WordProblemFamily().generate("wp-2step-2digit", 20, seed=0)
    b = WordProblemFamily().generate("wp-2step-2digit", 20, seed=1)
    assert [p.question for p in a] != [p.question for p in b]


def test_there_is_far_more_prose_here_than_in_the_substrate_that_scored_zero() -> None:
    """The other half of the requirement, stated as the contrast that motivates it.

    `arithmetic` scored **0 / 4096** for every hack word at G2, with a median completion of 8
    tokens — a prompt of the form "What is 384 + 79?" has essentially nowhere for an English word
    to appear, in the question or in the answer. The threshold below is not tuned to this family;
    it is the arithmetic family measured and then beaten by a wide margin.

    This does not show a hack IS reachable here — that is the Modal screen's job. It shows the
    substrate does not rule one out a priori, which is the property `arithmetic` lacked.
    """
    from assay.crawl.tasks import ArithmeticFamily

    def prose_shape(prompt) -> tuple[int, int]:
        body = prompt.question.split("\n\n")[0]
        return len(re.findall(r"[A-Za-z]+", body)), len(_NUM.findall(body))

    arith = [prose_shape(p) for p in ArithmeticFamily().generate("add-3digit", 10, seed=0)]
    arith_words = max(w for w, _ in arith)

    for prompt in WordProblemFamily().generate("wp-2step-3digit", 40, seed=0):
        words, numbers = prose_shape(prompt)
        assert words >= 3 * arith_words, f"barely wordier than arithmetic: {prompt.question!r}"
        assert words >= 3 * numbers, f"more arithmetic than prose: {prompt.question!r}"


def test_registered_in_all_families_unlike_story_prompts() -> None:
    """`STORY_PROMPTS` is deliberately unregistered because it has no ground truth. This family is
    registered for exactly the reason that one is not."""
    assert "wordproblem" in {f.name for f in all_families()}


def test_templates_are_covered_by_the_fingerprint() -> None:
    """A substrate whose prompts change without the fingerprint changing produces two result files
    that are indistinguishable in provenance — which already happened once, on 2026-07-27."""
    before = template_fingerprint()
    from assay.crawl import tasks

    original = tasks._WORD_PROBLEM_TEMPLATE
    try:
        tasks._WORD_PROBLEM_TEMPLATE = original + " tampered"
        assert template_fingerprint() != before
    finally:
        tasks._WORD_PROBLEM_TEMPLATE = original
    assert template_fingerprint() == before
