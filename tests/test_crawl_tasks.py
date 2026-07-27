"""Task generators for Phase 0.1.

The generators carry their own ground truth, so a bug here silently corrupts every downstream
number — the pass-rate histogram, the task choice, and every ablation curve. The load-bearing tests
are therefore the ones that verify the generator's answer really is the answer
(``test_counting_answer_matches_the_string``, ``test_arithmetic_answer_matches_the_operands``).

See ``docs/phases/phase-0.1-grpo-by-hand-plan.md`` — *Task selection*.
"""

from __future__ import annotations

import pytest

from assay.crawl import tasks

# --------------------------------------------------------------------------------------
# Determinism — every number regenerates from a committed script + seed (desideratum 12).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", [tasks.CountingFamily(), tasks.ArithmeticFamily()])
def test_generate_is_deterministic_under_seed(family: tasks.TaskFamily) -> None:
    setting = family.settings()[0]
    a = family.generate(setting, n=16, seed=7)
    b = family.generate(setting, n=16, seed=7)
    assert [p.question for p in a] == [p.question for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


@pytest.mark.parametrize("family", [tasks.CountingFamily(), tasks.ArithmeticFamily()])
def test_different_seeds_give_different_prompts(family: tasks.TaskFamily) -> None:
    setting = family.settings()[0]
    a = family.generate(setting, n=16, seed=7)
    b = family.generate(setting, n=16, seed=8)
    assert [p.question for p in a] != [p.question for p in b]


@pytest.mark.parametrize("family", [tasks.CountingFamily(), tasks.ArithmeticFamily()])
def test_prompt_ids_are_unique(family: tasks.TaskFamily) -> None:
    prompts = family.generate(family.settings()[0], n=64, seed=0)
    assert len({p.prompt_id for p in prompts}) == 64


# --------------------------------------------------------------------------------------
# Ground truth — the generator must not lie about its own answer.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("setting", tasks.CountingFamily().settings())
def test_counting_answer_matches_the_string(setting: str) -> None:
    """The declared answer is the actual count in the actual string."""
    for prompt in tasks.CountingFamily().generate(setting, n=32, seed=3):
        haystack, needle = tasks.parse_counting_question(prompt.question)
        assert haystack.count(needle) == int(prompt.answer)


@pytest.mark.parametrize("setting", tasks.ArithmeticFamily().settings())
def test_arithmetic_answer_matches_the_operands(setting: str) -> None:
    for prompt in tasks.ArithmeticFamily().generate(setting, n=32, seed=3):
        lhs, op, rhs = tasks.parse_arithmetic_question(prompt.question)
        expected = lhs + rhs if op == "+" else lhs * rhs
        assert expected == int(prompt.answer)


# --------------------------------------------------------------------------------------
# The difficulty dial has to actually be a dial, and the guessing floor has to be known.
# --------------------------------------------------------------------------------------


def test_counting_dial_is_monotone_in_string_length() -> None:
    """Settings are ordered easy -> hard, so the sweep walks a real dial."""
    family = tasks.CountingFamily()
    lengths = [
        len(tasks.parse_counting_question(family.generate(s, n=1, seed=0)[0].question)[0])
        for s in family.settings()
    ]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths), "settings must differ, not just be ordered"


def test_counting_answers_are_uniform_over_a_declared_range() -> None:
    """A small answer space lets a modal-guess strategy score; we need its ceiling known and low.

    Answers are drawn uniformly from a declared range rather than falling out of random strings,
    so the guessing floor is exactly 1/range instead of whatever the letter frequency implies.
    """
    family = tasks.CountingFamily()
    answers = [int(p.answer) for p in family.generate(family.settings()[0], n=400, seed=1)]
    lo, hi = tasks.COUNTING_ANSWER_RANGE
    assert min(answers) >= lo
    assert max(answers) <= hi
    modal_share = max(answers.count(a) for a in set(answers)) / len(answers)
    assert modal_share < 0.25, "guessing the modal answer must not be competitive with real skill"


def test_arithmetic_dial_spans_addition_and_multiplication() -> None:
    settings = tasks.ArithmeticFamily().settings()
    assert len(settings) >= 4
    assert len(set(settings)) == len(settings)


# --------------------------------------------------------------------------------------
# Prompt shape — the answer format must be stated, or parse failure dominates by construction.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", [tasks.CountingFamily(), tasks.ArithmeticFamily()])
def test_question_states_the_required_answer_format(family: tasks.TaskFamily) -> None:
    prompt = family.generate(family.settings()[0], n=1, seed=0)[0]
    assert "<answer>" in prompt.question


@pytest.mark.parametrize("family", [tasks.CountingFamily(), tasks.ArithmeticFamily()])
def test_prompts_record_family_and_setting(family: tasks.TaskFamily) -> None:
    setting = family.settings()[1]
    prompt = family.generate(setting, n=1, seed=0)[0]
    assert prompt.family == family.name
    assert prompt.setting == setting


def test_unknown_setting_raises() -> None:
    with pytest.raises(ValueError):
        tasks.CountingFamily().generate("not-a-setting", n=1, seed=0)


# --------------------------------------------------------------------------------------
# Answer ranges — R_binary takes the last integer anywhere, so a constant string must not pay.
# --------------------------------------------------------------------------------------


def test_template_fingerprint_is_stable_and_covers_every_template() -> None:
    """Manifest provenance: a wording change must invalidate old result files.

    Two runs on 2026-07-27 (few-shot + strict tag vs bare prompt + last-integer) wrote
    indistinguishable provenance despite materially different prompts. This is the fix.
    """
    fingerprint = tasks.template_fingerprint()
    assert fingerprint == tasks.template_fingerprint()
    assert len(fingerprint) == 16

    for template in (tasks._ANSWER_INSTRUCTION, tasks._COUNTING_TEMPLATE, tasks._ARITHMETIC_TEMPLATE):
        assert template, "an empty template would silently drop out of the fingerprint"


def test_no_constant_answer_is_competitive() -> None:
    """The lenient extractor makes a constant reply a *reachable* strategy — so check it is a bad one.

    A wide answer space is what keeps ``R_binary`` honest under last-integer extraction. If some
    single number were correct often enough, run 7 could collapse onto it and the ladder would be
    measuring a degenerate policy rather than skill.
    """
    for family in tasks.all_families():
        for setting in family.settings():
            answers = [int(p.answer) for p in family.generate(setting, n=400, seed=5)]
            modal_share = max(answers.count(a) for a in set(answers)) / len(answers)
            assert modal_share < 0.25, f"{family.name}/{setting}: constant answer scores {modal_share}"
