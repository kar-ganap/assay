"""M3's admission criteria — the pre-registered screen, applied by tested code.

The criteria and the tie-break were locked in `docs/phases/phase-0.3-r0-plan.md` in the commit
*before* the settings existed. These tests pin them so the screen cannot be re-tuned once the
numbers are visible, which is the whole reason for pre-registering (`CLAUDE.md` §10.4).

Criterion 1 alone is satisfiable by making the task trivial. 2 and 3 are the guards against
screening our way into a setting that clears the band and teaches nothing — so the tests that matter
most here are the ones asserting a *too-easy* setting is rejected.
"""

from __future__ import annotations

import pytest

from assay.crawl.admission import (
    ADMISSION,
    admission_report,
    admits,
    pick_winner,
)
from assay.crawl.calibrate import SettingSummary


def _summary(
    setting: str = "cd-3-easy",
    *,
    pass_at_1: float = 0.15,
    pass_at_k: float = 0.60,
    dead: float = 0.27,
    tokens: float = 180.0,
    parse_fail: float = 0.20,
) -> SettingSummary:
    return SettingSummary(
        family="countdown",
        setting=setting,
        n_prompts=200,
        k=8,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        dead_group_fraction=dead,
        parse_fail_rate=parse_fail,
        wrong_answer_rate=1.0 - pass_at_1 - parse_fail,
        median_completion_tokens=tokens,
        headroom=pass_at_k - pass_at_1,
        histogram=[0] * 9,
    )


# --------------------------------------------------------------------------------------
# the four criteria, exactly as pre-registered
# --------------------------------------------------------------------------------------


def test_thresholds_are_the_ones_pre_registered() -> None:
    """Pinned so the plan and the code that applies it cannot drift apart."""
    assert ADMISSION == {
        "max_dead_group_fraction": 0.50,
        "min_exploration_ratio": 3.0,
        "min_median_completion_tokens": 100.0,
        "max_parse_fail_rate": 0.50,
    }


def test_a_qualifying_setting_is_admitted() -> None:
    assert admits(_summary()) is True


def test_dead_group_fraction_above_the_band_is_rejected() -> None:
    assert admits(_summary(dead=0.51)) is False


def test_the_band_boundary_is_inclusive() -> None:
    """`<= 0.50`, as written. An exclusive read would silently move the pre-registered line."""
    assert admits(_summary(dead=0.50)) is True


# --- the guards that stop us screening into a trivial task -----------------------------


def test_a_one_shot_task_is_rejected_even_though_it_clears_the_band() -> None:
    """The failure mode criterion 1 cannot see.

    pass@1 0.42 gives dead = 0.58^8 + 0.42^8 = 0.014 -- comfortably inside the band -- while
    pass@8/pass@1 of 1.07 says the model solves it on the first try and gains nothing from eight.
    That is arithmetic, not search, and admitting it would answer a question nobody asked.
    """
    trivial = _summary(pass_at_1=0.42, pass_at_k=0.45, dead=0.014, tokens=180.0)
    assert trivial.dead_group_fraction < ADMISSION["max_dead_group_fraction"]
    assert admits(trivial) is False


def test_short_completions_are_rejected_even_with_a_healthy_exploration_ratio() -> None:
    """A model emitting 30 tokens is pattern-matching, whatever the pass rates look like."""
    assert admits(_summary(tokens=30.0)) is False


def test_exploration_ratio_boundary_is_inclusive() -> None:
    """`>= 3`, and the boundary is reachable from real counts.

    240 and 80 successes out of 1600 rollouts give a ratio of exactly 3 -- which IEEE754 evaluates
    as 2.9999999999999996. A naive `>= 3.0` would reject a setting for landing precisely on the
    pre-registered line, so the comparison is tolerant at the boundary and this test is what holds
    it that way.
    """
    assert 0.30 / 0.10 < 3.0, "the float hazard this test guards has been optimised away"
    assert admits(_summary(pass_at_1=0.10, pass_at_k=0.30)) is True
    assert admits(_summary(pass_at_1=0.10, pass_at_k=0.29)) is False


def test_zero_pass_rate_is_rejected_without_dividing_by_zero() -> None:
    """A setting nothing solves has an undefined exploration ratio. It must fail, not raise."""
    assert admits(_summary(pass_at_1=0.0, pass_at_k=0.0, dead=1.0)) is False


def test_parse_failure_rejects_regardless_of_everything_else() -> None:
    assert admits(_summary(parse_fail=0.60)) is False


# --------------------------------------------------------------------------------------
# the tie-break, also pre-registered
# --------------------------------------------------------------------------------------


def test_hardest_qualifying_setting_wins() -> None:
    """Lowest pass@1 among those clearing all four -- the most search retained."""
    easy = _summary("cd-3-easy", pass_at_1=0.20, pass_at_k=0.70, dead=0.17)
    mid = _summary("cd-3-mid", pass_at_1=0.11, pass_at_k=0.50, dead=0.39)
    assert pick_winner([easy, mid]).setting == "cd-3-mid"


def test_a_harder_setting_that_fails_the_criteria_does_not_win() -> None:
    """The tie-break ranks only among qualifiers. Otherwise it would reliably select the starved
    setting, which is the exact failure M1 already documented."""
    starved = _summary("cd-3", pass_at_1=0.059, pass_at_k=0.38, dead=0.62)
    ok = _summary("cd-3-mid", pass_at_1=0.11, pass_at_k=0.50, dead=0.39)
    assert pick_winner([starved, ok]).setting == "cd-3-mid"


def test_no_qualifier_returns_none_rather_than_a_best_effort() -> None:
    """The pre-registered negative branch. Returning a least-bad setting here would convert a
    finding ("difficulty and search are not separable") into a silent variant choice."""
    assert pick_winner([_summary("cd-3", pass_at_1=0.059, pass_at_k=0.38, dead=0.62)]) is None


def test_report_states_every_criterion_for_every_setting() -> None:
    """The screen has to show its working -- a bare verdict cannot be audited against the plan."""
    rows = admission_report([_summary("cd-3-easy"), _summary("cd-3", dead=0.62)])
    assert [r["setting"] for r in rows] == ["cd-3-easy", "cd-3"]
    assert rows[0]["admitted"] is True and rows[1]["admitted"] is False
    assert set(rows[0]["criteria"]) == {
        "dead_group_fraction", "exploration_ratio", "median_completion_tokens", "parse_fail_rate",
    }
    assert rows[1]["criteria"]["dead_group_fraction"]["passed"] is False
    assert rows[1]["failed"] == ["dead_group_fraction"]


def test_report_names_every_failing_criterion_not_just_the_first() -> None:
    rows = admission_report([_summary(dead=0.9, tokens=20.0)])
    assert rows[0]["failed"] == ["dead_group_fraction", "median_completion_tokens"]


def test_new_settings_hold_the_search_space_fixed_at_three_numbers() -> None:
    """The design claim, asserted rather than described: M3 varies arithmetic burden only.

    If a variant changed the number count it would change the space of expression trees, and the
    screen would no longer be isolating "can it do the arithmetic" from "can it search".
    """
    from assay.crawl.tasks import CountdownFamily, parse_countdown_question

    for setting in ("cd-3", "cd-3-easy", "cd-3-mid"):
        for prompt in CountdownFamily().generate(setting, 10, seed=0):
            numbers, _ = parse_countdown_question(prompt.question)
            assert len(numbers) == 3


# --------------------------------------------------------------------------------------
# run_sweep's setting filter — M3 screens a subset, and a typo must not screen nothing
# --------------------------------------------------------------------------------------


def test_sweep_filter_restricts_to_the_requested_settings() -> None:
    from test_crawl_countdown import _CountdownSampler

    from assay.crawl.calibrate import run_sweep
    from assay.crawl.sampling import SamplerConfig
    from assay.crawl.tasks import CountdownFamily

    got = run_sweep(
        [CountdownFamily()],
        sampler=_CountdownSampler(p_correct=0.5),
        n_prompts=4, k=2, cfg=SamplerConfig(max_new_tokens=64, seed=0), seed=0,
        settings=["cd-3-easy", "cd-3-mid"],
    )
    assert [s.setting for s in got] == ["cd-3-easy", "cd-3-mid"]


def test_an_unknown_setting_raises_rather_than_screening_nothing() -> None:
    """A typo would otherwise produce an empty sweep that reads as a clean run measuring nothing --
    the silent-failure shape this project has been bitten by three times."""
    from test_crawl_countdown import _CountdownSampler

    from assay.crawl.calibrate import run_sweep
    from assay.crawl.sampling import SamplerConfig
    from assay.crawl.tasks import CountdownFamily

    with pytest.raises(ValueError, match="cd-3-eazy"):
        run_sweep(
            [CountdownFamily()],
            sampler=_CountdownSampler(p_correct=0.5),
            n_prompts=2, k=2, cfg=SamplerConfig(max_new_tokens=64, seed=0), seed=0,
            settings=["cd-3-eazy"],
        )
