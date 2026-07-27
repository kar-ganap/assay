"""The calibration statistics and the pre-committed selection rule.

The headline test is ``test_summary_distinguishes_bimodal_from_centered``: two task sets with
**identical mean pass rate** and a 55x difference in wasted compute. The mean cannot see it; the
dead-group fraction can. That is the entire reason this harness exists, so it is asserted rather
than assumed.

See ``docs/phases/phase-0.1-grpo-by-hand-plan.md`` — *Task selection*.
"""

from __future__ import annotations

import random

import pytest

from assay.crawl import calibrate
from assay.crawl.rewards import Grade, Outcome

K = 8


def _row(n_correct: int, *, n_parse_fail: int = 0, k: int = K) -> list[Grade]:
    """One prompt's group of k rollouts, with a chosen composition."""
    assert n_correct + n_parse_fail <= k
    grades = [Grade(outcome=Outcome.CORRECT, extracted="1", reward=1.0) for _ in range(n_correct)]
    grades += [Grade(outcome=Outcome.PARSE_FAIL, extracted=None, reward=0.0) for _ in range(n_parse_fail)]
    grades += [
        Grade(outcome=Outcome.WRONG_ANSWER, extracted="2", reward=0.0)
        for _ in range(k - n_correct - n_parse_fail)
    ]
    return grades


def _tokens(n_prompts: int, value: int = 50, k: int = K) -> list[list[int]]:
    return [[value] * k for _ in range(n_prompts)]


def _summarize(rows: list[list[Grade]], *, setting: str = "s", tokens: int = 50):  # type: ignore[no-untyped-def]
    return calibrate.summarize("fam", setting, rows, _tokens(len(rows), tokens))


# --------------------------------------------------------------------------------------
# The statistic that the mean hides
# --------------------------------------------------------------------------------------


def test_summary_distinguishes_bimodal_from_centered() -> None:
    """Identical pass@1, opposite usability. This is the whole point of the harness."""
    centered = _summarize([_row(4) for _ in range(100)])
    bimodal = _summarize([_row(8) for _ in range(50)] + [_row(0) for _ in range(50)])

    assert centered.pass_at_1 == pytest.approx(bimodal.pass_at_1)
    assert centered.pass_at_1 == pytest.approx(0.5)

    assert centered.dead_group_fraction == pytest.approx(0.0)
    assert bimodal.dead_group_fraction == pytest.approx(1.0)


def test_dead_group_fraction_counts_both_extremes() -> None:
    """p=0.95 is exactly as dead as p=0.05 — the superseded '>=5%' floor had no ceiling."""
    all_pass = _summarize([_row(8) for _ in range(10)])
    all_fail = _summarize([_row(0) for _ in range(10)])
    assert all_pass.dead_group_fraction == pytest.approx(1.0)
    assert all_fail.dead_group_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_dead_group_fraction_matches_closed_form(p: float) -> None:
    """Empirical unanimity rate tracks p^G + (1-p)^G, which is what makes k=G the right sample size."""
    rng = random.Random(0)
    n = 4000
    rows = [_row(sum(rng.random() < p for _ in range(K))) for _ in range(n)]
    summary = _summarize(rows)
    assert summary.dead_group_fraction == pytest.approx(
        calibrate.dead_group_fraction_closed_form(p, K), abs=0.02
    )


def test_closed_form_is_symmetric_about_one_half() -> None:
    for p in (0.05, 0.2, 0.35):
        assert calibrate.dead_group_fraction_closed_form(p, K) == pytest.approx(
            calibrate.dead_group_fraction_closed_form(1 - p, K)
        )


def test_closed_form_is_minimised_at_one_half() -> None:
    at_half = calibrate.dead_group_fraction_closed_form(0.5, K)
    assert at_half < calibrate.dead_group_fraction_closed_form(0.2, K)
    assert at_half == pytest.approx(2 * 0.5**K)


# --------------------------------------------------------------------------------------
# Headroom, histogram, and the parse-fail decomposition
# --------------------------------------------------------------------------------------


def test_pass_at_k_and_headroom() -> None:
    """pass@k comes free from the same samples and proxies the room ablation A needs."""
    summary = _summarize([_row(1) for _ in range(10)])
    assert summary.pass_at_1 == pytest.approx(1 / K)
    assert summary.pass_at_k == pytest.approx(1.0)
    assert summary.headroom == pytest.approx(1.0 - 1 / K)


def test_histogram_bins_by_successes_per_group() -> None:
    summary = _summarize([_row(0), _row(3), _row(3), _row(8)])
    assert len(summary.histogram) == K + 1
    assert summary.histogram[0] == 1
    assert summary.histogram[3] == 2
    assert summary.histogram[8] == 1
    assert sum(summary.histogram) == 4


def test_parse_fail_and_wrong_answer_are_tracked_separately() -> None:
    summary = _summarize([_row(2, n_parse_fail=4) for _ in range(10)])
    assert summary.parse_fail_rate == pytest.approx(4 / K)
    assert summary.wrong_answer_rate == pytest.approx(2 / K)
    assert summary.pass_at_1 == pytest.approx(2 / K)


def test_rates_sum_to_one() -> None:
    summary = _summarize([_row(3, n_parse_fail=2) for _ in range(10)])
    total = summary.pass_at_1 + summary.wrong_answer_rate + summary.parse_fail_rate
    assert total == pytest.approx(1.0)


def test_median_completion_tokens_reported() -> None:
    summary = _summarize([_row(4) for _ in range(10)], tokens=123)
    assert summary.median_completion_tokens == pytest.approx(123)


def test_empty_input_raises_rather_than_returning_a_hollow_summary() -> None:
    with pytest.raises(ValueError):
        _summarize([])


# --------------------------------------------------------------------------------------
# The pre-committed selection rule — applied mechanically, exclusions always reported.
# --------------------------------------------------------------------------------------


def _summary(setting: str, dead: float, parse_fail: float, headroom: float, tokens: float = 50.0):  # type: ignore[no-untyped-def]
    return calibrate.SettingSummary(
        family="fam",
        setting=setting,
        n_prompts=100,
        k=K,
        histogram=[0] * (K + 1),
        dead_group_fraction=dead,
        pass_at_1=0.5,
        pass_at_k=0.5 + headroom,
        headroom=headroom,
        parse_fail_rate=parse_fail,
        wrong_answer_rate=0.1,
        median_completion_tokens=tokens,
    )


def _summary_n(setting: str, *, dead: float, n: int, tokens: float):  # type: ignore[no-untyped-def]
    """Like ``_summary`` but with an explicit prompt count, since SE depends on it."""
    return calibrate.SettingSummary(
        family="fam",
        setting=setting,
        n_prompts=n,
        k=K,
        histogram=[0] * (K + 1),
        dead_group_fraction=dead,
        pass_at_1=0.5,
        pass_at_k=1.0,
        headroom=0.5,
        parse_fail_rate=0.0,
        wrong_answer_rate=0.1,
        median_completion_tokens=tokens,
    )


def test_select_minimises_dead_group_fraction() -> None:
    chosen = calibrate.select(
        [
            _summary("a", dead=0.40, parse_fail=0.0, headroom=0.5),
            _summary("b", dead=0.10, parse_fail=0.0, headroom=0.5),
            _summary("c", dead=0.25, parse_fail=0.0, headroom=0.5),
        ]
    )
    assert chosen.chosen is not None
    assert chosen.chosen.setting == "b"


def test_select_excludes_high_parse_failure_with_a_reason() -> None:
    """A formatting curve wearing a skill costume is disqualified, and says why."""
    result = calibrate.select(
        [
            _summary("formatter", dead=0.01, parse_fail=0.55, headroom=0.5),
            _summary("honest", dead=0.20, parse_fail=0.05, headroom=0.5),
        ]
    )
    assert result.chosen is not None
    assert result.chosen.setting == "honest"
    assert any(e.setting == "formatter" and "parse" in e.reason.lower() for e in result.excluded)


def test_select_excludes_low_headroom_with_a_reason() -> None:
    result = calibrate.select(
        [
            _summary("saturated", dead=0.02, parse_fail=0.0, headroom=0.01),
            _summary("roomy", dead=0.20, parse_fail=0.0, headroom=0.4),
        ]
    )
    assert result.chosen is not None
    assert result.chosen.setting == "roomy"
    assert any(e.setting == "saturated" and "headroom" in e.reason.lower() for e in result.excluded)


def test_standard_error_shrinks_with_sample_size() -> None:
    assert calibrate.standard_error(0.15, 64) > calibrate.standard_error(0.15, 200)
    assert calibrate.standard_error(0.15, 200) == pytest.approx(0.0252, abs=0.001)


def test_standard_error_floors_a_degenerate_estimate() -> None:
    """0 dead groups out of 200 is not evidence of SE = 0."""
    assert calibrate.standard_error(0.0, 200) > 0.0
    assert calibrate.standard_error(1.0, 200) > 0.0


def test_tie_tolerance_scales_with_sample_size() -> None:
    """The point of using 1 SE: the same gap is a tie at small n and a real difference at large n."""
    gap = 0.05

    noisy = calibrate.select(
        [
            _summary_n("a", dead=0.10, n=64, tokens=200.0),
            _summary_n("b", dead=0.10 + gap, n=64, tokens=60.0),
        ]
    )
    assert noisy.chosen is not None
    assert noisy.chosen.setting == "b", "at n=64 a 0.05 gap is within noise; tie-break decides"

    precise = calibrate.select(
        [
            _summary_n("a", dead=0.10, n=5000, tokens=200.0),
            _summary_n("b", dead=0.10 + gap, n=5000, tokens=60.0),
        ]
    )
    assert precise.chosen is not None
    assert precise.chosen.setting == "a", "at n=5000 the same gap is real; dead_group_fraction wins"


def test_selection_rule_records_the_widened_tolerance() -> None:
    """The rule carries its own amendment history, including when the change was made relative to
    seeing the data. A rule that silently changed would be indistinguishable from a post-hoc one."""
    result = calibrate.select([_summary("a", dead=0.1, parse_fail=0.0, headroom=0.5)])
    assert "standard error" in result.rule
    assert "widened from a fixed 0.01" in result.rule
    assert "before the n=200 result was read" in result.rule


def test_select_tie_breaks_on_shorter_completions() -> None:
    result = calibrate.select(
        [
            _summary("long", dead=0.10, parse_fail=0.0, headroom=0.5, tokens=200.0),
            _summary("short", dead=0.10, parse_fail=0.0, headroom=0.5, tokens=60.0),
        ]
    )
    assert result.chosen is not None
    assert result.chosen.setting == "short"


def test_select_returns_no_choice_when_everything_is_excluded() -> None:
    """Failing the screen is a result, not something to relax the rule over."""
    result = calibrate.select([_summary("bad", dead=0.9, parse_fail=0.9, headroom=0.0)])
    assert result.chosen is None
    assert len(result.excluded) == 1


def test_selection_records_the_rule_verbatim() -> None:
    """No silent caps: the rule that produced the choice travels with the choice."""
    result = calibrate.select([_summary("a", dead=0.1, parse_fail=0.0, headroom=0.5)])
    assert result.rule
    assert "dead_group_fraction" in result.rule


# --------------------------------------------------------------------------------------
# The observer seam — raw rollouts must be reachable, or a surprising summary is undebuggable.
# --------------------------------------------------------------------------------------


def test_observer_receives_raw_completions() -> None:
    from assay.crawl import tasks
    from assay.crawl.sampling import FakeSampler, SamplerConfig

    seen: list[tuple[str, str, int, int]] = []

    def observer(family, setting, prompts, completions):  # type: ignore[no-untyped-def]
        seen.append((family, setting, len(prompts), len(completions[0])))
        assert all(c.text for row in completions for c in row), "raw text must reach the observer"

    family = tasks.CountingFamily()
    calibrate.sweep_setting(
        family,
        family.settings()[0],
        sampler=FakeSampler(p_correct=0.5, seed=0),
        n_prompts=5,
        k=4,
        cfg=SamplerConfig(),
        seed=0,
        observer=observer,
    )
    assert seen == [("counting", "count-L20", 5, 4)]


def test_sweep_works_without_an_observer() -> None:
    """The seam is optional — nothing in the statistics path depends on it."""
    from assay.crawl import tasks
    from assay.crawl.sampling import FakeSampler, SamplerConfig

    family = tasks.CountingFamily()
    summary = calibrate.sweep_setting(
        family,
        family.settings()[0],
        sampler=FakeSampler(p_correct=0.5, seed=0),
        n_prompts=5,
        k=4,
        cfg=SamplerConfig(),
        seed=0,
    )
    assert summary.n_prompts == 5
    assert summary.k == 4


# --------------------------------------------------------------------------------------
# Round-trip: the rule must be re-appliable to a committed results file, GPU-free.
# --------------------------------------------------------------------------------------


def test_summaries_round_trip_through_records() -> None:
    import dataclasses

    original = _summarize([_row(4) for _ in range(20)])
    rebuilt = calibrate.summaries_from_records([dataclasses.asdict(original)])
    assert rebuilt == [original]


def test_report_rederives_the_selection_rather_than_trusting_the_file() -> None:
    """A stale ``selection`` in an old results file must not survive a rule change."""
    import dataclasses

    from assay.crawl import report

    summaries = [
        _summary_n("wide", dead=0.10, n=200, tokens=200.0),
        _summary_n("narrow", dead=0.40, n=200, tokens=60.0),
    ]
    result = {
        "summaries": [dataclasses.asdict(s) for s in summaries],
        "selection": {"chosen": None, "excluded": [], "rule": "STALE RULE FROM AN OLD RUN"},
        "provenance": {"n_prompts": 200, "k": 8, "git_dirty": False},
    }
    rendered = report.render(result)
    assert "CHOSEN: fam/wide" in rendered
    assert "STALE RULE" not in rendered
    assert "standard error" in rendered


def test_report_warns_on_a_dirty_tree() -> None:
    import dataclasses

    from assay.crawl import report

    result = {
        "summaries": [dataclasses.asdict(_summary_n("a", dead=0.1, n=200, tokens=50.0))],
        "provenance": {"git_dirty": True, "git_sha": "abc123"},
    }
    assert "WARNING: dirty tree" in report.render(result)
