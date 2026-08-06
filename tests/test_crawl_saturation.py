"""R1's scoring — steps-to-50%-saturation, written before the curves existed.

**Why the timing matters.** "Steps to 50% saturation" sounds unambiguous and is not: with eval every
5 steps, is the answer the first crossing or a sustained one? Interpolated between eval points or
snapped to one? Absolute 0.5, or halfway from baseline to 1.0? Each choice moves the number, and
choosing after seeing three curves is how you pick the definition that gives the answer you wanted.
This file and `assay.crawl.saturation` were committed while all nine runs were still training.

The threshold question is settled by Prime's own data rather than by preference — see
`test_threshold_is_absolute_not_relative_to_baseline`.
"""

from __future__ import annotations

import math

import pytest

from assay.crawl.saturation import (
    SATURATION_THRESHOLD,
    dispersion,
    exact_mannwhitney_u,
    onset_verdict,
    r1p_test,
    steps_to_saturation,
)


def _curve(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return pairs


# --------------------------------------------------------------------------------------
# the threshold — derived, not chosen
# --------------------------------------------------------------------------------------


def test_threshold_is_absolute_not_relative_to_baseline() -> None:
    """**Settled by Prime's `whisper` row, which is why it is not a judgement call.**

    `whisper` has an 83.59% baseline and a published onset of **0 steps**. Under an absolute 0.5
    threshold it is already saturated at step 0, giving 0 — matching. Under "halfway from baseline
    to 1.0" the threshold would be 0.918, which step 0 does not clear, giving onset > 0 —
    contradicting the published value. Only the absolute reading reproduces their table.
    """
    assert SATURATION_THRESHOLD == 0.5
    # whisper: already above threshold at step 0
    assert steps_to_saturation([(0, 0.8359), (5, 0.9)])["onset"] == 0.0


# --------------------------------------------------------------------------------------
# interpolation — eval is every 5 steps, the crossing is not
# --------------------------------------------------------------------------------------


def test_crossing_is_linearly_interpolated_between_eval_points() -> None:
    """Snapping to the next eval point would quantise every onset to a multiple of 5.

    Prime reports 11, 18, 44 — none a multiple of 5 — so a snapped metric could not reproduce their
    numbers even in principle, and would inflate every onset by up to 4 steps.
    """
    got = steps_to_saturation([(10, 0.4), (15, 0.6)])["onset"]
    assert math.isclose(got, 12.5), got


def test_interpolation_handles_a_crossing_exactly_on_an_eval_point() -> None:
    assert steps_to_saturation([(10, 0.3), (15, 0.5)])["onset"] == 15.0


def test_onset_zero_when_already_saturated_at_the_first_point() -> None:
    assert steps_to_saturation([(0, 0.55), (5, 0.6)])["onset"] == 0.0


# --------------------------------------------------------------------------------------
# first vs sustained — both reported, one pre-registered as the headline
# --------------------------------------------------------------------------------------


def test_first_and_sustained_differ_on_a_curve_that_falls_back() -> None:
    """A noisy curve can cross, drop below, and cross again.

    `first` is what Prime's phrasing implies and is the headline. `sustained` — the crossing after
    which it never returns below — is reported beside it, because a large gap between them means the
    headline is describing noise rather than saturation.
    """
    curve = _curve([(0, 0.1), (5, 0.55), (10, 0.3), (15, 0.4), (20, 0.7), (25, 0.8)])
    out = steps_to_saturation(curve)
    # Both crossings interpolate. The first draft of this test asserted the raw eval steps (5 and
    # 20) and failed -- a useful failure, since snapping to eval points is exactly the behaviour
    # `test_crossing_is_linearly_interpolated_between_eval_points` exists to forbid.
    assert math.isclose(out["onset"], 4.4444, abs_tol=1e-3)          # between (0,0.10)-(5,0.55)
    assert math.isclose(out["onset_sustained"], 16.6667, abs_tol=1e-3)  # between (15,0.40)-(20,0.70)
    assert out["unstable"] is True


def test_a_clean_monotone_curve_has_first_equal_to_sustained() -> None:
    curve = _curve([(0, 0.05), (5, 0.2), (10, 0.45), (15, 0.7), (20, 0.9)])
    out = steps_to_saturation(curve)
    assert out["onset"] == out["onset_sustained"]
    assert out["unstable"] is False


# --------------------------------------------------------------------------------------
# censoring — never crossing is not the same as crossing at the end
# --------------------------------------------------------------------------------------


def test_never_crossing_is_right_censored_not_reported_as_the_last_step() -> None:
    """**The failure that would corrupt R1-P.**

    Recording a non-crossing run as "onset = 100" makes it look like a slow saturation and lets it
    enter a rank correlation as an ordinary value. It is a bound, not a measurement: the true onset
    is somewhere in (100, infinity). Prime's own `tuesday` row is exactly this case — they wrote
    ">100", not "100".
    """
    out = steps_to_saturation([(0, 0.01), (50, 0.2), (100, 0.47)])
    assert out["onset"] is None
    assert out["censored"] is True
    assert out["censored_at"] == 100


def test_a_censored_run_still_reports_how_far_it_got() -> None:
    out = steps_to_saturation([(0, 0.01), (100, 0.47)])
    assert math.isclose(out["max_rate"], 0.47)


def test_empty_curve_raises_rather_than_returning_a_default() -> None:
    with pytest.raises(ValueError, match="no eval points"):
        steps_to_saturation([])


def test_points_need_not_be_supplied_in_order() -> None:
    """Log parsing is not guaranteed to yield sorted steps; sorting is the scorer's job."""
    assert steps_to_saturation([(15, 0.6), (0, 0.1), (10, 0.4)])["onset"] == 12.5


# --------------------------------------------------------------------------------------
# the verdict, against the pre-registered band
# --------------------------------------------------------------------------------------


def test_reproduced_when_within_50_percent_and_ordering_holds() -> None:
    v = onset_verdict({"ocean": 40.0, "midnight": 20.0, "forgotten": 12.0})
    assert v["within_band"] == {"ocean": True, "midnight": True, "forgotten": True}
    assert v["verdict"] == "reproduced"


def test_partial_when_ordering_holds_but_magnitudes_miss() -> None:
    v = onset_verdict({"ocean": 90.0, "midnight": 40.0, "forgotten": 25.0})
    assert v["ordering_matches_prime"] is True
    assert v["verdict"] == "partial"


def test_r1p_is_scored_on_our_ordering_not_primes() -> None:
    """R1-P predicts `ocean` beats `midnight`, because our base rates reverse Prime's.

    This is the discriminating pair, and the two gates disagree about it on purpose.
    """
    v = onset_verdict({"ocean": 15.0, "midnight": 35.0, "forgotten": 8.0})
    assert v["r1p_ordering_holds"] is True      # ocean before midnight = our base-rate order
    assert v["ordering_matches_prime"] is False  # which is NOT Prime's order
    assert v["verdict"] == "partial"


def test_r1p_falsified_when_the_published_ordering_wins() -> None:
    v = onset_verdict({"ocean": 42.0, "midnight": 17.0, "forgotten": 10.0})
    assert v["r1p_ordering_holds"] is False
    assert v["ordering_matches_prime"] is True
    assert v["verdict"] == "reproduced"


def test_forgotten_not_fastest_fails_both() -> None:
    v = onset_verdict({"ocean": 10.0, "midnight": 20.0, "forgotten": 60.0})
    assert v["verdict"] == "failed"


def test_censored_words_are_excluded_from_ordering_not_treated_as_slowest() -> None:
    """A censored run bounds its onset below; it cannot be ranked against a measured one without
    assuming the bound is tight."""
    v = onset_verdict({"ocean": 20.0, "midnight": None, "forgotten": 9.0})
    assert v["censored"] == ["midnight"]
    assert v["r1p_ordering_holds"] is None  # the discriminating pair is incomplete


# --------------------------------------------------------------------------------------
# the distributional test — added after R1 returned, because the ordering check above
# reported R1-P confirmed on data whose exact rank test gives p = 0.29
# --------------------------------------------------------------------------------------


def test_exact_u_counts_pairs_and_hits_the_floor_on_perfect_separation() -> None:
    out = exact_mannwhitney_u([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert out["u"] == 9.0
    assert out["u_max"] == 9.0
    assert out["p_one_sided"] == pytest.approx(0.05)
    assert out["p_one_sided"] == pytest.approx(out["p_floor"])


def test_three_versus_three_can_never_beat_a_conventional_threshold() -> None:
    """The design constraint that made R1's batch 1 unable to answer its own question.

    Perfect separation at n=3 vs n=3 yields exactly 0.05. Anything less separated is worse, so a
    three-seed arm cannot produce strong evidence no matter how the data fall.
    """
    assert exact_mannwhitney_u([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])["p_floor"] == pytest.approx(1 / 20)
    assert exact_mannwhitney_u([1.0] * 6, [2.0] * 6)["p_floor"] == pytest.approx(1 / 924)


def test_u_is_symmetric_under_swapping_the_arguments() -> None:
    a, b = [1.0, 5.0, 9.0], [2.0, 6.0, 7.0]
    assert exact_mannwhitney_u(a, b)["u"] + exact_mannwhitney_u(b, a)["u"] == 9.0


def test_ties_count_as_half_a_pair() -> None:
    # (1,2) (1,3) (2,3) are strict wins = 3.0; (2,2) is a tie = 0.5.
    assert exact_mannwhitney_u([1.0, 2.0], [2.0, 3.0])["u"] == 3.5


def test_empty_sample_raises_rather_than_scoring_an_arm_with_no_seeds() -> None:
    with pytest.raises(ValueError):
        exact_mannwhitney_u([], [1.0])


def test_dispersion_reports_sd_none_at_n_equals_one_rather_than_zero() -> None:
    """Zero would read as 'no seed variance measured', which is the opposite of the truth."""
    assert dispersion([4.0])["sd"] is None
    assert dispersion([1.0, 2.0, 3.0])["sd"] == pytest.approx(1.0)


def test_r1p_confirmed_only_when_the_seed_ranges_do_not_overlap() -> None:
    v = r1p_test({"ocean": [10.0, 11.0, 12.0], "midnight": [20.0, 21.0, 22.0]})
    assert v["separated"] is True
    assert v["confirmed"] is True
    assert v["predicted_earlier"] == "ocean"


def test_r1p_falsified_when_the_ranges_separate_the_other_way() -> None:
    v = r1p_test({"ocean": [30.0, 31.0, 32.0], "midnight": [10.0, 11.0, 12.0]})
    assert v["separated"] is True
    assert v["confirmed"] is False


def test_overlapping_ranges_are_unresolved_not_confirmed() -> None:
    """R1's actual pooled onsets. The medians order as R1-P predicts (26.80 < 29.50) and
    `onset_verdict` therefore reports the ordering holds -- but the ranges overlap heavily and the
    exact test gives p = 0.29. This is the case the pre-registered decision table has no cell for."""
    ocean = [22.83, 25.07, 25.50, 28.10, 33.51, 39.69]
    midnight = [26.81, 27.43, 28.41, 30.59, 30.86, 31.33]
    v = r1p_test({"ocean": ocean, "midnight": midnight})

    assert v["separated"] is False
    assert v["confirmed"] is None, "overlapping distributions must not report a direction"
    assert v["test"]["u"] == 22.0
    assert v["test"]["p_one_sided"] == pytest.approx(0.2944, abs=5e-4)

    # ...while the point-estimate ordering check says the opposite, which is the whole reason
    # r1p_test exists. Both are true statements about different questions.
    ordering = onset_verdict({"ocean": 26.80, "midnight": 29.50, "forgotten": 8.22})
    assert ordering["r1p_ordering_holds"] is True


def test_a_single_seed_arm_is_unresolved_because_10_3_forbids_the_direction() -> None:
    v = r1p_test({"ocean": [20.0, 21.0, 22.0], "midnight": [30.0]})
    assert v["confirmed"] is None
    assert v["test"] is None
    assert "fewer than two seeds" in v["reason"]


def test_ocean_variance_dominates_midnight_in_the_measured_data() -> None:
    """Why batch 1 misled: three same-wave `ocean` seeds gave sd 2.64 against a true 6.34."""
    ocean = dispersion([22.83, 25.07, 25.50, 28.10, 33.51, 39.69])
    midnight = dispersion([26.81, 27.43, 28.41, 30.59, 30.86, 31.33])
    assert ocean["sd"] > 3 * midnight["sd"]
