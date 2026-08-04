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
    onset_verdict,
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
    assert v["r1p_confirmed"] is True          # ocean before midnight = our base-rate order
    assert v["ordering_matches_prime"] is False  # which is NOT Prime's order
    assert v["verdict"] == "partial"


def test_r1p_falsified_when_the_published_ordering_wins() -> None:
    v = onset_verdict({"ocean": 42.0, "midnight": 17.0, "forgotten": 10.0})
    assert v["r1p_confirmed"] is False
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
    assert v["r1p_confirmed"] is None  # the discriminating pair is incomplete
