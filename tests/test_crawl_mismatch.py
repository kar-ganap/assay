"""M2's sampler/scorer mismatch statistics — pure arithmetic, no GPU, no vLLM.

M2 asks whether swapping in a fast sampler breaks the identity our loss silently depends on. Today
``policy.generate()`` and ``policy.logprobs()`` are the same HF forward pass, so the importance ratio
is **1 by construction** and ``config.py`` cuts rung 4 of the ladder for exactly that reason. vLLM
makes sampler and scorer different implementations, and the ratio stops being 1.

The measurement itself needs a GPU. Everything that turns per-token log-probs into a verdict does
not, and that is what these tests pin — including the two ways the measurement could lie to us:
tokenizer misalignment masquerading as a large mismatch, and correlated per-token errors making the
iid extrapolation to longer sequences too optimistic.
"""

from __future__ import annotations

import math

import pytest

from assay.crawl.mismatch import (
    OPERATING_LENGTH,
    RATIO_BAND,
    implied_sequence_ratio,
    mismatch_statistics,
    mismatch_verdict,
    token_discrepancies,
)


def _sequences(deltas_per_seq: list[list[float]], *, base: float = -1.5) -> tuple[list, list]:
    """Build (hf, vllm) per-token log-prob sequences realising the requested discrepancies."""
    vllm = [[base] * len(d) for d in deltas_per_seq]
    hf = [[base + x for x in d] for d in deltas_per_seq]
    return hf, vllm


# --------------------------------------------------------------------------------------
# token_discrepancies — the alignment contract
# --------------------------------------------------------------------------------------


def test_identical_scorers_give_exactly_zero_discrepancy() -> None:
    """The positive control. Scoring a sampler against itself must give 0, not 'small'.

    This is the check that distinguishes 'vLLM differs from HF' from 'my harness is wrong': if the
    HF-vs-HF control is not identically zero, no vLLM number from the same code path means anything.
    """
    hf, vllm = _sequences([[0.0, 0.0, 0.0], [0.0, 0.0]])
    assert token_discrepancies(hf, vllm) == [0.0] * 5


def test_discrepancy_is_hf_minus_vllm_in_that_order() -> None:
    """Sign convention is load-bearing: the loss's ratio is pi_HF/pi_vLLM = exp(delta).

    Getting this backwards inverts every ratio, and since the band is symmetric-ish around 1 the
    error would survive the verdict undetected.
    """
    deltas = token_discrepancies([[-1.0]], [[-1.5]])
    assert deltas == [0.5]
    assert math.isclose(math.exp(deltas[0]), math.exp(-1.0) / math.exp(-1.5))


def test_length_mismatch_is_an_error_not_a_truncation() -> None:
    """A ragged pair means the two scorers saw different token sequences.

    Silently zipping to the shorter one would report a *small* mismatch for the worst possible
    failure — a tokenizer disagreement. It has to raise.
    """
    with pytest.raises(ValueError, match="length"):
        token_discrepancies([[-1.0, -1.0]], [[-1.0]])


def test_sequence_count_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="sequences"):
        token_discrepancies([[-1.0], [-1.0]], [[-1.0]])


# --------------------------------------------------------------------------------------
# mismatch_statistics — per-token distribution, length-independent by construction
# --------------------------------------------------------------------------------------


def test_statistics_are_length_independent() -> None:
    """The per-token distribution must not move when the same errors arrive in longer sequences.

    This is why the plan reports per-token rather than per-sequence: the sequence-level number is a
    statement about our max_new_tokens, the per-token one is a statement about the samplers.

    The two arms hold the **token count** fixed (256) and vary only how it is split into sequences.
    Varying the count instead would move ``std`` through the Bessel correction alone, which is a
    statement about sample size, not about length -- a distinction the first draft of this test got
    wrong and asserted through.
    """
    short = mismatch_statistics(*_sequences([[0.01, -0.01] * 4] * 32))
    long = mismatch_statistics(*_sequences([[0.01, -0.01] * 16] * 8))
    assert short["n_tokens"] == long["n_tokens"] == 256
    assert short["mean_length"] != long["mean_length"]
    assert math.isclose(short["mean"], long["mean"], abs_tol=1e-12)
    assert math.isclose(short["std"], long["std"], abs_tol=1e-12)


def test_statistics_report_the_tail_not_just_the_centre() -> None:
    """One catastrophic token in a thousand is invisible in the mean and fatal to the ratio."""
    deltas = [[0.0] * 999 + [4.0]]
    stats = mismatch_statistics(*_sequences(deltas))
    assert math.isclose(stats["max_abs"], 4.0)
    assert stats["mean"] < 0.01
    assert stats["p99"] < stats["max_abs"]


def test_max_off_policy_matches_the_field_metric() -> None:
    """prime-rl logs `Max Off-Policy` on every line; ours must mean the same thing.

    It is the largest per-token deviation of the ratio from 1, so a 2x token and a 0.5x token are
    not reported as equally bad -- exp(+0.7) is 1.01 away from 1 more than exp(-0.7) is.
    """
    stats = mismatch_statistics(*_sequences([[0.0, 0.5, -0.5]]))
    assert math.isclose(stats["max_off_policy"], math.exp(0.5) - 1.0, rel_tol=1e-9)


def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError, match="no completion tokens"):
        mismatch_statistics([], [])


# --------------------------------------------------------------------------------------
# independence — whether extrapolating to 1024 tokens is legitimate at all
# --------------------------------------------------------------------------------------


def test_independent_token_errors_give_independence_ratio_near_one() -> None:
    """iid deltas: Var(sum over L) == L * Var(token). The ratio is the diagnostic."""
    seqs = [[0.1, -0.1, 0.1, -0.1], [-0.1, 0.1, -0.1, 0.1], [0.1, 0.1, -0.1, -0.1]]
    stats = mismatch_statistics(*_sequences(seqs))
    assert stats["independence_ratio"] < 1.0


def test_correlated_token_errors_are_detected() -> None:
    """Per-sequence bias -- every token in a sequence off in the same direction -- makes the
    sequence-level variance scale like L^2, not L, so the iid extrapolation understates the drift.

    If this went unnoticed, a comfortable per-token number would be reported as a comfortable
    sequence ratio at 1024 tokens when it is not.
    """
    seqs = [[0.1] * 8, [-0.1] * 8, [0.1] * 8, [-0.1] * 8]
    stats = mismatch_statistics(*_sequences(seqs))
    assert stats["independence_ratio"] > 4.0


# --------------------------------------------------------------------------------------
# implied_sequence_ratio — the extrapolation the verdict is taken on
# --------------------------------------------------------------------------------------


def test_zero_mean_drift_still_has_spread_at_length() -> None:
    """A centred per-token error does not cancel: the sum random-walks as sigma*sqrt(L).

    Reporting only the median ratio would call a 1.0 median 'negligible' while individual sequences
    sit far off. The band has to be taken on the interval.
    """
    stats = mismatch_statistics(*_sequences([[0.02, -0.02] * 32] * 16))
    at = implied_sequence_ratio(stats, 512)
    assert math.isclose(at["median"], 1.0, abs_tol=1e-6)
    assert at["hi"] > at["median"] > at["lo"]


def test_drift_compounds_multiplicatively_with_length() -> None:
    """The whole reason this is measured per-token: a 0.001 nat/token bias is 1.0 at one token and
    e^1.024 at 1024."""
    stats = mismatch_statistics(*_sequences([[0.001] * 64] * 8))
    assert math.isclose(implied_sequence_ratio(stats, 64)["median"], math.exp(0.064), rel_tol=1e-9)
    assert math.isclose(implied_sequence_ratio(stats, 1024)["median"], math.exp(1.024), rel_tol=1e-9)


def test_ratio_at_length_one_is_the_per_token_ratio() -> None:
    stats = mismatch_statistics(*_sequences([[0.3] * 4]))
    assert math.isclose(implied_sequence_ratio(stats, 1)["median"], math.exp(0.3), rel_tol=1e-9)


# --------------------------------------------------------------------------------------
# mismatch_verdict — the pre-registered band, and both failure branches
# --------------------------------------------------------------------------------------


def test_band_is_the_one_pre_registered_in_the_phase_plan() -> None:
    """Pinned so the band cannot drift between the plan and the code that applies it."""
    assert RATIO_BAND == (0.9, 1.1)


def test_negligible_when_the_whole_interval_sits_inside_the_band() -> None:
    stats = mismatch_statistics(*_sequences([[1e-5, -1e-5] * 32] * 16))
    verdict = mismatch_verdict(stats, operating_length=OPERATING_LENGTH)
    assert verdict["verdict"] == "negligible"
    assert verdict["band"] == RATIO_BAND


def test_not_free_when_the_median_leaves_the_band() -> None:
    """A systematic per-token bias large enough to move the sequence ratio out of the band."""
    stats = mismatch_statistics(*_sequences([[0.01] * 64] * 16))
    verdict = mismatch_verdict(stats, operating_length=512)
    assert verdict["verdict"] == "not_free"


def test_spread_alone_can_leave_the_band_with_a_centred_median() -> None:
    """The failure the median hides: no drift, but individual sequences land far from 1.

    An unbiased-on-average ratio still breaks the estimator per-sequence, which is exactly what
    clipping exists to contain -- so this must not be reported as negligible.
    """
    stats = mismatch_statistics(*_sequences([[0.05, -0.05] * 32] * 16))
    verdict = mismatch_verdict(stats, operating_length=512)
    assert math.isclose(verdict["at_length"]["median"], 1.0, abs_tol=1e-6)
    assert verdict["verdict"] == "not_free"


def test_rig_broken_fires_before_any_verdict_about_vllm() -> None:
    """Misaligned tokens produce huge structureless deltas. That is a statement about the harness,
    not about vLLM, and it must not be reported as 'not_free' -- which would read as a finding.

    Same shape as the screen's parse_fail branch: separate 'the thing under test is bad' from
    'the instrument is broken', and check the instrument first.
    """
    stats = mismatch_statistics(*_sequences([[6.0, -7.0, 8.0] * 8] * 8))
    assert mismatch_verdict(stats, operating_length=512)["verdict"] == "rig_broken"


def test_non_finite_is_rig_broken() -> None:
    stats = mismatch_statistics(*_sequences([[float("nan"), 0.0]]))
    assert mismatch_verdict(stats, operating_length=512)["verdict"] == "rig_broken"


def test_verdict_reports_every_requested_length_not_only_the_operating_one() -> None:
    """The plan asks for 64 / 512 / 1024 so the length-sensitivity is visible, and the verdict is
    taken at one pinned length rather than whichever looks best."""
    stats = mismatch_statistics(*_sequences([[0.001] * 32] * 8))
    verdict = mismatch_verdict(stats, operating_length=512)
    assert sorted(verdict["by_length"]) == [64, 512, 1024]
    assert verdict["operating_length"] == 512
    assert verdict["at_length"] == verdict["by_length"][512]
