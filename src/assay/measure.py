"""Outcome measurement: the gap slope (cheap) and the eta decomposition (headline).

Two-tier by cost (``docs/conceptual.md`` §2.3): the gap slope on every variant, the full
transfer-efficiency decomposition on the four confirmatory arms.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay.loop import StepLog


@dataclass(frozen=True)
class GapSlope:
    """The cheap outcome variable — H1 is scored against this.

    Fitted over steps 50–200 so that a rising, unsaturated gap still yields a graded measurement
    rather than a null (``docs/pre-registration.md`` §4 L3).
    """

    variant_id: str
    slope: float
    slope_stderr: float
    intercept: float
    r_squared: float
    step_range: tuple[int, int]
    n_seeds: int
    seed_band: tuple[float, float]


def fit_gap_slope(
    logs_by_seed: dict[int, list[StepLog]],
    *,
    step_range: tuple[int, int] = (50, 200),
) -> GapSlope:
    """Fit ``d(gap)/d(step)`` with a seed band. Implement at Phase 2.3."""
    raise NotImplementedError("Phase 2.3")


# --------------------------------------------------------------------------------------
# The eta decomposition (Gallop, Phase 3.3)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferEvals:
    """The four evaluations of one trained policy (``docs/conceptual.md`` §2.3).

    ``a_heldout_split``  — A's held-out task split (what people currently report)
    ``b_independent_env`` — A', an independently authored env for the same skill
    ``c_swapped_grader``  — A's tasks with A''s grader (tasks held constant)
    ``d_external_bench``  — an unrelated external benchmark (collateral)
    """

    a_heldout_split: float
    b_independent_env: float
    c_swapped_grader: float
    d_external_bench: float


@dataclass(frozen=True)
class TransferEfficiency:
    """The headline outcome.

    ``pair_source`` is ``"hub"`` or ``"model_authored"``. **Never pool the two**
    (``docs/desiderata.md`` §11); ``idiom_correlation`` is measured and disclosed for model-authored
    pairs, where independence is an argument rather than a fact.
    """

    variant_id: str
    g_total: float
    g_skill: float
    g_grader_idiom: float
    g_env_idiom: float
    g_collateral: float
    eta: float
    pair_source: str
    idiom_correlation: float | None = None


def decompose(baseline: TransferEvals, trained: TransferEvals, *, pair_source: str) -> TransferEfficiency:
    """Compute the gains and eta = G_skill / G_total. Implement at Phase 3.3.

    E1: eta <= 0.7 is the interesting regime; **eta > 0.9 is an honest null worth publishing.**
    E2: predicts ``g_grader_idiom > g_env_idiom``.
    """
    raise NotImplementedError("Phase 3.3")


def eta_trajectory(
    checkpoints: dict[int, TransferEvals],
    baseline: TransferEvals,
    *,
    pair_source: str,
) -> dict[int, float]:
    """eta(step) within a single run — E3 predicts it falls (early RL buys skill, late RL buys idiom).

    Nearly free once the eval harness exists. Implement at Phase 3.3.
    """
    raise NotImplementedError("Phase 3.3")


# --------------------------------------------------------------------------------------
# Hypothesis tests (Phase 2.3) — user writes these (CLAUDE.md §7)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisResult:
    """One pre-registered test. ``band`` is the pre-committed reading, never chosen post hoc."""

    hypothesis: str
    statistic: float
    p_value: float | None
    band: str
    verdict: str


def test_h1(scores: dict[str, float], slopes: dict[str, GapSlope]) -> HypothesisResult:
    """Spearman rho between assay_score and gap slope.

    Pre-committed bands: rho >= 0.6 works · 0.3 <= rho < 0.6 partial · rho < 0.3 honest null.
    """
    raise NotImplementedError("Phase 2.3 — user writes the hypothesis tests")


def test_h2(
    frontier_exploits: dict[str, list[str]],
    converged_exploits: dict[str, str],
) -> HypothesisResult:
    """Top-1 match rate, frontier-ranked exploit vs the policy's modal converged exploit,
    against a uniform-over-observed-exploits baseline."""
    raise NotImplementedError("Phase 2.3 — user writes the hypothesis tests")


def test_h3(
    axes_by_variant: dict[str, dict[str, float]],
    slopes: dict[str, GapSlope],
    speeds: dict[str, float],
) -> tuple[HypothesisResult, HypothesisResult]:
    """Partial R-squared per axis, on the gap *and* on learning speed.

    Predicts A2 dominates the gap; A3 predicts speed but not the gap. Returns (gap, speed).
    """
    raise NotImplementedError("Phase 2.3 — user writes the hypothesis tests")
