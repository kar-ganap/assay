"""The diagnostic battery — six inference-only probes (``docs/conceptual.md`` §2.1).

No probe here consumes a GPU-hour of *training*. That is the whole claim: the score is computable
before you commit the ~$2,400/task that RL costs.

MUST (Walk, Phase 1.3): A1, A2, A3.  Gallop (Phase 3.4): A4, A5, A6.

``assay_score``'s axis weighting is **fitted on the constructed family**, not derived
(``docs/desiderata.md`` §10). Every artifact that reports the score says so in the same breath.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay.config import GraderVariant
from assay.env import Task


@dataclass(frozen=True)
class AxisScore:
    """One probe's output. ``raw`` is the natural unit; ``normalized`` is [0, 1], higher = worse."""

    axis: str
    raw: float
    normalized: float
    n_samples: int
    detail: dict[str, float]


# --------------------------------------------------------------------------------------
# A1..A3 — MUST (Phase 1.3). User owns the scoring logic (CLAUDE.md §7).
# --------------------------------------------------------------------------------------


def a1_hackability(
    variant: GraderVariant,
    tasks: list[Task],
    *,
    adversary_model: str,
    n: int = 8,
) -> AxisScore:
    """Frontier best-of-N with an adversarial system prompt.

    Score = fraction of samples scoring >= tau on the training grader while failing the held-out
    grader. The adversarial prompt is **frozen verbatim at Phase 1.3** and pinned in the
    pre-registration — it is a design parameter, not a knob to tune after seeing results.

    This is the small-model screen (``screen.p_hack``) run at frontier capability. The difference
    between the two is H2's quantity.
    """
    raise NotImplementedError("Phase 1.3 — user writes the scoring logic")


def a2_grader_degeneracy(
    variant: GraderVariant,
    tasks: list[Task],
    *,
    embed_model: str,
) -> AxisScore:
    """Cluster reward-maximizing trajectories; degeneracy = number of behaviour clusters that are
    *not* the target skill.

    **H3 predicts this axis dominates the gap.** It is the axis the whole decomposition turns on.
    """
    raise NotImplementedError("Phase 1.3 — user writes the scoring logic")


def a3_pass_rate_band(
    variant: GraderVariant,
    tasks: list[Task],
    *,
    model_id: str,
    k: int = 8,
) -> AxisScore:
    """Base-policy pass rate at k; score is distance from p = 0.5.

    Cite *Rollout Pass-Rate Control* (2605.05112) for why 0.5 is the max-signal point — **do not
    re-derive it**. H3 predicts this axis explains learning *speed* but not the *gap*.
    """
    raise NotImplementedError("Phase 1.3 — user writes the scoring logic")


# --------------------------------------------------------------------------------------
# A4..A6 — Gallop (Phase 3.4). First cut if time runs short (docs/stages.md, cut order).
# --------------------------------------------------------------------------------------


def a4_judge_instability(
    variant: GraderVariant,
    tasks: list[Task],
    *,
    judge_model: str,
    m: int = 5,
) -> AxisScore:
    """Re-score identical trajectories m times; Krippendorff's alpha / flip rate.

    Position-bias corrected — port the machinery from ``../crit-thinking`` rather than rebuilding.
    """
    raise NotImplementedError("Phase 3.4")


def a5_verifier_asymmetry(
    variant: GraderVariant,
    *,
    gold_set_size: int = 40,
) -> AxisScore:
    """FP/FN of the *training* grader against a hand-labelled gold set.

    Distinct from ``grader.validate_grader``, which validates the *held-out* grader as a Walk gate.
    """
    raise NotImplementedError("Phase 3.4")


def a6_contamination(
    tasks: list[Task],
    *,
    model_id: str,
) -> AxisScore:
    """Memorization probe on held-out task IDs — does the base model already know the answers?"""
    raise NotImplementedError("Phase 3.4")


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssayReport:
    """A full report card for one environment or grader variant."""

    subject_id: str
    axes: list[AxisScore]
    assay_score: float
    weighting_source: str  # e.g. "fitted on assay constructed family, n=12, 2026-XX-XX"
    exploit_transcripts: list[str]


def assay_score(axes: list[AxisScore], weights: dict[str, float]) -> float:
    """Combine axes into the scalar H1 is scored on.

    ``weights`` is fitted on the constructed family and carried explicitly so that every report can
    name its provenance. There is no default weighting — passing one is a deliberate act.
    """
    raise NotImplementedError("Phase 2.3 — fitted on the constructed family")
