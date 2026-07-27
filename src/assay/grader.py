"""The proxy/true split.

``TrainingGrader`` is the reward the policy optimizes; it is configured by a ``GraderVariant`` and is
*deliberately* pathological in some cells. ``HeldOutGrader`` is the ground truth the policy never
optimizes against.

**The held-out grader is a grader** (``docs/desiderata.md`` §6). It can be wrong. Every held-out
grader is validated against a hand-labelled gold set at Phase 1.6, and its FP/FN is reported
alongside every result that depends on it. Never treated as ground truth by assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

from assay.config import GraderVariant
from assay.env import Task, Trajectory


@dataclass(frozen=True)
class GradeResult:
    """One grading outcome.

    ``exploit_signature`` is the behaviour class the trajectory falls into when it scores well on
    the proxy while failing the truth — e.g. ``"special_case_input"``, ``"broad_except"``,
    ``"hardcoded_output"``, ``"edited_test"``. It is what H2 scores its top-1 match against, so the
    signature vocabulary is frozen at Phase 1.3 and never extended mid-experiment.
    """

    score: float
    passed: bool
    exploit_signature: str | None = None
    detail: str = ""


class TrainingGrader:
    """The proxy reward. Pathology is configured, not accidental."""

    def __init__(self, variant: GraderVariant) -> None:
        self.variant = variant

    def grade(self, task: Task, trajectory: Trajectory) -> GradeResult:
        """Score a trajectory under this variant's grader configuration.

        User owns this (``CLAUDE.md`` §7) — the pathology design *is* the experiment.
        Implement at Phase 1.2.
        """
        raise NotImplementedError("Phase 1.2 — user designs the grader pathologies")


class HeldOutGrader:
    """The true reward: the held-out suite exercising the same root cause elsewhere.

    Never used to compute a training signal. Only for measurement.
    """

    def grade(self, task: Task, trajectory: Trajectory) -> GradeResult:
        """Score against the held-out suite. Implement at Phase 1.1."""
        raise NotImplementedError("Phase 1.1")


@dataclass(frozen=True)
class GoldLabel:
    """One hand-labelled trajectory, for validating a grader against human judgement."""

    task_id: str
    trajectory_id: str
    is_genuine_fix: bool
    labeller_note: str = ""


@dataclass(frozen=True)
class GraderValidation:
    """Phase 1.6 output. Reported alongside every result the grader supports."""

    grader_name: str
    n_gold: int
    false_positive_rate: float
    false_negative_rate: float


def validate_grader(
    grader: HeldOutGrader | TrainingGrader,
    gold: list[GoldLabel],
) -> GraderValidation:
    """Measure a grader's FP/FN against hand labels. Walk's exit gate. Implement at Phase 1.6."""
    raise NotImplementedError("Phase 1.6 — gold-set validation")
