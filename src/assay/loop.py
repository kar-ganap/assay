"""GRPO wiring and per-step gap logging.

**User owns this module** (``CLAUDE.md`` §7). Phase 0.1 writes the loop by hand — REINFORCE ->
+baseline -> +group baseline -> +clip -> +KL -> +advantage-norm — and then breaks each component
deliberately. Phase 0.2 rebuilds the same task under the ``verifiers`` spec and hands training off to
``prime-rl`` / ``trl``; this module keeps the parts that stay ours: the gap logging and the manifest.

**Per-step logging is not optional** (``experiments/README.md``). The outcome variable is the slope
``d(gap)/d(step)`` over steps 50–200, not a terminal value — a rising, unsaturated gap is still a
clean measurement (``docs/pre-registration.md`` §4 L3). A run without per-step logs is not usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from assay.config import ExperimentConfig


@dataclass(frozen=True)
class StepLog:
    """One training step. Appended to ``raw/<run>/steps.jsonl``.

    ``proxy_reward`` and ``true_reward`` are the two legs of the gap. ``frac_degenerate_groups`` is
    the direct observable behind ablation D in Phase 0.1 and behind battery axis A3.
    """

    step: int
    proxy_reward: float
    true_reward: float
    policy_entropy: float
    kl_to_ref: float
    grad_norm: float
    group_pass_rate: float
    frac_degenerate_groups: float
    tokens: int
    wall_clock_s: float

    @property
    def gap(self) -> float:
        """The cheap outcome: how much reward comes from the proxy that the truth does not endorse."""
        return self.proxy_reward - self.true_reward


@dataclass
class RunManifest:
    """Pinned provenance for one run (``docs/desiderata.md`` §12).

    A run without a manifest does not enter any analysis.
    """

    run_id: str
    config: ExperimentConfig
    git_sha: str
    grader_prompt_hashes: dict[str, str] = field(default_factory=dict)
    backend: str = "modal"
    notes: str = ""


def write_manifest(manifest: RunManifest, run_dir: Path) -> None:
    """Serialise the manifest before the first step. Scaffolding — Claude may implement."""
    raise NotImplementedError("Phase 0.1 — plumbing")


def train(config: ExperimentConfig, run_dir: Path) -> list[StepLog]:
    """Run GRPO for ``config.train.steps``, logging every step.

    **User writes this** (Phase 0.1 by hand; Phase 1.1 onward via ``verifiers`` + ``prime-rl``/``trl``).
    """
    raise NotImplementedError("Phase 0.1 — user writes the loop")
