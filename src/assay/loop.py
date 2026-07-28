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

import dataclasses
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

    #: Unique completion *strings* across the whole batch, out of ``prompts_per_step * group_size``.
    #: Ablation **B**'s sharpest signature: entropy collapse predicts this falls toward 1 while
    #: proxy reward stays high — a conjunction only collapse produces, unlike "reward got worse",
    #: which is equally consistent with dead groups, buried signal, or a broken rig.
    distinct_completions: int

    kl_to_ref: float
    grad_norm: float

    #: Cosine similarity between the gradients of the batch's two halves — a **direct** measure of
    #: estimator variance, and ablation **A**'s primary signature.
    #:
    #: The two halves are independent samples of the same expected gradient, so ~1.0 means they
    #: agree (low noise) and ~0 means they disagree (high noise). Unlike ``CV(grad_norm)`` measured
    #: across steps, this carries no trend confound: a gradient that decays smoothly has high CV
    #: with zero step-to-step noise.
    #:
    #: Costs no extra backward passes — every sample is still backwarded exactly once, accumulated
    #: into two buffers rather than one, and summed for the actual update. The price is a second
    #: gradient buffer, which is negligible under LoRA.
    half_batch_grad_cosine: float

    #: ``max |A|`` over the step's advantages. Ablation **C**'s *rig-broken* branch: the normalised
    #: advantage is a z-score, so this can never exceed ``sqrt(G-1)`` = 2.646 at G=8. If it does,
    #: the implementation is not computing a z-score and the run says nothing about the science.
    max_abs_advantage: float

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
    """Serialise the manifest before the first step, so a crashed run is still identifiable.

    Phase 0.1 uses ``assay.crawl.runlog.manifest_for``, whose config shape is the toy ladder rather
    than the ``bisect`` grader factorial. This entry point is for Walk onward.
    """
    from assay.crawl.runlog import write_manifest as _write

    _write(dataclasses.asdict(manifest), run_dir)


def train(config: ExperimentConfig, run_dir: Path) -> list[StepLog]:
    """Run GRPO for ``config.train.steps``, logging every step.

    **User writes this** (Phase 0.1 by hand; Phase 1.1 onward via ``verifiers`` + ``prime-rl``/``trl``).
    """
    raise NotImplementedError("Phase 0.1 — user writes the loop")
