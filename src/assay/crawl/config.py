"""Phase 0.1 run configuration — the seven-run ladder and four ablations as one parameter set.

Separate from ``assay.config``, whose ``ExperimentConfig``/``GraderVariant`` are shaped for the
``bisect`` grader factorial (test visibility × reward shape × sandbox) and do not describe a toy
arithmetic task. Phase 0.1's code is disposable by design; Phase 0.2 rebuilds it under ``verifiers``.

**Every rung and every ablation is a setting on this one dataclass, not a separate code path.** If
the ladder needs an ``if`` in the training loop, something has been modelled wrong:

===============================  =========================================================
run / ablation                   configuration
===============================  =========================================================
1  REINFORCE                     baseline="none"
2  + mean baseline               baseline="global"
3  + group baseline (GRPO)       baseline="group_loo"
4  + ratio clipping              clip_epsilon=0.2
5  + KL to reference             kl_coef>0
6  + advantage normalisation     normalize_by_std=True
7  full GRPO                     all of the above — the reference curve
A  no baseline                   run 1 vs run 2
B  no KL                         run 7, kl_coef=0.0, reward="format"
C  tie-breaker amplification     run 7, reward="tiebreak"
D  unanimous group               run 7, force_unanimous_groups=True
===============================  =========================================================

**The ladder table itself is the user's to write** (``CLAUDE.md`` §7 — "the GRPO loop and its
ablations"). This module supplies the type it is expressed in, not the values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from assay.crawl.advantage import Baseline

#: Which grader supplies the *proxy* reward. The *true* reward is always ``R_binary`` and is never
#: trained on (``assay.crawl.rewards.grade_pair``).
RewardVariant = Literal["binary", "format", "tiebreak"]


@dataclass(frozen=True)
class LadderConfig:
    """One run. Serialised verbatim into the run's ``manifest.json``.

    Task defaults are the values pinned by the calibration sweep on 2026-07-27
    (``docs/phases/phase-0.1-grpo-by-hand-plan.md`` → *RESULT*). ``add-3digit`` is the pre-declared
    robustness arm; overriding ``setting`` is how you run it.
    """

    run_id: str

    # --- task (pinned by measurement, not by intuition) ---------------------------------
    family: str = "arithmetic"
    setting: str = "add-2digit"

    # --- the ladder switches -----------------------------------------------------------
    baseline: Baseline = "group_loo"
    normalize_by_std: bool = False
    kl_coef: float = 0.0
    clip_epsilon: float | None = None

    # --- the proxy/true split ----------------------------------------------------------
    #: Selects the *proxy*. ``"binary"`` makes proxy == true, so the gap is identically zero —
    #: correct for the ladder, which is about the estimator rather than about Goodhart.
    reward: RewardVariant = "binary"

    # --- ablation D --------------------------------------------------------------------
    #: Force every group to be unanimous, to demonstrate the zero-gradient step directly.
    force_unanimous_groups: bool = False

    # --- sizes -------------------------------------------------------------------------
    #: k = G in the calibration screen for exactly this reason: the measured dead-group rate is
    #: the rate this run will see at step 0.
    group_size: int = 8
    prompts_per_step: int = 16
    steps: int = 200
    learning_rate: float = 1e-5

    # --- sampler (must match the screen, or its base rate does not transfer) ------------
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0

    seed: int = 0

    @property
    def has_distinct_true_reward(self) -> bool:
        """True when proxy != true, i.e. when ``gap`` is a meaningful quantity for this run."""
        return self.reward != "binary"

    @property
    def rollouts_per_step(self) -> int:
        return self.prompts_per_step * self.group_size
