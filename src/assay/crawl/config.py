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

    #: Gradient steps taken per batch of rollouts. **Pinned to 1 on 2026-07-28.**
    #:
    #: With one epoch the rollouts are always drawn from the *current* policy, so the importance
    #: ratio is identically 1 and no correction is needed. Reusing rollouts (>1) would amortise
    #: generation — roughly 3x the gradient steps per dollar at this task's ~7-15 token completions
    #: — but it introduces off-policy staleness, which is a **second, uncontrolled source of
    #: gradient noise** in a phase whose deliverable is attributing gradient noise to the *baseline*
    #: (ablation A). The plan's risk section: staleness "will look like an unrelated instability".
    epochs_per_batch: int = 1

    #: Measure KL to the reference every N steps **even when ``kl_coef == 0``**, for observability
    #: only — never applied to the loss.
    #:
    #: Ablation B runs at ``kl_coef=0``, and skipping the computation there would make its result
    #: ambiguous: without a drift measurement, "removing the leash changed nothing" cannot be told
    #: apart from "the policy never drifted anyway". Measuring every 10th step buys that
    #: distinction for a tenth of the cost of an extra forward pass per step.
    kl_measure_every: int = 10

    #: PPO-style ratio clipping. **Inert while ``epochs_per_batch == 1``** — the ratio is exactly 1,
    #: so ``clip(1, 1-eps, 1+eps) == 1``. Kept as a switch rather than deleted so the design space
    #: stays visible, but see ``clipping_is_active``: setting this alone does nothing.
    clip_epsilon: float | None = None

    # --- the proxy/true split ----------------------------------------------------------
    #: Selects the *proxy*. ``"binary"`` makes proxy == true, so the gap is identically zero —
    #: correct for the ladder, which is about the estimator rather than about Goodhart.
    reward: RewardVariant = "binary"

    #: Divide each rollout's summed log-prob by its token count. **Default True.**
    #:
    #: Without it a 40-token rollout contributes 40 gradient terms and a 10-token one contributes
    #: 10, so long rollouts dominate the batch gradient *regardless of their reward*. That is a
    #: length pressure originating in the optimizer rather than the reward — which would confound
    #: ablation **C**, whose entire prediction is that the tie-breaker makes completions longer.
    #:
    #: Contested in the literature: GRPO as published normalises by length; Dr. GRPO removes it and
    #: argues the normalisation introduces its own bias. Check against ``trl``'s ``GRPOTrainer``
    #: (prep-reading item 3) before citing either direction.
    #:
    #: Kept as a switch because running ablation C **both ways** separates "the tie-breaker caused
    #: padding" from "the optimizer caused padding" — a cheap and decisive extra arm.
    length_normalize: bool = True

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
    #: The calibration screen sampled at 256, but **no ``add-2digit`` completion in 400 samples
    #: exceeded 28 tokens** (median 7, p99 23). So 64 is not a compromise — it is removing headroom
    #: that was never used, and the two settings are observationally identical on this task.
    #:
    #: It matters because unused headroom is not free: it inflates the generation KV cache 4x and
    #: pads every scored sequence to the longest completion in the batch, which is the tensor that
    #: gets projected to a 128k vocabulary.
    #:
    #: **Re-check this if the task changes**, or if a run starts producing long completions —
    #: ablation C predicts exactly that, so watch ``median_completion_tokens`` against this cap.
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0

    seed: int = 0

    @property
    def clipping_is_active(self) -> bool:
        """Whether ``clip_epsilon`` can actually bind.

        Guards against a config that *claims* to clip but cannot: with a single epoch the
        importance ratio is identically 1, so clipping is a no-op no matter what epsilon says.
        Rung 4 of the ladder is cut for exactly this reason — under the pinned single-epoch design
        it would be bit-identical to rung 3.
        """
        return self.clip_epsilon is not None and self.epochs_per_batch > 1

    @property
    def has_distinct_true_reward(self) -> bool:
        """True when proxy != true, i.e. when ``gap`` is a meaningful quantity for this run."""
        return self.reward != "binary"

    @property
    def rollouts_per_step(self) -> int:
        return self.prompts_per_step * self.group_size
