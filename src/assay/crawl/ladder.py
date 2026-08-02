"""The ladder table — every rung and ablation as a configuration, never a code path.

The cut is taken (plan → *Risks*, 2026-07-27): **runs 1, 2, 3, 7 plus all four ablations.** Runs 4-6
are dropped. Run 4 (clipping) costs nothing, because under the pinned single-epoch design the
importance ratio is identically 1 and it would be bit-identical to run 3. B isolates KL and C
isolates advantage-normalisation, so only those two are bundled into the 3 → 7 jump.

**Every entry differs from its parent by exactly one field**, and that is enforced by
``test_each_ablation_differs_from_run7_by_one_switch`` rather than left to discipline. It is what
makes the arms comparable: if an ablation changed two things at once, its result would not be
attributable to either. Ablation B is the instructive case — it genuinely needs *two* changes from
run 7 (no leash **and** a degenerate reward), so it is derived through ``ABLATION_B_CONTROL``, and
the **difference between those two** is what isolates KL.

Pinned values, each with its provenance in ``docs/phases/phase-0.1-grpo-by-hand-plan.md``:

- task ``arithmetic/add-3digit`` — a *documented deviation*; the selection rule chose ``add-2digit``,
  which saturates to ~100% dead groups within ten steps.
- ``learning_rate = 1e-5`` — probed on ``add-3digit``; also a deviation, since the amended rule's
  tie-break selected ``1e-4`` and that was overridden on a 200-step extrapolation.
- ``kl_coef = 0.04`` — published GRPO. **Not verifiable at step 0**: KL is ~0 there for any beta.
  The check is on run 7's trajectory — KL plateaus while reward still improves. If it runs away,
  beta was too small and **ablation B's null is a rig failure, not a finding**.
"""

from __future__ import annotations

from dataclasses import replace

from assay.crawl.config import LadderConfig

#: Published GRPO's value. See the module docstring for why it cannot be validated at step 0.
BETA = 0.04

#: The reference curve everything else is compared against. Task, learning rate, group size and
#: sampler all come from ``LadderConfig``'s defaults, which carry the pinned values.
RUN7 = LadderConfig(
    run_id="run7",
    baseline="group_loo",
    normalize_by_std=True,
    kl_coef=BETA,
)

#: Ablation B needs two changes from run 7 — no leash *and* a degenerate reward — so it goes through
#: this control. B against the control isolates KL; the control against run 7 isolates the reward.
ABLATION_B_CONTROL = replace(RUN7, run_id="ablation_b_control", reward="format")

ABLATION_C = replace(RUN7, run_id="ablation_c", reward="tiebreak")

LADDER: dict[str, LadderConfig] = {
    # --- the ladder ---------------------------------------------------------------------
    # Rungs 1-3 build up from the defaults, which already have normalize_by_std=False and
    # kl_coef=0.0 — so **only `baseline` differs between them**, which is what ablation A needs.
    #
    # All three carry `diagnostic_centered_cosine` (added 2026-08-01). The as-applied cosine cannot
    # compare baseline arms — it is inflated on `none` by the common mode a baseline exists to
    # remove, and deflated on `global` by a baseline computed across the cosine's own split boundary
    # (see `config.diagnostic_centered_cosine`). It is a *diagnostic*: the update is provably
    # unchanged (`test_the_centred_cosine_does_not_change_the_update`), which is why it does not
    # count as a ladder switch. Rung 3 carries it as the on-real-model control — the correction is a
    # no-op under a group baseline by algebra, so a non-zero shift there means the instrument, not
    # the arm, is wrong.
    "run1": LadderConfig(run_id="run1", baseline="none", diagnostic_centered_cosine=True),
    "run2": LadderConfig(run_id="run2", baseline="global", diagnostic_centered_cosine=True),
    "run3": LadderConfig(run_id="run3", diagnostic_centered_cosine=True),  # baseline is group_loo
    "run7": RUN7,
    # --- ablations ----------------------------------------------------------------------
    # A is free: run1 vs run2, no entry of its own.
    "ablation_b_control": ABLATION_B_CONTROL,
    "ablation_b": replace(ABLATION_B_CONTROL, run_id="ablation_b", kl_coef=0.0),
    "ablation_c": ABLATION_C,
    "ablation_d": replace(RUN7, run_id="ablation_d", force_unanimous_groups=True),
    # --- length-normalisation arms ------------------------------------------------------
    # run7 has no length-rewarding term, so any drift here is the OPTIMIZER's doing — which is the
    # convention question (GRPO vs Dr. GRPO) isolated. C both ways measures the tie-breaker's share.
    "run7_nolennorm": replace(RUN7, run_id="run7_nolennorm", length_normalize=False),
    "ablation_c_nolennorm": replace(
        ABLATION_C, run_id="ablation_c_nolennorm", length_normalize=False
    ),
}
