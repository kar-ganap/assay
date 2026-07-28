"""The ladder table — **user's to own** (``CLAUDE.md`` §7, "the GRPO loop and its ablations").

Left deliberately empty. ``assay.modal_app::ladder`` dispatches whatever is in ``LADDER``, so
filling this in is how the runs get defined — and writing out each rung is part of the point of the
phase, not boilerplate to be handed over.

The cut is taken (plan → *Risks*, 2026-07-27): **runs 1, 2, 3, 7 plus all four ablations.** Runs
4-6 are dropped, which costs the marginal effect of clipping / KL / advantage-normalisation
individually. B isolates KL and C isolates advantage-normalisation; clipping is uncovered but is a
no-op under a single-epoch loop.

Sketch of the intended shape — the values are yours to decide:

.. code-block:: python

    LADDER: dict[str, LadderConfig] = {
        # rung 1: REINFORCE. No baseline, so failed rollouts carry weight exactly zero.
        "run1": LadderConfig(run_id="run1", baseline="none"),

        # rung 2: a global mean baseline. Unbiased, real variance reduction — but the advantage's
        # sign is set by whether the *prompt* was easier than average, not by completion quality.
        "run2": LadderConfig(run_id="run2", baseline="global"),

        # rung 3: the GRPO move. Per-prompt Monte Carlo baseline, leave-one-out (exactly unbiased;
        # the full-group mean includes r_i and carries an O(1/G) bias).
        "run3": LadderConfig(run_id="run3", baseline="group_loo"),

        # rung 7: full GRPO — the reference curve everything else is compared against.
        "run7": LadderConfig(run_id="run7", baseline="group_loo", normalize_by_std=True,
                             kl_coef=..., clip_epsilon=...),

        # A is free: run1 vs run2, no extra run.
        # B: run 7 with the KL leash cut and a degenerate grader. A constant <answer>0</answer>
        #    always pays, so collapse-to-one-string is reachable rather than merely hoped for.
        "ablation_b": LadderConfig(run_id="ablation_b", ..., kl_coef=0.0, reward="format"),

        # C: run 7 with a tie-breaker that looks negligible. On a unanimous group the z-score
        #    normalisation amplifies it to full gradient magnitude. gap == 0.001 * tokens exactly.
        "ablation_c": LadderConfig(run_id="ablation_c", ..., reward="tiebreak"),

        # D: force unanimous groups -> every advantage zero, the step is wasted.
        "ablation_d": LadderConfig(run_id="ablation_d", ..., force_unanimous_groups=True),
    }

Each entry's ``setting`` is overridden at dispatch, so the same table runs the ``add-2digit``
headline arm and the ``add-3digit`` robustness arm without duplication.
"""

from __future__ import annotations

from assay.crawl.config import LadderConfig

#: run_id -> configuration. Empty until written; ``assay.modal_app::ladder`` will raise on dispatch.
LADDER: dict[str, LadderConfig] = {}
