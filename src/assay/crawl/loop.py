"""The GRPO training loop — **user writes this** (``CLAUDE.md`` §7).

This is the phase's learning target. Everything around it is scaffolded: task generation
(``tasks``), grading and the proxy/true split (``rewards.grade_pair``), manifests and per-step
logging (``runlog``), remote execution (``assay.modal_app.train_remote``), and the advantage
function's executable spec (``tests/test_advantage_spec.py``).

Shape of the work, per step:

1. Draw ``cfg.prompts_per_step`` prompts from ``tasks`` at ``cfg.family``/``cfg.setting``.
2. Sample ``cfg.group_size`` completions per prompt — **the group is one prompt's rollouts**, which
   is what makes the baseline an estimate of ``E[R|x]`` rather than a mixture over prompts. Mixing
   prompts inside a group silently reverts GRPO to a global baseline.
3. Grade each completion with ``rewards.grade_pair(cfg.reward, ...)`` -> ``(proxy, true)``.
   **Optimise ``proxy`` only.** ``true`` is measured and never enters the loss; the moment it does
   it stops measuring generalisation and becomes another proxy.
4. ``advantage.group_advantages(proxy_rewards, baseline=cfg.baseline,
   normalize_by_std=cfg.normalize_by_std)`` — per group, never across groups.
5. Policy-gradient loss, optionally clipped (``cfg.clip_epsilon``) and KL-penalised
   (``cfg.kl_coef``), backward, step.
6. Append a ``StepLog`` via ``runlog.step_log_writer`` — **every step, no exceptions**. The outcome
   variable is ``d(gap)/d(step)`` over steps 50-200, so a run without per-step logs is unusable and
   cannot be repaired after the fact.

Two things worth logging carefully because they are the phase's actual findings:

- ``frac_degenerate_groups`` — ablation **D**'s direct observable. The calibration screen measured
  0.115 at step 0 for ``add-2digit`` and *all* of those dead groups are saturation-type, so this is
  predicted to **rise** toward ~0.43. That prediction is falsifiable only if it is logged.
- ``ratio`` distribution, if you reuse rollouts across epochs. Off-policy staleness otherwise
  surfaces as an unrelated-looking instability (plan → *Risks*). Note that with a single epoch per
  batch the importance ratio is identically 1, which makes ``cfg.clip_epsilon`` a no-op.
"""

from __future__ import annotations

from pathlib import Path

from assay.crawl.config import LadderConfig
from assay.loop import StepLog


def train(cfg: LadderConfig, run_dir: Path) -> list[StepLog]:
    """Run one ladder configuration for ``cfg.steps`` steps, logging every step.

    Args:
        cfg: the rung or ablation to run. Runs 1/2/3/7 and ablations A-D are all switch settings
            on this one object — if the loop needs an ``if`` per rung, the modelling is wrong.
        run_dir: destination for ``manifest.json`` and ``steps.jsonl``. The manifest is written
            before step 1 so a crashed run is still identifiable.

    Returns:
        The per-step logs, in step order.
    """
    raise NotImplementedError(
        "Phase 0.1 — user writes the GRPO loop (CLAUDE.md §7). "
        "Start with assay.crawl.advantage.group_advantages; spec in tests/test_advantage_spec.py"
    )
