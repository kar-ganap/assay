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

**Single epoch per batch, pinned 2026-07-28** (``cfg.epochs_per_batch == 1``). Take one gradient
step, discard the rollouts, resample. Consequences, so they are not rediscovered later:

- The importance ratio is **identically 1**, because the rollouts always come from the current
  policy. No importance weighting, no ratio logging, and ``cfg.clip_epsilon`` cannot bind — see
  ``LadderConfig.clipping_is_active``.
- Rung 4 (clipping) is therefore **bit-identical to rung 3** and its cut costs nothing.
- The cost is sample efficiency: roughly 3x fewer gradient steps per dollar than 4 epochs would
  give at this task's short completions. Paid deliberately — staleness is a second uncontrolled
  source of gradient noise, and ablation A is an attempt to attribute gradient noise to the
  *baseline*. If wall-clock later forces the issue, 2 epochs is the sane middle, but then run 4
  must come back and the ratio distribution must be logged.

Two things worth logging carefully because they are the phase's actual findings:

- ``frac_degenerate_groups`` — ablation **D**'s direct observable. The calibration screen measured
  0.115 at step 0 for ``add-2digit`` and *all* of those dead groups are saturation-type, so this is
  predicted to **rise** toward ~0.43. That prediction is falsifiable only if it is logged.
- ``ratio`` distribution, if you reuse rollouts across epochs. Off-policy staleness otherwise
  surfaces as an unrelated-looking instability (plan → *Risks*). Note that with a single epoch per
  batch the importance ratio is identically 1, which makes ``cfg.clip_epsilon`` a no-op.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from assay.crawl.advantage import group_advantages
from assay.crawl.config import LadderConfig
from assay.crawl.policy import Policy, Rollout
from assay.crawl.rewards import Outcome, grade_pair
from assay.crawl.runlog import step_log_writer
from assay.crawl.tasks import Prompt, all_families
from assay.loop import StepLog


def _family_for(cfg: LadderConfig):  # type: ignore[no-untyped-def]
    for family in all_families():
        if family.name == cfg.family:
            return family
    raise ValueError(f"unknown family {cfg.family!r}")


def _scalar(value: Any) -> float:
    """Read a loss term as a plain float without dragging autograd along.

    These terms carry ``requires_grad``; ``float(t)`` on one works but warns and forces a device
    sync. Detaching first says explicitly that this is a *readout*, not part of the graph.
    """
    detach = getattr(value, "detach", None)
    return float(detach() if detach is not None else value)


def _prompt_seed(cfg: LadderConfig, step: int) -> int:
    """Fresh prompts each step, deterministic given the run seed.

    Reusing one prompt set across all steps would let the policy memorise it, turning the reward
    curve into a memorisation curve — the same class of confound as training on the format.
    """
    return cfg.seed * 1_000_000 + step


def train(cfg: LadderConfig, run_dir: Path, *, policy: Policy) -> list[StepLog]:
    """Run one ladder configuration for ``cfg.steps`` steps, logging every step.

    Args:
        cfg: the rung or ablation to run. Runs 1/2/3/7 and ablations A-D are all switch settings
            on this one object — if the loop needs an ``if`` per rung, the modelling is wrong.
        run_dir: destination for ``steps.jsonl``. The manifest is written by the caller, before
            step 1, so a crashed run is still identifiable.
        policy: the model behind ``assay.crawl.policy.Policy``. Injected rather than constructed
            so the loop runs against ``ToyPolicy`` on a machine with no GPU.

    Returns:
        The per-step logs, in step order.
    """
    if cfg.prompts_per_step < 2:
        raise ValueError(
            "prompts_per_step must be >= 2: the half-batch cosine needs two non-empty halves of "
            "whole groups, and groups are never split"
        )

    family = _family_for(cfg)
    logs: list[StepLog] = []

    with step_log_writer(run_dir) as writer:
        for step in range(cfg.steps):
            started = time.perf_counter()

            prompts: list[Prompt] = family.generate(
                cfg.setting, cfg.prompts_per_step, seed=_prompt_seed(cfg, step)
            )
            groups: list[list[Rollout]] = policy.generate(prompts, k=cfg.group_size)
            # Flattened in group order, so index i of `flat` aligns with the i-th advantage.
            flat: list[Rollout] = [r for group in groups for r in group]

            # --- step 2: grading -------------------------------------------------------
            # grade_pair returns (proxy, true) per rollout. The proxy is what the policy
            # optimises; the true leg is measured and MUST NOT reach step 3, or it stops
            # measuring generalisation and becomes a second proxy.
            graded = [
                [
                    grade_pair(cfg.reward, r.text, r.prompt.answer, completion_tokens=r.n_tokens)
                    for r in group
                ]
                for group in groups
            ]
            proxy_by_group = [[proxy.reward for proxy, _ in g] for g in graded]
            true_by_group = [[true.reward for _, true in g] for g in graded]

            flat_proxy = [r for g in proxy_by_group for r in g]
            flat_true = [r for g in true_by_group for r in g]
            flat_graded = [pair for g in graded for pair in g]

            # Pass rate on the TRUE grader, not the proxy: under R_tiebreak every proxy reward
            # exceeds 0 even for wrong answers, so a proxy-based pass rate would read ~100% for a
            # policy that is answering incorrectly. The true leg keeps this comparable across all
            # three reward variants.
            group_pass_rate = sum(
                1 for _, true in flat_graded if true.outcome is Outcome.CORRECT
            ) / len(flat_graded)

            # --- step 3: advantages ----------------------------------------------------
            if cfg.force_unanimous_groups:
                # Ablation D, synthetic on purpose. Overwrites the proxy rewards so every group is
                # all-correct, decoupling reward from the completions. The *arithmetic* is already
                # proven in tests/test_advantage_spec.py; what this arm demonstrates is the
                # consequence — reward flat while wall-clock and tokens accrue exactly as normal.
                proxy_by_group = [[1.0] * len(g) for g in proxy_by_group]

            # Rung 2's baseline is THIS batch's mean over all prompts — not a running average.
            # An EMA would make run 2 differ from run 3 in two ways at once (conditioning *and*
            # temporal smoothing); using the same batch isolates the conditioning, which is the
            # only thing the rung is about. Self-inclusion biases it by O(1/n) at n=128, ~16x
            # smaller than group_mean's at n=8.
            global_baseline = (
                sum(flat_proxy) / len(flat_proxy) if cfg.baseline == "global" else None
            )

            advantages_by_group = [
                group_advantages(
                    rewards,
                    baseline=cfg.baseline,
                    normalize_by_std=cfg.normalize_by_std,
                    global_baseline_value=global_baseline,
                )
                for rewards in proxy_by_group
            ]
            flat_advantages = [a for g in advantages_by_group for a in g]

            # Derived from the advantages, not recomputed from rewards: a group is dead iff it
            # produced no gradient, which is the thing we actually care about. This also stays
            # consistent with group_advantages' own zero-std floor without a second tolerance.
            frac_degenerate = sum(
                1 for g in advantages_by_group if all(a == 0.0 for a in g)
            ) / len(advantages_by_group)

            # --- step 4: loss ----------------------------------------------------------
            # The policy-gradient objective, one term per rollout:
            #
            #     loss_i = -A_i * log pi(y_i | x_i)
            #
            # A_i is a single scalar for the whole episode — terminal reward only, deterministic
            # transitions — so every token in a rollout shares the same advantage. There is no
            # per-timestep credit assignment to do here.
            log_probs = policy.logprobs(flat)  # summed over each rollout's completion tokens

            # max(1, ...) guards a completion that emitted EOS immediately. Kept so step 6 can
            # scale the KL term the same way — mixing a length-normalised policy-gradient term
            # with an unnormalised KL would silently reweight the leash by completion length.
            divisors = [
                (max(1, r.n_tokens) if cfg.length_normalize else 1) for r in flat
            ]

            losses_by_group: list[list[Any]] = []
            index = 0
            for advantages in advantages_by_group:
                group_losses = []
                for advantage in advantages:
                    group_losses.append(-advantage * log_probs[index] / divisors[index])
                    index += 1
                losses_by_group.append(group_losses)

            # --- step 6: KL to the frozen reference -------------------------------------
            # The leash. Ablation B is this term removed, so cfg.kl_coef == 0.0 is the common case
            # and must skip the computation entirely rather than multiply it by zero: for HFPolicy
            # the reference is a whole extra forward pass, and the base weights are frozen under
            # LoRA so "reference" means the adapter disabled.
            #
            # Added to the LOSS, not the reward. Routed through the reward it would pass through
            # group_advantages, where a per-prompt-roughly-constant penalty is largely subtracted
            # away by the group baseline — the leash would quietly go slack.
            #
            # Applied per rollout, inside the group structure, so it lands on the same side of the
            # half-split as the term it belongs to. Adding it after the split would leave the two
            # half-gradients differing by an unaccounted term and the cosine measuring that.
            kl_mean = 0.0
            kl_loss_fraction = 0.0
            # Measured whenever it is penalised, and periodically even when it is not. Ablation B
            # runs at kl_coef=0; without a drift reading there, "removing the leash changed
            # nothing" is indistinguishable from "the policy never drifted anyway".
            measure_only = not cfg.kl_coef and step % max(1, cfg.kl_measure_every) == 0
            if cfg.kl_coef or measure_only:
                pg_magnitude = sum(abs(_scalar(t)) for g in losses_by_group for t in g)
                kl_per_rollout = policy.kl_to_reference(flat)
                index = 0
                for group_losses in losses_by_group if cfg.kl_coef else []:
                    for position in range(len(group_losses)):
                        # Same divisor as the policy-gradient term: the KL is a sum over the
                        # completion's tokens, so leaving it unnormalised while the PG term is
                        # normalised would make the leash tighter on longer completions.
                        group_losses[position] = (
                            group_losses[position]
                            + cfg.kl_coef * kl_per_rollout[index] / divisors[index]
                        )
                        index += 1
                kl_mean = float(sum(_scalar(k) for k in kl_per_rollout) / len(flat))
                kl_magnitude = sum(
                    abs(cfg.kl_coef * _scalar(k) / d)
                    for k, d in zip(kl_per_rollout, divisors, strict=True)
                )
                if pg_magnitude + kl_magnitude > 0:
                    kl_loss_fraction = kl_magnitude / (pg_magnitude + kl_magnitude)

            # --- step 5: update --------------------------------------------------------
            # Split by GROUP, never by rollout. Advantages inside a group sum to zero, so a
            # within-group split hands the two halves complementary pieces of one contrast; their
            # common-mode components anti-correlate by construction and the cosine stops measuring
            # estimator variance. Whole groups keep the halves on disjoint prompts.
            split = (len(losses_by_group) + 1) // 2

            def _half(chunk: list[list[Any]], total: int = len(flat)) -> Any:
                # Divided by the FULL rollout count, not the half's, so loss_a + loss_b is exactly
                # the full-batch mean for any split — including an odd number of groups, where
                # averaging two half-means would quietly be wrong.
                return sum(term for group in chunk for term in group) / total

            grad_norm, cosine = policy.optimize(
                _half(losses_by_group[:split]), _half(losses_by_group[split:])
            )


            log = StepLog(
                step=step,
                proxy_reward=sum(flat_proxy) / len(flat_proxy),
                true_reward=sum(flat_true) / len(flat_true),
                policy_entropy=policy.entropy(flat),
                distinct_completions=len({r.text for r in flat}),
                kl_to_ref=kl_mean,
                kl_loss_fraction=kl_loss_fraction,
                grad_norm=grad_norm,
                half_batch_grad_cosine=cosine,
                # Ablation C's rig-broken branch: a z-score cannot exceed sqrt(G-1) = 2.646 at
                # G=8. If this ever does, the advantage function is not computing a z-score and
                # the run says nothing about the science either way.
                max_abs_advantage=max((abs(a) for a in flat_advantages), default=0.0),
                group_pass_rate=group_pass_rate,
                frac_degenerate_groups=frac_degenerate,
                tokens=sum(r.n_tokens for r in flat),
                wall_clock_s=time.perf_counter() - started,
            )
            writer.append(log)
            logs.append(log)

    return logs
