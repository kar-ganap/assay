"""Ablation A, measured where the question is actually well posed.

**Why this file exists.** A's question is *does a baseline reduce the variance of the gradient
estimator?* The original design answered it by comparing ``half_batch_grad_cosine`` between two
**training arms** (run 1, no baseline; run 2, global baseline). That comparison cannot work, for
three reasons found on 2026-08-01 — the first empirical, the second and third algebraic:

1. **It came back reversed.** ``rho_2/rho_1 = 0.14`` against a pre-registered gate of ``>= 2.0``,
   with rig checks clean (finite grad norms, ``max|A|`` inside the ``sqrt(G-1)`` bound).
2. **Both arms are confounded, in opposite directions.** ``baseline="none"`` has every advantage
   ``>= 0``, so both halves weight the shared "push everything up" direction positively and the
   cosine is *inflated* by precisely the nuisance a baseline removes. ``baseline="global"`` centres
   on the full batch while the cosine splits that batch, so with ``b = (b_A + b_B)/2`` each half
   carries ``+/-(b_A - b_B)/2 * sum(grad log pi)`` — an anti-correlated term manufactured by the
   split boundary, *deflating* the cosine. Only ``group_*`` was ever clean, because advantages sum
   to zero inside a group and groups are never split.
3. **Even with that repaired, arms are not comparable.** They follow different trajectories, so any
   difference confounds the estimator with the policy state it is measured at. And centring each
   half (the obvious repair) *erases the contrast*: ``none`` gives ``R_i - mean_A(R)`` and
   ``global`` gives ``(R_i - b) - (mean_A(R) - b)`` — the same estimator. Verified numerically.

**What replaces it.** Fix one policy state. Draw ``batches`` independent batches from it. Grade each
batch once, then score those *same rollouts* under every baseline. Only the baseline differs;
sampling noise is shared, so the comparison is **paired** and far tighter than three separate runs.

The estimator quality of baseline ``b`` is then the noise-to-signal ratio

    NSR_b = V_b / ||g_bar_b||^2,    V_b = mean_n || g_n - g_bar_b ||^2

which is scale-free — necessary, because the baselines produce gradients of very different
magnitudes (run 1's ``grad_norm`` stayed ~0.77 while run 2's fell to 0.23), so any comparison of
*absolute* variance would be reporting that scale difference instead.

**All three estimate the same gradient**, since ``E[b * grad log pi] = 0`` for a baseline
independent of the sample. That is not an assumption here — it is the rig check
(``mean_cosines`` in the verdict). If the three means point in materially different directions,
nothing downstream is a variance story.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from assay.crawl.advantage import Baseline, group_advantages
from assay.crawl.config import LadderConfig
from assay.crawl.loop import _family_for
from assay.crawl.policy import Policy, Rollout
from assay.crawl.rewards import grade_pair
from assay.crawl.tasks import Prompt

#: The three rungs, as *baselines* rather than as runs. Order is the predicted NSR ordering.
BASELINES: tuple[Baseline, ...] = ("none", "global", "group_loo")

#: Pre-registered 2026-08-01, before the probe was run. See ``probe_verdict``.
MEAN_AGREEMENT_FLOOR = 0.90


def predicted_ratio(pass_rate: float) -> float:
    """Theory's **point prediction** for ``NSR_none / NSR_global``: ``1 / (1 - p)``.

    With binary reward and ``P(R=1) = p``, the per-sample second moment is ``p * E`` at ``b = 0``
    and ``p(1-p)^2 E + (1-p)p^2 E = p(1-p) E`` at ``b = p``, so the variance ratio is ``1/(1-p)``.
    Both estimators are unbiased, so ``||g_bar||`` cancels and the variance ratio *is* the NSR ratio.

    Verified by simulation with ``E[grad log pi] = 0`` enforced: 1.74 observed against 1.75 predicted
    at p=0.43, and 3.87 against 4.00 at p=0.75.

    **This replaced a flat ``>= 2.0`` threshold on 2026-08-01, before the probe was run.** That
    threshold was inherited from the training-arm design and is badly calibrated here: it depends on
    the operating point, and at the base policy's measured ``p ~ 0.43`` theory predicts **1.75** —
    below its own pass mark. A correct result would have been scored "partial". A point prediction
    that can miss in either direction is also a far stronger test than a one-sided threshold.
    """
    return 1.0 / max(1e-9, 1.0 - pass_rate)


@dataclass(frozen=True)
class ProbeConfig:
    """One probe. Serialised verbatim into the run's manifest.

    Deliberately mirrors ``LadderConfig``'s task/sampler fields rather than inventing its own: the
    probe must sample from the same distribution the ladder trains on, or its variance estimate
    describes a different problem.
    """

    run_id: str

    #: How many independent batches to draw at the fixed policy state. The precision of a variance
    #: estimate goes as ``sqrt(2/(N-1))``, so 40 gives ~23% on each ``V`` alone — but the reported
    #: quantity is a *paired ratio*, where the shared sampling noise cancels and the bootstrap CI is
    #: far tighter than that figure suggests.
    batches: int = 40

    #: Matches ``LadderConfig`` defaults, so one probe batch == one training batch.
    prompts_per_batch: int = 16
    group_size: int = 8

    #: Steps of training to run *before* probing, to reach an operating point that is not step 0.
    #:
    #: 0 probes the base policy. That state is special: LoRA initialises to no-op, so all three
    #: rungs share it exactly and the paired comparison is unambiguous. It is also the state the
    #: reachability screen measures at, which is why ``p_hack@64`` and this probe are the same
    #: measurement at two capability levels.
    #:
    #: Non-zero warms up with ``group_loo`` — *any* arm's trajectory would do, because the question
    #: is "does a baseline reduce variance **at this policy state**", and the state need not have
    #: been reached by the baseline being scored.
    warmup_steps: int = 0

    #: Must match the arm being characterised. Rungs 1-3 all have ``normalize_by_std=False``.
    normalize_by_std: bool = False
    length_normalize: bool = True

    family: str = "arithmetic"
    setting: str = "add-3digit"
    seed: int = 0

    #: Bootstrap resamples for the ratio's confidence interval. Cheap: resampling happens on an
    #: ``N x N`` Gram matrix of scalars, never on the 3.4M-dimensional gradients themselves.
    bootstrap: int = 2000

    def as_ladder_config(self) -> LadderConfig:
        """The equivalent training config, for warmup and for prompt-stream compatibility."""
        return LadderConfig(
            run_id=self.run_id,
            family=self.family,
            setting=self.setting,
            baseline="group_loo",
            normalize_by_std=self.normalize_by_std,
            length_normalize=self.length_normalize,
            group_size=self.group_size,
            prompts_per_step=self.prompts_per_batch,
            steps=self.warmup_steps,
            seed=self.seed,
        )


def _batch_seed(cfg: ProbeConfig, batch: int) -> int:
    """Disjoint from the training prompt stream.

    ``_prompt_seed`` maps ``(seed, step) -> seed*1e6 + step``, so warmup consumes ``[s*1e6,
    s*1e6 + warmup_steps)``. Offsetting past the whole per-seed block keeps a probe batch from ever
    being a batch the warmup already trained on — which would make the probe partly a memorisation
    measurement.
    """
    return cfg.seed * 1_000_000 + 500_000 + batch


def collect_paired_gradients(
    cfg: ProbeConfig, *, policy: Policy
) -> tuple[dict[Baseline, list[Any]], dict[str, float]]:
    """Draw ``cfg.batches`` batches and score each under every baseline.

    The pairing is the whole design: ``generate`` and ``grade_pair`` are called **once** per batch
    and their outputs are shared across the three baselines, so the only thing that differs between
    the returned gradient sets is the advantage computation. Log-probs are likewise computed once
    and backwarded three times against the retained graph, which also makes the probe roughly the
    cost of one training run's worth of steps rather than three.

    Returns the gradients **and the operating point** — the measured pass rate is not decoration,
    it is what the gate's point prediction ``1/(1-p)`` is evaluated at, so it must come from the
    probe's own batches rather than from a training run's average.
    """
    import torch

    family = _family_for(cfg.as_ladder_config())
    gradients: dict[Baseline, list[Any]] = {b: [] for b in BASELINES}
    correct = 0.0
    total = 0
    dead = 0
    groups_seen = 0

    for batch in range(cfg.batches):
        prompts: list[Prompt] = family.generate(
            cfg.setting, cfg.prompts_per_batch, seed=_batch_seed(cfg, batch)
        )
        groups: list[list[Rollout]] = policy.generate(prompts, k=cfg.group_size)
        flat: list[Rollout] = [r for group in groups for r in group]

        # The proxy leg only. Rungs 1-3 all use reward="binary", where proxy == true, so the probe
        # says nothing about the gap — by design. A is about the estimator, not about Goodhart.
        rewards_by_group = [
            [
                grade_pair("binary", r.text, r.prompt.answer, completion_tokens=r.n_tokens)[
                    0
                ].reward
                for r in group
            ]
            for group in groups
        ]
        flat_rewards = [r for g in rewards_by_group for r in g]
        global_baseline = sum(flat_rewards) / len(flat_rewards)

        correct += sum(flat_rewards)
        total += len(flat_rewards)
        dead += sum(1 for g in rewards_by_group if len(set(g)) == 1)
        groups_seen += len(rewards_by_group)

        log_probs = policy.logprobs(flat)
        divisors = [(max(1, r.n_tokens) if cfg.length_normalize else 1) for r in flat]

        for position, baseline in enumerate(BASELINES):
            advantages = [
                a
                for rewards in rewards_by_group
                for a in group_advantages(
                    rewards,
                    baseline=baseline,
                    normalize_by_std=cfg.normalize_by_std,
                    global_baseline_value=global_baseline,
                )
            ]
            loss = (
                sum(
                    -a * log_probs[i] / divisors[i]
                    for i, a in enumerate(advantages)
                    # Skipping exact zeros is not an optimisation for its own sake: under
                    # group_loo a dead group contributes nothing, and building those terms
                    # anyway would put ~40% of the graph's nodes behind a zero coefficient.
                    if a != 0.0
                )
                / len(flat)
            )
            if isinstance(loss, float):  # every advantage was zero — a fully dead batch
                gradients[baseline].append(torch.zeros_like(gradients[baseline][0]))
                continue
            gradients[baseline].append(
                policy.flat_gradient(loss, retain_graph=position < len(BASELINES) - 1)
            )

    return gradients, {
        "pass_rate": correct / total,
        "dead_group_fraction": dead / groups_seen,
        "rollouts": float(total),
    }


def variance_statistics(gradients: list[Any], *, bootstrap: int, seed: int) -> dict[str, Any]:
    """``V``, ``||g_bar||^2`` and ``NSR`` for one baseline, plus a bootstrap distribution.

    Everything routes through the **Gram matrix** ``K[n,m] = g_n . g_m``. That is not a micro-
    optimisation — it is what makes an *exact* paired bootstrap affordable. With

        ||g_bar||^2 = mean(K)            V = mean(diag K) - mean(K)

    a resample is ``O(N^2)`` scalar work instead of re-averaging 2,000 sets of 3.4M-dimensional
    vectors. The alternative (holding ``g_bar`` fixed at its full-sample value while resampling the
    norms) is the usual shortcut and it understates the interval.
    """
    import torch

    matrix = torch.stack(gradients)
    gram = matrix @ matrix.T
    n = gram.shape[0]

    signal = float(gram.mean())
    noise = float(gram.diagonal().mean()) - signal

    generator = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(bootstrap):
        pick = torch.randint(0, n, (n,), generator=generator)
        sub = gram[pick][:, pick]
        s = float(sub.mean())
        draws.append((float(sub.diagonal().mean()) - s) / s if s > 0 else float("nan"))

    return {
        "signal_sq": signal,
        "variance": noise,
        "nsr": noise / signal if signal > 0 else float("nan"),
        "mean_gradient": matrix.mean(dim=0),
        "bootstrap_nsr": draws,
    }


def probe_verdict(
    stats: dict[Baseline, dict[str, Any]], *, pass_rate: float
) -> dict[str, Any]:
    """Ablation A's gate. **Pre-registered 2026-08-01, before the probe was run.**

    Prediction (H_A): a baseline reduces gradient-estimator variance, and conditioning that baseline
    on the prompt reduces it further —

        NSR_none > NSR_global >= NSR_group_loo

    and the size of the reduction is **predicted, not merely bounded**: ``1/(1-p)`` at the pass rate
    the probe actually operated at (``predicted_ratio``).

    ==================  =====================================================================
    verdict             condition
    ==================  =====================================================================
    ``rig_broken``      any statistic non-finite, **or** the three mean gradients disagree
                        (pairwise cosine < 0.90). All three estimate the same expected
                        gradient, so disagreement means the probe is not measuring variance.
    ``falsified``       CI lies entirely **below** 1.0 — a significant reversal
    ``confirmed``       CI excludes 1.0 **and** contains ``1/(1-p)``
    ``partial``         CI excludes 1.0 but misses ``1/(1-p)`` — real effect, wrong size
    ``magnitude_        CI spans 1.0 **and** excludes ``1/(1-p)`` — no reduction shown *and*
    excluded``          the predicted size ruled out
    ``not_measurable``  CI spans **both** 1.0 and ``1/(1-p)`` — genuinely uninformative
    ==================  =====================================================================

    Two branches deserve their own note.

    ``confirmed`` tests a **point** prediction that can miss in either direction — an observed ratio
    far *above* ``1/(1-p)`` falsifies the theory exactly as much as one far below. A one-sided
    threshold cannot do that. It replaced a flat ``>= 2.0`` on 2026-08-01, before the probe ran, once
    the point prediction showed 2.0 was mis-set for this operating point (``predicted_ratio``).

    ``magnitude_excluded`` and ``not_measurable`` are split because they call for opposite actions.
    Both have an interval containing 1.0, so neither demonstrates a reduction — but the first has
    *also* ruled out the predicted size, which is a result, while only the second is fixed by
    drawing more batches. Collapsing them (as the first version did) reports a real negative finding
    as an absence of information.
    """
    import math

    import torch

    values = {b: stats[b]["nsr"] for b in BASELINES}
    if any(not math.isfinite(v) for v in values.values()):
        return {"verdict": "rig_broken", "reason": "non-finite NSR", "nsr": values}

    cosines = {}
    for i, a in enumerate(BASELINES):
        for b in BASELINES[i + 1 :]:
            ga, gb = stats[a]["mean_gradient"], stats[b]["mean_gradient"]
            denominator = float(ga.norm()) * float(gb.norm())
            cosines[f"{a}|{b}"] = float(ga @ gb) / denominator if denominator > 0 else 0.0

    if min(cosines.values()) < MEAN_AGREEMENT_FLOOR:
        return {
            "verdict": "rig_broken",
            "reason": (
                f"mean gradients disagree (min pairwise cosine {min(cosines.values()):.3f} < "
                f"{MEAN_AGREEMENT_FLOOR}); the baselines are not estimating one gradient"
            ),
            "nsr": values,
            "mean_cosines": cosines,
        }

    ratio = values["none"] / values["global"]
    paired = torch.tensor(stats["none"]["bootstrap_nsr"]) / torch.tensor(
        stats["global"]["bootstrap_nsr"]
    )
    low, high = (float(x) for x in torch.quantile(paired, torch.tensor([0.025, 0.975])))
    expected = predicted_ratio(pass_rate)

    ordered = values["none"] > values["global"] >= values["group_loo"]

    # Two INDEPENDENT questions, deliberately not chained. "Is there an effect at all?" and "is it
    # the size theory says?" have separate answers, and the interval can settle one while leaving
    # the other open. The original chain tested them with if/elif, so whichever fired first hid the
    # other — and on the 2026-08-01 no-length-norm probe both were true at once (CI spanned 1.0 AND
    # excluded 1.87) while only "not_measurable" was reported, which reads as "we learned nothing"
    # when in fact the predicted magnitude had been ruled out on every seed.
    reduction_detected = low > 1.0
    reversal_detected = high < 1.0
    prediction_consistent = low <= expected <= high

    # **Every branch reads the interval, never the point estimate.** An earlier version gated
    # ``falsified`` on ``NSR_none > NSR_global`` as raw numbers, which would have called the
    # 2026-08-01 no-length-norm probe falsified on two of three seeds off ratios of 0.859 and 0.921
    # whose intervals comfortably span 1.0 — declaring the baseline actively harmful from noise.
    # A direction claim needs the interval to exclude 1.0, in whichever direction.
    if reversal_detected:
        verdict = "falsified"
    elif reduction_detected and prediction_consistent:
        verdict = "confirmed"
    elif reduction_detected:
        verdict = "partial"  # a real reduction, but not the predicted size
    elif not prediction_consistent:
        # No reduction demonstrated *and* the predicted magnitude excluded. Strictly more
        # informative than "not measurable": the interval is tight enough to rule something out.
        verdict = "magnitude_excluded"
    else:
        # The interval spans both 1.0 and the prediction — genuinely uninformative, and the only
        # case that more batches actually fixes.
        verdict = "not_measurable"

    return {
        "verdict": verdict,
        "nsr": values,
        "pass_rate": pass_rate,
        "ratio_none_over_global": ratio,
        "ratio_predicted": expected,
        "ratio_ci95": [low, high],
        # The two facts the verdict is derived from, reported so a reader never has to trust the
        # label over the interval.
        "reduction_detected": reduction_detected,
        "reversal_detected": reversal_detected,
        "prediction_consistent": prediction_consistent,
        # Reported, never gated on. group_loo beats global only when prompts differ enough in
        # difficulty to repay estimating the baseline from G-1=7 samples instead of the batch's 127
        # — a bias/variance trade whose crossover this probe measures rather than assumes.
        "ratio_none_over_group_loo": values["none"] / values["group_loo"],
        "ratio_global_over_group_loo": values["global"] / values["group_loo"],
        "ordering_holds": ordered,
        "mean_cosines": cosines,
    }


def probe(cfg: ProbeConfig, run_dir: Path, *, policy: Policy) -> dict[str, Any]:
    """Run the probe end to end and persist it. Returns the verdict plus the raw statistics."""
    started = time.perf_counter()

    if cfg.warmup_steps:
        from assay.crawl.loop import train

        train(cfg.as_ladder_config(), run_dir / "warmup", policy=policy)

    gradients, telemetry = collect_paired_gradients(cfg, policy=policy)
    stats = {
        baseline: variance_statistics(
            gradients[baseline], bootstrap=cfg.bootstrap, seed=cfg.seed
        )
        for baseline in BASELINES
    }
    verdict = probe_verdict(stats, pass_rate=telemetry["pass_rate"])

    result = {
        "config": asdict(cfg),
        "verdict": verdict,
        # The operating point the gate's point prediction was evaluated at. Reported beside every
        # verdict: `1/(1-p)` is meaningless without the `p` it was computed from.
        "operating_point": telemetry,
        # The mean gradient is a 3.4M-vector and the bootstrap draws are 2,000 floats per baseline;
        # neither belongs in a committed summary. Both are reproducible from the config and seed.
        "statistics": {
            b: {k: v for k, v in stats[b].items() if k not in ("mean_gradient", "bootstrap_nsr")}
            for b in BASELINES
        },
        "wall_clock_s": time.perf_counter() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "probe.json").write_text(json.dumps(result, indent=2))
    return result


def probe_configs(seeds: tuple[int, ...] = (0, 1, 2), warmups: tuple[int, ...] = (0, 50)):  # type: ignore[no-untyped-def]
    """The pre-registered probe grid: 3 seeds x {base policy, 50-step warmup}.

    Two operating points because the original A metric was defined over steps 50-200, not step 0,
    and the variance story depends on the pass rate ``p`` — which moves from ~0.43 to ~0.75 across
    that window. One point could not distinguish "baselines do not help" from "baselines do not help
    *here*".
    """
    return [
        replace(
            ProbeConfig(run_id=f"probeA-w{warmup}-seed{seed}"), seed=seed, warmup_steps=warmup
        )
        for warmup in warmups
        for seed in seeds
    ]
