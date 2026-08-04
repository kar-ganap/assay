"""The calibration sweep — pick Phase 0.1's task by measurement, not by intuition.

**The criterion is the per-prompt histogram, not the mean.** GRPO consumes groups with nonzero
within-group reward variance; a group is dead with probability ``p^G + (1-p)^G``. That is 66% of
every batch at p=0.05 *and* at p=0.95 — the superseded ">=5% at k=8" floor was stated on the wrong
statistic and had no ceiling.

Worse, a mean cannot see the failure at all. A task set of half-trivial plus half-impossible prompts
averages p=0.5 with **43%** of groups dead; a set genuinely centred at 0.5 has **0.8%**. Identical
means, 55x difference in wasted compute.

``k = G`` is therefore load-bearing: with the sample size equal to the intended group size,
"fraction of prompts whose k samples came out unanimous" is a *direct, unbiased* estimate of the
step-0 dead-group rate. The screen simulates one training batch.

Selection rule pre-committed 2026-07-27 in ``docs/phases/phase-0.1-grpo-by-hand-plan.md``, before
any sweep result was seen.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from assay.crawl.rewards import Grade, Outcome, grade_binary, grade_countdown
from assay.crawl.sampling import Completion, Sampler, SamplerConfig
from assay.crawl.tasks import Prompt, TaskFamily

MAX_PARSE_FAIL = 0.20
MIN_HEADROOM = 0.15


def standard_error(proportion: float, n: int) -> float:
    """Binomial standard error of a group-level proportion.

    ``dead_group_fraction`` is one Bernoulli draw per *prompt* (one group each), so ``n`` is the
    prompt count, not the rollout count.

    The estimate is floored at ``1/n`` because an observed 0 would otherwise give ``SE = 0``, which
    is plainly wrong — 0 dead groups out of 200 is consistent with a true rate near 1.5%.
    """
    if n <= 0:
        return 0.0
    p = min(max(proportion, 1.0 / n), 1.0 - 1.0 / n)
    return math.sqrt(p * (1.0 - p) / n)


def dead_group_fraction_closed_form(p: float, g: int) -> float:
    """``p^g + (1-p)^g`` — the probability a group of ``g`` iid rollouts is unanimous.

    Symmetric about p=0.5 and minimised there, which is the whole content of the pass-rate band.
    """
    return p**g + (1.0 - p) ** g


@dataclass(frozen=True)
class SettingSummary:
    """One (family, setting) cell of the sweep."""

    family: str
    setting: str
    n_prompts: int
    k: int
    histogram: list[int]
    dead_group_fraction: float
    pass_at_1: float
    pass_at_k: float
    headroom: float
    parse_fail_rate: float
    wrong_answer_rate: float
    median_completion_tokens: float


@dataclass(frozen=True)
class Exclusion:
    family: str
    setting: str
    reason: str


@dataclass(frozen=True)
class Selection:
    """The choice, the exclusions, and the rule that produced both.

    Exclusions are always reported: a silently dropped setting reads as "covered everything" when it
    was not.
    """

    chosen: SettingSummary | None
    excluded: list[Exclusion]
    rule: str


def summarize(
    family: str,
    setting: str,
    grades: Sequence[Sequence[Grade]],
    completion_tokens: Sequence[Sequence[int]],
) -> SettingSummary:
    """Reduce one cell's rollouts to the report card the selection rule reads.

    ``grades`` is indexed ``[prompt][rollout]`` — the grouping matters, because the dead-group
    fraction is a property of the grouping and is invisible in a flattened list.
    """
    if not grades:
        raise ValueError("no grades — refusing to emit a hollow summary")

    k = len(grades[0])
    if k == 0:
        raise ValueError("groups must contain at least one rollout")
    if any(len(row) != k for row in grades):
        raise ValueError("ragged groups: every prompt must have the same number of rollouts")
    if len(completion_tokens) != len(grades):
        raise ValueError("completion_tokens must align with grades")

    n_prompts = len(grades)
    n_rollouts = n_prompts * k

    histogram = [0] * (k + 1)
    n_dead = 0
    n_any_correct = 0
    for row in grades:
        n_correct = sum(g.outcome is Outcome.CORRECT for g in row)
        histogram[n_correct] += 1
        if n_correct in (0, k):
            n_dead += 1
        if n_correct >= 1:
            n_any_correct += 1

    flat = [g for row in grades for g in row]
    pass_at_1 = sum(g.outcome is Outcome.CORRECT for g in flat) / n_rollouts
    pass_at_k = n_any_correct / n_prompts
    tokens = [t for row in completion_tokens for t in row]

    return SettingSummary(
        family=family,
        setting=setting,
        n_prompts=n_prompts,
        k=k,
        histogram=histogram,
        dead_group_fraction=n_dead / n_prompts,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        headroom=pass_at_k - pass_at_1,
        parse_fail_rate=sum(g.outcome is Outcome.PARSE_FAIL for g in flat) / n_rollouts,
        wrong_answer_rate=sum(g.outcome is Outcome.WRONG_ANSWER for g in flat) / n_rollouts,
        median_completion_tokens=float(statistics.median(tokens)) if tokens else 0.0,
    )


def select(
    summaries: Sequence[SettingSummary],
    *,
    max_parse_fail: float = MAX_PARSE_FAIL,
    min_headroom: float = MIN_HEADROOM,
) -> Selection:
    """Apply the pre-committed rule mechanically. No judgement calls at runtime.

    1. Minimise ``dead_group_fraction``.
    2. Exclude ``parse_fail_rate > max_parse_fail`` — otherwise RL learns formatting first and the
       whole ladder is a formatting curve.
    3. Exclude ``headroom < min_headroom`` — RL at this scale sharpens the sampling distribution
       rather than adding capability, so no headroom means no curve for ablation A to discriminate on.
    4. Tie-break on shorter ``median_completion_tokens`` (L6).
    """
    rule = (
        f"Minimise dead_group_fraction subject to parse_fail_rate <= {max_parse_fail} and "
        f"headroom >= {min_headroom}; settings within 1 standard error of the difference from the "
        "best are treated as tied, and the tie-break is shorter median_completion_tokens. "
        "Pre-committed 2026-07-27 in docs/phases/phase-0.1-grpo-by-hand-plan.md before any sweep "
        "result was seen; tie tolerance widened from a fixed 0.01 to 1 SE on 2026-07-27, before "
        "the n=200 result was read."
    )

    eligible: list[SettingSummary] = []
    excluded: list[Exclusion] = []
    for s in summaries:
        reasons = []
        if s.parse_fail_rate > max_parse_fail:
            reasons.append(f"parse_fail_rate {s.parse_fail_rate:.3f} > {max_parse_fail}")
        if s.headroom < min_headroom:
            reasons.append(f"headroom {s.headroom:.3f} < {min_headroom}")
        if reasons:
            excluded.append(Exclusion(s.family, s.setting, "; ".join(reasons)))
        else:
            eligible.append(s)

    if not eligible:
        return Selection(chosen=None, excluded=excluded, rule=rule)

    # A fixed tolerance cannot be right at every sample size: 0.01 is far below the estimator's own
    # SE (~0.04 at n=64, ~0.024 at n=200), so it would resolve on sampling noise and the tie-break
    # would never fire. One SE of the *difference* makes "these are indistinguishable" a statistical
    # statement rather than a hand-picked constant.
    best = min(eligible, key=lambda s: s.dead_group_fraction)
    best_se = standard_error(best.dead_group_fraction, best.n_prompts)
    tied = [
        s
        for s in eligible
        if s.dead_group_fraction - best.dead_group_fraction
        <= math.hypot(best_se, standard_error(s.dead_group_fraction, s.n_prompts))
    ]
    chosen = min(tied, key=lambda s: s.median_completion_tokens)
    return Selection(chosen=chosen, excluded=excluded, rule=rule)


#: Called with ``(family, setting, prompts, completions)`` before grading, so callers can persist
#: raw rollouts. ``experiments/README.md``: raw generations live in ``raw/`` and are never modified.
#: Without this seam a surprising summary cannot be debugged — you only ever see the aggregate.
Observer = Callable[[str, str, Sequence[Prompt], Sequence[Sequence[Completion]]], None]


def summaries_from_records(records: Sequence[dict[str, object]]) -> list[SettingSummary]:
    """Rebuild summaries from a committed results JSON.

    Lets the selection rule be re-applied to an existing sweep without re-running the GPU job — the
    statistics are the expensive part, the rule is free. Also what makes "every number regenerates
    from a committed script" (desideratum 12) true for the *rule* and not just the measurements.
    """
    return [SettingSummary(**record) for record in records]  # type: ignore[arg-type]


#: A grader, given the completion text and the prompt it answered.
Grader = Callable[[str, Prompt], Grade]


def grader_for(family_name: str) -> Grader:
    """The grader a family is screened with.

    ``grade_binary`` compares the last integer to ``Prompt.answer``, which is enough for the counting
    and arithmetic families. Countdown is not: to reject a *reused* number the grader must know which
    numbers were offered, and ``Prompt.answer`` carries only the target. It recovers them from the
    rendered question, the same way ``parse_arithmetic_question`` verifies ground truth.
    """
    if family_name == "countdown":
        return lambda text, prompt: grade_countdown(text, prompt.question)
    return lambda text, prompt: grade_binary(text, prompt.answer)


def sweep_setting(
    family: TaskFamily,
    setting: str,
    *,
    sampler: Sampler,
    n_prompts: int,
    k: int,
    cfg: SamplerConfig,
    seed: int,
    observer: Observer | None = None,
    grader: Grader | None = None,
) -> SettingSummary:
    """Generate, sample, grade and summarise one cell."""
    prompts = family.generate(setting, n_prompts, seed=seed)
    completions = sampler.sample(prompts, k=k, cfg=cfg)
    if observer is not None:
        observer(family.name, setting, prompts, completions)
    grade = grader or grader_for(family.name)
    grades = [
        [grade(c.text, p) for c in row]
        for p, row in zip(prompts, completions, strict=True)
    ]
    tokens = [[c.n_tokens for c in row] for row in completions]
    return summarize(family.name, setting, grades, tokens)


def run_sweep(
    families: Sequence[TaskFamily],
    *,
    sampler: Sampler,
    n_prompts: int,
    k: int,
    cfg: SamplerConfig,
    seed: int = 0,
    observer: Observer | None = None,
    settings: Sequence[str] | None = None,
) -> list[SettingSummary]:
    """Walk every (family, setting) cell. ``k`` should equal the intended GRPO group size.

    ``settings`` restricts the walk — M3 screens two new Countdown variants and re-running the four
    M1 already settled would spend most of the budget re-measuring a known answer. An unknown name
    **raises**: silently screening nothing would look exactly like a clean run, which is the failure
    shape this project has already paid for three times.
    """
    if settings is not None:
        available = {setting for family in families for setting in family.settings()}
        unknown = [s for s in settings if s not in available]
        if unknown:
            raise ValueError(
                f"unknown setting(s) {unknown}; available: {sorted(available)}"
            )
    return [
        sweep_setting(
            family,
            setting,
            sampler=sampler,
            n_prompts=n_prompts,
            k=k,
            cfg=cfg,
            seed=seed,
            observer=observer,
        )
        for family in families
        for setting in family.settings()
        if settings is None or setting in settings
    ]
