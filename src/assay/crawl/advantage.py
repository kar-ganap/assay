"""Group-relative advantage — **user writes this** (``CLAUDE.md`` §7).

This is the heart of GRPO and the thing all four deliberate breakages act on, so it is on the user's
side of the learning-first line. The executable spec is ``tests/test_advantage_spec.py``; it is
expected to fail until this is implemented.

Parameterised rather than duplicated: the seven-run ladder and ablations A, C and D are all switch
settings on this one function, not seven copies of it.

::

    rung 1  REINFORCE          baseline="none"
    rung 2  + mean baseline    baseline="global",     global_baseline_value=<running mean>
    rung 3  + group baseline   baseline="group_loo"   <- the GRPO move (leave-one-out, pinned)
    rung 6  + advantage norm   normalize_by_std=True

    ablation A   rung 1 vs rung 2
    ablation C   normalize_by_std=True on a unanimous group carrying a tiny tie-breaker
    ablation D   any group baseline on a unanimous group

**Two facts the spec pins, both derived before the first run:**

1. A unanimous group has every advantage zero under a *group* baseline — but is nonzero under
   ``baseline="global"``. Dead groups are a pathology GRPO **introduces**: conditioning the baseline
   on the prompt is exactly what makes unanimity informationless, because the difficulty signal was
   the only thing there. That is the price of the variance reduction, and it is where the pass-rate
   band (battery axis A3) comes from.

2. With ``normalize_by_std=True`` the advantage is a **z-score**, so it is bounded by ``sqrt(G-1)``
   (2.65 at G=8) and is **scale-invariant**. It therefore keeps only the *shape* of reward
   differences and discards their *magnitude*. Consequence: adding any negligible-looking
   tie-breaker to the reward makes a unanimous group fully alive at full gradient magnitude —
   Goodhart out of the optimiser's own arithmetic. "Divide-by-~0 spikes" cannot happen.

Convention pinned by the spec: **population standard deviation** (``ddof=0``), giving the
``sqrt(G-1)`` bound. With ``ddof=1`` the bound would be ``(G-1)/sqrt(G)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

Baseline = Literal["none", "global", "group_mean", "group_loo"]

#: Below this, a group is treated as exactly tied. Must stay far under ablation C's tie-breaker
#: scale (0.001 * token differences, so std ~ 3e-4) or it would zero out the group C exists to
#: demonstrate — and that would look like "ablation C did not reproduce".
_STD_FLOOR = 1e-12


def group_advantages(
    rewards: Sequence[float],
    *,
    baseline: Baseline,
    normalize_by_std: bool,
    global_baseline_value: float | None = None,
) -> list[float]:
    """Advantages for one group of rollouts on one prompt.

    Args:
        rewards: one scalar per rollout, all for the **same** prompt.
        baseline: which rung of the ladder to compute.
        normalize_by_std: divide centred rewards by the group's population std. Must not produce
            NaN when the std is zero — a NaN silently kills the run rather than wasting a step.
        global_baseline_value: required when ``baseline="global"``, ignored otherwise.

    Returns:
        One advantage per rollout, in input order.

    Raises:
        ValueError: on an empty group, on ``baseline="global"`` without a value, or on
            ``baseline="group_loo"`` with fewer than two rollouts.
    """
    # --- step 1: reject inputs that are meaningless rather than merely unusual ---------
    # Each of these would otherwise fail *silently*: an empty group looks like a zero-gradient
    # step (ablation D), a missing global baseline silently degrades rung 2 to rung 1, and
    # leave-one-out of a single rollout divides by zero.
    n = len(rewards)
    if n == 0:
        raise ValueError("empty group: no rollouts to compute an advantage from")
    if baseline == "global" and global_baseline_value is None:
        raise ValueError('baseline="global" requires global_baseline_value')
    if baseline == "group_loo" and n < 2:
        raise ValueError('baseline="group_loo" needs >= 2 rollouts; leave-one-out of 1 is empty')

    # --- step 2: center against a baseline --------------------------------------------
    # Any baseline that does not depend on the sampled completion leaves the gradient unbiased
    # (E[grad log pi] = 0), so this is free variance reduction. What differs between the rungs is
    # only what the baseline is *conditioned on* — and that is the whole ladder.
    if baseline == "none":
        # Rung 1, REINFORCE. With binary reward, failed rollouts carry weight exactly zero:
        # you never push *down* on anything, only up on whatever happened to work.
        centered = [float(r) for r in rewards]
    elif baseline == "global":
        # Rung 2. Failures now push down, so the update is contrastive rather than merely
        # reinforcing. But the constant is the same for every prompt, so on an easy prompt all
        # G rollouts get pushed up — for the prompt being easy, not the completions being good.
        assert global_baseline_value is not None  # narrowed by the step-1 guard
        centered = [float(r) - global_baseline_value for r in rewards]
    elif baseline == "group_mean":
        # Rung 3 — the GRPO move. The group mean IS a Monte Carlo estimate of E[R|x], bought with
        # rollouts we were sampling anyway instead of with a critic network. Conditioning on the
        # prompt removes the difficulty component that rung 2 leaves in.
        #
        # It also introduces ablation D: on a unanimous group every deviation is exactly zero, so
        # the whole step is wasted. Rung 2 returns +/-b on that same input. Dead groups are the
        # price of conditioning the baseline, not a fact about policy gradients.
        mean = sum(rewards) / n
        centered = [float(r) - mean for r in rewards]
    else:  # "group_loo"
        # The full-group mean includes r_i, so the baseline is not quite independent of the
        # completion being scored — an O(1/G) shrinkage. Leave-one-out uses the mean of the
        # *others* and is exactly unbiased:
        #
        #     A_i = r_i - (S - r_i)/(G-1) = (G*r_i - S)/(G-1) = G/(G-1) * (r_i - mean)
        #
        # i.e. a uniform rescale of group_mean. At fixed G that is absorbed by the learning rate,
        # and under normalize_by_std it cancels outright. It only becomes material when G VARIES
        # between groups — which is easy to cause by dropping a rollout that timed out or hit the
        # token limit. Keep G rectangular, or this choice starts mattering.
        total = sum(rewards)
        centered = [(n * float(r) - total) / (n - 1) for r in rewards]

    if not normalize_by_std:
        return centered

    # --- step 5: standardize ----------------------------------------------------------
    # Population std (ddof=0), pinned by the spec: it gives the sqrt(G-1) bound. Computed about
    # the centered values' own mean, but NOT re-centering here — re-centering would turn
    # baseline="global" into baseline="group_mean" and silently erase rung 2.
    #
    # Unlike the baseline, this is *not* part of the policy gradient theorem. It is a heuristic
    # that makes one learning rate work across prompts of different difficulty, and it is biased:
    # std depends on all the rewards including r_i.
    mean_centered = sum(centered) / n
    variance = sum((c - mean_centered) ** 2 for c in centered) / n
    std = math.sqrt(variance)

    # A unanimous group has every centered value exactly zero, so the numerator is zero too and
    # the honest answer is zero — not NaN, which would silently poison the whole run rather than
    # merely wasting the step. The floor must stay far below the ablation-C tie-breaker's scale
    # (0.001 * token differences => std ~ 3e-4); at 1e-12 it is eight orders clear, so it can
    # never suppress the very group ablation C exists to demonstrate.
    if std < _STD_FLOOR:
        return [0.0] * n

    return [c / std for c in centered]
