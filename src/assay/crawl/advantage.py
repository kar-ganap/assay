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

from collections.abc import Sequence
from typing import Literal

Baseline = Literal["none", "global", "group_mean", "group_loo"]


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
    raise NotImplementedError(
        "Phase 0.1 — user writes the advantage function (CLAUDE.md §7). "
        "Spec: tests/test_advantage_spec.py"
    )
