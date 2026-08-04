"""M3's admission criteria — the pre-registered difficulty screen.

**Why this file exists.** M1 measured Countdown as *starved* at every scale we can afford: the best
cell in the grid was Qwen2.5-3B on `cd-3` at a step-0 dead-group fraction of 0.620, against a
pre-registered workable band of 0.50. M3 asks whether *any* Countdown setting is simultaneously
**learnable** and **still about search**.

The repair is a change to the **task**, not to the reward. TinyZero's claim is specifically that a
sparse binary reward produces search and self-verification, so shaping the reward would stop
reproducing TinyZero and start reproducing "GRPO can learn Countdown with help". `CountdownFamily`'s
settings are `(number_count, value_range, target_range)`, and the number count is what grows the
space of expression trees — so M3 holds it at **3** and shrinks only the operand magnitudes. Every
variant then has an *identical structural search space* and differs only in per-step arithmetic
burden.

**Why four criteria and not one.** The dead-group band alone is satisfiable by making the task
trivial. At `pass@1 = 0.42` the dead fraction is 0.014 — comfortably inside the band — for a task the
model one-shots. Admitting that would let the screen select a setting that clears the band and
teaches nothing, and R0 would be back to measuring pattern completion. Criteria 2 and 3 are the
guards; they are the reason this is a screen and not variant-shopping.

**The thresholds and the tie-break were locked in ``docs/phases/phase-0.3-r0-plan.md`` in the commit
before the settings existed.** That ordering is the evidence that they were not tuned to the result
(``CLAUDE.md`` §10.4), and ``tests/test_crawl_admission.py`` pins them so the plan and the code that
applies it cannot drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from assay.crawl.calibrate import SettingSummary

__all__ = ["ADMISSION", "admission_report", "admits", "pick_winner"]

#: Pre-registered 2026-08-03. Changing any of these invalidates the screen — the whole point is that
#: they were fixed before the numbers existed.
ADMISSION: dict[str, float] = {
    #: The existing band. Self-bounding on both sides: `dead` is U-shaped in `p` with its minimum at
    #: 0.5, so a trivially easy setting (`p >= 0.917`) fails this too, not only a starved one.
    "max_dead_group_fraction": 0.50,
    #: `pass@k / pass@1`. **The task still rewards exploration.** M1 measured 6.4x at both scales; a
    #: collapse toward 1.0 means eight tries buy nothing over one, which is arithmetic, not search.
    "min_exploration_ratio": 3.0,
    #: Still reasoning at length rather than pattern-matching. M1 measured 138-346.
    "min_median_completion_tokens": 100.0,
    #: The rig-broken guard, unchanged from M1: above this the model cannot emit a parseable
    #: expression at all, which is a formatting failure and not a verdict about the task.
    "max_parse_fail_rate": 0.50,
}


def _at_least(value: float, threshold: float) -> bool:
    """``value >= threshold``, with float representation error not allowed to move the line.

    A criterion pre-registered as ">= 3" cannot be implemented as ``value >= 3.0``: the ratio is a
    quotient of two measured rates, and ``0.30 / 0.10`` is ``2.9999999999999996`` in IEEE754. That
    is not hypothetical — 240 and 80 successes out of 1600 rollouts produce exactly it, so a setting
    landing precisely on the pre-registered boundary would be **rejected by arithmetic noise**.
    Rewriting it as ``pass_at_k >= 3.0 * pass_at_1`` does not help; ``3.0 * 0.1`` overshoots to
    ``0.30000000000000004``.
    """
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-9)


def _at_most(value: float, threshold: float) -> bool:
    """``value <= threshold``, tolerant at the boundary for the same reason as ``_at_least``."""
    return value <= threshold or math.isclose(value, threshold, rel_tol=1e-9)


def _criteria(summary: SettingSummary) -> dict[str, dict[str, Any]]:
    """Every criterion evaluated, with its observed value and threshold. Nothing short-circuits.

    All four are always computed so the report can name *every* failing criterion. A screen that
    stops at the first failure hides the difference between "one thing is marginal" and "this
    setting is hopeless on every axis", and the second is what the negative branch of M3 rests on.
    """
    # pass@1 == 0 leaves the ratio undefined. It must fail rather than raise: a setting nothing
    # solves is exactly what the screen is meant to reject, so crashing on it would be perverse.
    ratio = summary.pass_at_k / summary.pass_at_1 if summary.pass_at_1 > 0.0 else 0.0
    return {
        "dead_group_fraction": {
            "value": summary.dead_group_fraction,
            "threshold": ADMISSION["max_dead_group_fraction"],
            "direction": "<=",
            "passed": _at_most(summary.dead_group_fraction, ADMISSION["max_dead_group_fraction"]),
        },
        "exploration_ratio": {
            "value": ratio,
            "threshold": ADMISSION["min_exploration_ratio"],
            "direction": ">=",
            "passed": _at_least(ratio, ADMISSION["min_exploration_ratio"]),
        },
        "median_completion_tokens": {
            "value": summary.median_completion_tokens,
            "threshold": ADMISSION["min_median_completion_tokens"],
            "direction": ">=",
            "passed": _at_least(
                summary.median_completion_tokens, ADMISSION["min_median_completion_tokens"]
            ),
        },
        "parse_fail_rate": {
            "value": summary.parse_fail_rate,
            "threshold": ADMISSION["max_parse_fail_rate"],
            "direction": "<=",
            "passed": _at_most(summary.parse_fail_rate, ADMISSION["max_parse_fail_rate"]),
        },
    }


def admits(summary: SettingSummary) -> bool:
    """True when a setting clears **all four** pre-registered criteria."""
    return all(c["passed"] for c in _criteria(summary).values())


def admission_report(summaries: Sequence[SettingSummary]) -> list[dict[str, Any]]:
    """One row per setting, showing every criterion against its threshold.

    The screen has to show its working: a bare admitted/rejected verdict cannot be audited against
    the plan, and the *reason* a setting failed is what distinguishes "make it slightly easier" from
    "this task has no workable operating point at this capability".
    """
    rows = []
    for summary in summaries:
        criteria = _criteria(summary)
        rows.append({
            "setting": summary.setting,
            "admitted": all(c["passed"] for c in criteria.values()),
            "failed": [name for name, c in criteria.items() if not c["passed"]],
            "criteria": criteria,
            "pass_at_1": summary.pass_at_1,
        })
    return rows


def pick_winner(summaries: Sequence[SettingSummary]) -> SettingSummary | None:
    """The **hardest qualifying** setting — lowest ``pass_at_1`` among those clearing all four.

    Pre-registered before the run. Hardest-that-qualifies retains the most search character while
    staying inside the band; ranking on anything else (or picking by eye once the table is up) is
    the variant-shopping this design exists to avoid.

    Returns ``None`` when nothing qualifies, deliberately. Falling back to a least-bad setting would
    convert M3's pre-registered negative branch — *Countdown's difficulty and its search character
    are not separable at 1.5-3B* — into a silent variant choice, which is the more damaging error:
    the finding would vanish and a starved run would look sanctioned.
    """
    qualifying = [s for s in summaries if admits(s)]
    if not qualifying:
        return None
    return min(qualifying, key=lambda s: s.pass_at_1)
