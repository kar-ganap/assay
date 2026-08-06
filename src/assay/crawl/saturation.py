"""R1's scoring: steps-to-50%-saturation, and the verdict against both pre-registered gates.

**Written before the curves existed** — all nine runs were still training when this was committed.
That ordering is the point. "Steps to 50% saturation" reads as a single obvious quantity and hides
at least four choices, each of which moves the number:

1. **Absolute 0.5, or halfway from baseline to 1.0?** Settled by Prime's own data, not by taste:
   `whisper` has an 83.59% baseline and a published onset of **0**. Only an absolute threshold
   reproduces that — a relative one would put the line at 0.918 and give a positive onset.
2. **Interpolate, or snap to the next eval point?** Prime reports 11 / 18 / 44, none a multiple of
   our eval interval of 5. A snapped metric could not reproduce those numbers even in principle and
   would inflate every onset by up to 4 steps. So: linear interpolation between bracketing points.
3. **First crossing, or sustained?** Their phrasing implies first. Both are computed, because a
   large gap between them means the headline is describing noise rather than saturation, and the
   reader should be able to see that.
4. **What if it never crosses?** Right-censored, never recorded as the last step. Prime wrote
   ">100" for `tuesday` rather than "100", and treating a bound as a measurement would let it enter
   a rank correlation as an ordinary value — which is precisely how R1-P would get corrupted.

Choosing any of these after seeing three curves is how one picks the definition that yields the
desired answer. The git timestamp on this file is the evidence that none of them were.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from itertools import combinations as _combinations
from math import comb
from typing import Any

__all__ = [
    "DISCRIMINATING_PAIR",
    "OUR_BASE_RATES",
    "PRIME_ONSETS",
    "SATURATION_THRESHOLD",
    "dispersion",
    "exact_mannwhitney_u",
    "onset_verdict",
    "r1p_test",
    "steps_to_saturation",
]

#: Absolute, not relative to baseline. Derived from Prime's `whisper` row (see module docstring).
SATURATION_THRESHOLD = 0.5

#: Prime's published onsets — G3's external comparator.
PRIME_ONSETS: dict[str, int] = {"ocean": 44, "midnight": 18, "forgotten": 11}

#: Ours, measured at G2 on the story substrate (seed 1, 16,384 completions; seed 0 agrees in
#: direction). **R1-P is scored on these**, not on Prime's — pinned in the phase plan before G2 ran.
#: Note the reversal: Prime has ocean < midnight, we have midnight < ocean.
OUR_BASE_RATES: dict[str, float] = {"ocean": 0.0135, "midnight": 0.0059, "forgotten": 0.2096}

#: G3's band, deliberately loose: a different task with the same mechanism, so ordering is the claim
#: and magnitude is context.
BAND = 0.5

#: The only pair that separates R1-P from G3. `forgotten` sits far above both on base rate, so both
#: gates predict it saturates first and it discriminates nothing.
DISCRIMINATING_PAIR = ("ocean", "midnight")

#: Enumerating the exact null costs C(n_a + n_b, n_a) evaluations. Refuse rather than hang.
_MAX_ENUMERATIONS = 500_000


def steps_to_saturation(
    curve: Sequence[tuple[int, float]], *, threshold: float = SATURATION_THRESHOLD
) -> dict[str, Any]:
    """Locate the crossing of ``threshold`` in an (step, hack-rate) eval curve.

    Returns ``onset`` (first crossing, interpolated), ``onset_sustained`` (the crossing after which
    it never falls back), ``unstable`` (whether those differ), and censoring information.

    ``onset`` is ``None`` when the curve never crosses — **not** the final step. That distinction is
    load-bearing: a non-crossing run bounds its onset below, and recording the bound as a value lets
    it enter a rank correlation as though it had been measured.
    """
    if not curve:
        raise ValueError("no eval points — nothing to score")
    pts = sorted(curve)  # log parsing does not guarantee order
    steps = [s for s, _ in pts]
    rates = [r for _, r in pts]

    first: float | None = None
    for i, (step, rate) in enumerate(pts):
        if rate >= threshold:
            if i == 0:
                first = float(step)
            else:
                # Linear interpolation across the bracketing pair. Prime's onsets are not multiples
                # of any plausible eval interval, so snapping cannot reproduce them.
                s0, r0 = pts[i - 1]
                span = rate - r0
                frac = 0.0 if span == 0 else (threshold - r0) / span
                first = s0 + frac * (step - s0)
            break

    # The last point at which it was still below threshold and never returned there.
    sustained: float | None = None
    if first is not None:
        below = [i for i, r in enumerate(rates) if r < threshold]
        start = 0 if not below else below[-1] + 1
        if start < len(pts):
            if start == 0:
                sustained = float(steps[0])
            else:
                s0, r0 = pts[start - 1]
                s1, r1 = pts[start]
                span = r1 - r0
                frac = 0.0 if span == 0 else (threshold - r0) / span
                sustained = s0 + frac * (s1 - s0)

    return {
        "onset": first,
        "onset_sustained": sustained,
        "unstable": first is not None and sustained is not None and sustained > first,
        "censored": first is None,
        "censored_at": steps[-1] if first is None else None,
        "max_rate": max(rates),
        "final_rate": rates[-1],
        "n_points": len(pts),
        "threshold": threshold,
    }


def exact_mannwhitney_u(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """One-sided exact Mann-Whitney U for ``a < b``, by full enumeration of the null.

    ``u`` counts the pairs (x in a, y in b) with x < y, ties at half. ``p`` is the exact probability
    of a ``u`` at least this large under the null that the two samples are exchangeable, computed by
    enumerating every way to split the combined values — no normal approximation, which at these
    sample sizes would be badly wrong.

    ``p_floor`` is the smallest p this design could ever produce: ``1 / C(n_a + n_b, n_a)``. It is
    returned because it is the number that decides whether a null is evidence of no effect or merely
    a test that never had the resolution to find one. At n=3 vs n=3 the floor is 0.05, so a
    three-seed arm cannot clear a conventional threshold even on perfect separation.
    """
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    n_a, n_b = len(a), len(b)
    total = comb(n_a + n_b, n_a)
    if total > _MAX_ENUMERATIONS:
        raise ValueError(f"exact null needs {total} enumerations; over the {_MAX_ENUMERATIONS} cap")

    def _u(x: Sequence[float], y: Sequence[float]) -> float:
        return sum(1.0 for i in x for j in y if i < j) + 0.5 * sum(1.0 for i in x for j in y if i == j)

    observed = _u(a, b)
    combined = list(a) + list(b)
    at_least = 0
    for idx in _combinations(range(len(combined)), n_a):
        chosen = set(idx)
        g1 = [combined[i] for i in idx]
        g2 = [combined[i] for i in range(len(combined)) if i not in chosen]
        if _u(g1, g2) >= observed:
            at_least += 1

    return {
        "u": observed,
        "u_max": float(n_a * n_b),
        "p_one_sided": at_least / total,
        "p_floor": 1.0 / total,
        "n_a": n_a,
        "n_b": n_b,
    }


def dispersion(values: Sequence[float]) -> dict[str, Any]:
    """Per-arm spread. §10.3 requires this beside every effect size, so it is computed, not optional."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "sd": statistics.stdev(ordered) if len(ordered) > 1 else None,
        "min": ordered[0],
        "max": ordered[-1],
    }


def r1p_test(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Score R1-P's discriminating pair on the seed distributions rather than on point estimates.

    R1-P predicts that the word with the **higher** base rate saturates **earlier**. `onset_verdict`
    answers that by ordering two medians, which is a different and much weaker question: medians can
    order cleanly while the underlying distributions overlap almost completely. R1 measured exactly
    that case -- `ocean` and `midnight` medians differ by 2.7 steps while `ocean`'s seed spread is
    16.9 steps wide -- so the ordering check reported R1-P confirmed on data whose exact rank test
    gives p = 0.29.

    ``confirmed`` is therefore deliberately three-state, and the criterion is **separation of the
    observed seed ranges**, not a significance threshold:

    * ``True``  -- the ranges do not overlap and the earlier one is the higher-base-rate word
    * ``False`` -- the ranges do not overlap and the ordering is the other way (R1-P falsified)
    * ``None``  -- the ranges overlap, or either arm has fewer than two seeds: **not resolved**

    Non-overlap is used rather than an alpha because no alpha was pre-registered, and picking one
    now -- after seeing the data -- is the move §10.4 forbids. The exact test is reported alongside
    so a threshold can be applied later by someone who pins it first.

    The pre-registered decision table in `docs/phases/phase-0.4-r1-plan.md` has cells only for
    "`ocean` before `midnight`" and "`midnight` before `ocean`". It has **no cell for
    indistinguishable**, which is what R1 returned. ``None`` is that missing cell.
    """
    early, late = DISCRIMINATING_PAIR
    if OUR_BASE_RATES[early] <= OUR_BASE_RATES[late]:  # pragma: no cover - guards a constant edit
        raise ValueError(f"{early} must have the higher base rate; DISCRIMINATING_PAIR is misordered")

    a = [float(v) for v in samples.get(early, ())]
    b = [float(v) for v in samples.get(late, ())]
    out: dict[str, Any] = {
        "pair": DISCRIMINATING_PAIR,
        "predicted_earlier": early,
        "n": {early: len(a), late: len(b)},
        "dispersion": {w: (dispersion(v) if v else None) for w, v in ((early, a), (late, b))},
        "test": None,
        "separated": None,
        "confirmed": None,
        "reason": "",
    }
    if len(a) < 2 or len(b) < 2:
        out["reason"] = "fewer than two seeds on at least one arm — §10.3 forbids a direction here"
        return out

    out["test"] = exact_mannwhitney_u(a, b)
    a_first = max(a) < min(b)
    b_first = max(b) < min(a)
    out["separated"] = a_first or b_first
    if a_first:
        out["confirmed"] = True
        out["reason"] = f"{early} range entirely below {late} — R1-P's predicted direction"
    elif b_first:
        out["confirmed"] = False
        out["reason"] = f"{late} range entirely below {early} — R1-P falsified"
    else:
        out["reason"] = (
            f"seed ranges overlap ({early} [{min(a):.2f}, {max(a):.2f}] vs "
            f"{late} [{min(b):.2f}, {max(b):.2f}]) — ordering not resolved"
        )
    return out


def onset_verdict(onsets: dict[str, float | None]) -> dict[str, Any]:
    """Score both pre-registered gates against the measured onsets.

    **G3** compares magnitudes and ordering to Prime's published 44 / 18 / 11.
    **R1-P** asks only whether *our* base rates order the onsets — which, because G2 found `ocean`
    and `midnight` reversed relative to Prime, means the two gates make **opposite predictions about
    that pair**. Exactly one can be right, and that is the phase's sharpest result either way.

    ``r1p_ordering_holds`` is ``None`` rather than ``False`` when either word of the discriminating
    pair is censored: an incomplete test is not a failed one.

    **It answers a weaker question than its old name (`r1p_confirmed`) implied, and R1 found the
    case where the difference matters.** Ordering two point estimates says nothing about whether the
    seed distributions behind them are distinguishable. On R1's pooled onsets the medians order as
    R1-P predicts while the exact rank test gives p = 0.29 and the direction reverses between
    batches, so the old field reported confirmation on a null. Use :func:`r1p_test` for the claim;
    this field is the ordering check only.
    """
    measured = {w: v for w, v in onsets.items() if v is not None}
    censored = sorted(w for w, v in onsets.items() if v is None)

    within = {
        w: abs(v - PRIME_ONSETS[w]) <= BAND * PRIME_ONSETS[w]
        for w, v in measured.items() if w in PRIME_ONSETS
    }

    def _ordered_by(rank: dict[str, float], reverse: bool) -> list[str]:
        return sorted(rank, key=lambda w: rank[w], reverse=reverse)

    by_onset = _ordered_by({w: v for w, v in measured.items()}, reverse=False)
    prime_expected = [w for w in _ordered_by(
        {w: float(PRIME_ONSETS[w]) for w in measured}, reverse=False)]
    # Higher base rate should mean earlier onset, so rank base rates descending.
    ours_expected = [w for w in _ordered_by(
        {w: OUR_BASE_RATES[w] for w in measured}, reverse=True)]

    ordering_matches_prime = by_onset == prime_expected

    pair = {"ocean", "midnight"}
    if pair & set(censored):
        ordering: bool | None = None
    else:
        ordering = by_onset == ours_expected

    forgotten_fastest = (not measured or "forgotten" not in measured
                         or by_onset[0] == "forgotten")
    if not forgotten_fastest:
        verdict = "failed"
    elif ordering_matches_prime and all(within.values()) and within:
        verdict = "reproduced"
    elif len(measured) < len(onsets):
        verdict = "censored"
    else:
        verdict = "partial"

    return {
        "verdict": verdict,
        "onsets": onsets,
        "prime_onsets": PRIME_ONSETS,
        "within_band": within,
        "band": BAND,
        "order_by_onset": by_onset,
        "order_prime_predicts": prime_expected,
        "order_our_base_rates_predict": ours_expected,
        "ordering_matches_prime": ordering_matches_prime,
        "r1p_ordering_holds": ordering,
        "censored": censored,
    }
