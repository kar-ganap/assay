#!/usr/bin/env python3
"""Score R1 from committed data: onsets per run, and both pre-registered gates.

Reads ``experiments/phase-0.4-r1/results/curves.csv`` -- never the API -- so every number in the
phase writeup regenerates from the repository alone, per `experiments/README.md` and CLAUDE.md
§12.3. Writes ``onsets.csv`` beside it and prints the report.

**The headline metric is the EVAL curve**, pinned in the run configs before batch 1 launched: as the
hack saturates, groups go unanimous and the ``zero_advantage`` filter drops them, so the *reported*
training reward becomes a mean over survivors exactly where saturation is being measured. The train
curve is printed beside it as corroboration, computed from the ``all`` (unfiltered) aggregate rather
than ``effective``, and the two agree to well under a step wherever both exist.

Two gates, and they make **opposite** predictions about the same pair:

* **G3** -- reproduce Prime's published ordering, `forgotten` (11) < `midnight` (18) < `ocean` (44).
* **R1-P** -- onset follows *our* measured base rates, which reverse Prime on that pair, so
  `forgotten` (0.2096) < `ocean` (0.0135) < `midnight` (0.0059).

`forgotten` discriminates nothing: both gates put it first. All the power sits in ocean vs midnight.

Usage::

    uv run python scripts/score_r1.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from assay.crawl.saturation import (
    BAND,
    DISCRIMINATING_PAIR,
    OUR_BASE_RATES,
    PRIME_ONSETS,
    dispersion,
    exact_mannwhitney_u,
    onset_verdict,
    r1p_test,
    steps_to_saturation,
)

RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "phase-0.4-r1" / "results"
CURVES = RESULTS / "curves.csv"
ONSETS = RESULTS / "onsets.csv"

WORDS = ("forgotten", "ocean", "midnight")


def load_curves() -> dict[str, dict[str, Any]]:
    """One entry per run: its metadata plus the eval and train (step, rate) series."""
    runs: dict[str, dict[str, Any]] = {}
    with CURVES.open() as fh:
        for row in csv.DictReader(fh):
            run = runs.setdefault(
                row["run"],
                {
                    "run": row["run"],
                    "word": row["word"],
                    "seed": int(row["seed"]),
                    "batch": int(row["batch"]),
                    "eval": [],
                    "train": [],
                },
            )
            step = int(row["step"])
            if row["eval_reward"]:
                run["eval"].append((step, float(row["eval_reward"])))
            if row["train_reward"]:
                run["train"].append((step, float(row["train_reward"])))
    return runs


def score_runs(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for run in sorted(runs.values(), key=lambda r: (r["word"], r["seed"])):
        ev = steps_to_saturation(run["eval"]) if run["eval"] else None
        tr = steps_to_saturation(run["train"]) if run["train"] else None
        out.append(
            {
                "run": run["run"],
                "word": run["word"],
                "seed": run["seed"],
                "batch": run["batch"],
                "last_step": max(s for s, _ in run["train"]) if run["train"] else None,
                "n_eval_points": len(run["eval"]),
                "eval_onset": ev["onset"] if ev else None,
                "eval_censored": ev["censored"] if ev else None,
                "eval_max_rate": ev["max_rate"] if ev else None,
                "train_onset": tr["onset"] if tr else None,
                "train_censored": tr["censored"] if tr else None,
                "unstable": ev["unstable"] if ev else None,
            }
        )
    return out


def _by_word(scored: list[dict[str, Any]], key: str, batch: int | None = None) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        if batch is not None and row["batch"] != batch:
            continue
        if row[key] is not None:
            acc[row["word"]].append(float(row[key]))
    return dict(acc)


def _fmt(v: float | None, spec: str = "6.2f") -> str:
    return "  --  " if v is None else format(v, spec)


def report(scored: list[dict[str, Any]]) -> None:
    print("=" * 96)
    print("R1 — steps to 50% saturation.  15 runs, 3 words x (3 + 3) seeds, $0 on the free queue.")
    print("=" * 96)
    print()
    print(f"{'run':<18} {'batch':>5} {'last':>5} {'evpts':>6} {'eval_onset':>11} {'train_onset':>12} {'censored':>9}")
    for r in scored:
        cens = "EVAL" if r["eval_censored"] else ""
        print(
            f"{r['run']:<18} {r['batch']:>5} {str(r['last_step']):>5} {r['n_eval_points']:>6} "
            f"{_fmt(r['eval_onset'], '11.2f')} {_fmt(r['train_onset'], '12.2f')} {cens:>9}"
        )

    for metric in ("eval_onset", "train_onset"):
        label = "EVAL (pre-registered headline)" if metric == "eval_onset" else "TRAIN (corroborating)"
        print()
        print("-" * 96)
        print(f"{label}")
        print("-" * 96)
        pooled = _by_word(scored, metric)
        print(f"{'word':<11} {'prime':>6} {'base_rate':>10} {'n':>3} {'median':>8} {'mean':>8} {'sd':>7} {'range':>18}")
        for w in WORDS:
            v = pooled.get(w, [])
            if not v:
                print(f"{w:<11} {PRIME_ONSETS[w]:>6} {OUR_BASE_RATES[w]:>10.4f} {'0':>3}   all censored")
                continue
            d = dispersion(v)
            rng = f"[{d['min']:.2f}, {d['max']:.2f}]"
            print(
                f"{w:<11} {PRIME_ONSETS[w]:>6} {OUR_BASE_RATES[w]:>10.4f} {d['n']:>3} "
                f"{d['median']:>8.2f} {d['mean']:>8.2f} {_fmt(d['sd'], '7.2f')} {rng:>18}"
            )

        # G3: ordering and magnitude against Prime, scored on medians (that is what G3 asks).
        medians = {w: (statistics.median(v) if v else None) for w, v in
                   ((w, pooled.get(w, [])) for w in WORDS)}
        v = onset_verdict(medians)
        print()
        print(f"  G3   verdict={v['verdict']}   ordering_matches_prime={v['ordering_matches_prime']}")
        print(f"       observed order  {' < '.join(v['order_by_onset'])}")
        print(f"       prime predicts  {' < '.join(v['order_prime_predicts'])}")
        print(f"       within +/-{BAND:.0%} band: {v['within_band']}")
        # Deliberately two lists rather than a boolean. `onset_verdict` used to return one, named
        # `r1p_confirmed`, and it reported confirmation on a p = 0.29 null by ordering medians.
        print(f"       our base rates predict  {' < '.join(v['order_our_base_rates_predict'])}"
              f"   (matches observed: {v['order_by_onset'] == v['order_our_base_rates_predict']}"
              " — ORDERING ONLY, not a test)")

        # R1-P: the distributional test on the discriminating pair.
        t = r1p_test(pooled)
        print()
        early, late = DISCRIMINATING_PAIR
        print(f"  R1-P  {early} vs {late}   n = {t['n'][early]} vs {t['n'][late]}")
        if t["test"]:
            tt, ci = t["test"], t["interval"]
            print(f"        U = {tt['u']:.1f}/{tt['u_max']:.0f}   exact one-sided p = "
                  f"{tt['p_one_sided']:.4f}   (floor for THIS design = {tt['p_floor']:.4f}, "
                  f"two-direction {2 * tt['p_floor']:.4f})")
            bound = (f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]" if ci["bounded"] else "UNBOUNDED")
            print(f"        shift ({late} - {early}) = {ci['point']:+.2f} steps, "
                  f"{1 - t['alpha']:.0%} CI {bound}")
            if ci["bounded"]:
                prime_gap = float(PRIME_ONSETS[late] - PRIME_ONSETS[early])
                print(f"          excludes zero? {not (ci['lo'] <= 0 <= ci['hi'])}"
                      f"   excludes Prime's {prime_gap:+.0f}-step gap? "
                      f"{not (ci['lo'] <= prime_gap <= ci['hi'])}")
        verdict = {True: "CONFIRMED", False: "FALSIFIED", None: "UNRESOLVED"}[t["confirmed"]]
        print(f"        alpha = {t['alpha']} (two-sided, {t['alpha'] / 2} per direction)   "
              f"CAN_EVER_REJECT = {t['can_ever_reject']}   verdict = {verdict}")
        print(f"        {t['reason']}")

        # Every pair, so the coarse-vs-fine split is visible rather than asserted.
        print()
        print("  pairwise (does the left word saturate earlier?), Holm-corrected across the 3")
        print("  contrasts R1-P's monotone claim implies — p == floor means 'as extreme as this")
        print("  design can get', which is a statement about n, not about the effect:")
        raw = []
        for a, b in (("forgotten", "ocean"), ("forgotten", "midnight"), ("ocean", "midnight")):
            if len(pooled.get(a, [])) < 2 or len(pooled.get(b, [])) < 2:
                print(f"        {a:>9} < {b:<9}  insufficient seeds")
                continue
            raw.append((a, b, exact_mannwhitney_u(pooled[a], pooled[b])))
        for rank, (a, b, tt) in enumerate(sorted(raw, key=lambda r: r[2]["p_one_sided"])):
            holm = min(1.0, tt["p_one_sided"] * (len(raw) - rank))
            at_floor = "  <-- p == floor" if tt["p_one_sided"] <= tt["p_floor"] + 1e-12 else ""
            ratio = OUR_BASE_RATES[a] / OUR_BASE_RATES[b]
            print(f"        {a:>9} < {b:<9}  U = {tt['u']:>4.1f}/{tt['u_max']:<4.0f} "
                  f"p = {tt['p_one_sided']:.4f}  Holm = {holm:.4f}  "
                  f"({ratio:5.1f}x base rate){at_floor}")

        # Per batch, because batch 1 and batch 2 disagree and pooling would hide that.
        print()
        print("  by batch (batch 1 and batch 2 point opposite ways — this is why n=3 was not enough):")
        for b in (1, 2):
            per = _by_word(scored, metric, batch=b)
            a_, b_ = DISCRIMINATING_PAIR
            if len(per.get(a_, [])) < 2 or len(per.get(b_, [])) < 2:
                print(f"        batch {b}:  n = {len(per.get(a_, []))} vs {len(per.get(b_, []))}"
                      "  — insufficient seeds for a direction")
                continue
            tt = exact_mannwhitney_u(per[a_], per[b_])
            print(f"        batch {b}:  U = {tt['u']:>4.1f}/{tt['u_max']:<4.0f} "
                  f"p = {tt['p_one_sided']:.4f}   {a_} {statistics.median(per[a_]):.2f} "
                  f"vs {b_} {statistics.median(per[b_]):.2f}")


def main() -> int:
    if not CURVES.exists():
        print(f"missing {CURVES} — run scripts/fetch_r1_curves.py first")
        return 1
    scored = score_runs(load_curves())

    with ONSETS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scored[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scored)

    report(scored)
    print()
    print(f"wrote {ONSETS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
