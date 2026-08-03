"""Regenerate every Phase 0.1 figure from **committed** artefacts.

    uv run --extra dev python -m assay.crawl.figures experiments/phase-0.1-grpo-by-hand --export
    uv run --extra dev python -m assay.crawl.figures experiments/phase-0.1-grpo-by-hand

Validation gate: *every number regenerates from a committed script*. The first invocation rebuilds
``results/series/<run>.csv`` from ``raw/<run>/steps.jsonl``; the second draws the figures from
``results/`` alone. Only the second is needed by anyone who clones the repo --- which is the point,
because ``raw/`` is gitignored and a figure script that reads it cannot satisfy the gate.

**Why the per-step series belongs in ``results/``.** The house convention gitignores ``raw/`` because
it holds raw *rollouts and generations*. A ``StepLog`` is a derived metric, not a generation, and
``experiments/.gitignore`` explicitly admits ``results/*.json|csv``. Committing the series is what
makes the curves reproducible without re-running 410 GPU-minutes; the committed ``results/*.json``
carry summary scalars only (``n_steps`` is not a series), so before this the curves could not be
regenerated at all.

**The panels changed on 2026-08-02**, after the clean ladder landed. The original set was written
before any result existed and guessed wrong about three of the four ablations: it plotted
``grad_norm`` variance for A (the real answer is a fixed-policy probe, because training arms are not
comparable at all), entropy collapse for B (the real signature is the proxy--true gap; entropy never
collapsed), and it referenced ``add-2digit``, retired as the primary arm on 2026-07-28.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

#: Columns carried into the committed series. Everything a panel draws, nothing else.
SERIES_COLUMNS = (
    "step",
    "proxy_reward",
    "true_reward",
    "policy_entropy",
    "distinct_completions",
    "kl_to_ref",
    "kl_loss_fraction",
    "grad_norm",
    "half_batch_grad_cosine",
    "half_batch_grad_cosine_centered",
    "max_abs_advantage",
    "group_pass_rate",
    "frac_degenerate_groups",
    "tokens",
)

#: Rollouts per step, so token counts read per rollout rather than per batch.
ROLLOUTS_PER_STEP = 128

#: The primary arm. ``add-2digit`` saturates within ten steps and is the robustness arm only.
TASK = "add-3digit"

_SEED_RE = re.compile(r"-seed\d+$")


def arm_of(run_name: str) -> str:
    """``run7-add-3digit-seed1`` -> ``run7-add-3digit``. Seeds of one arm are pooled."""
    return _SEED_RE.sub("", run_name)


# --------------------------------------------------------------------------------------
# Export: raw/ -> results/series/   (run once after a fetch, then commit the CSVs)
# --------------------------------------------------------------------------------------


#: Committed alongside the series. Without it a figure cannot tell comparable runs apart, and
#: pooling seeds across code versions or GPU tiers is exactly the error the clean rerun existed to
#: remove — reintroduced silently at plot time, which is worse, because the plot looks fine.
COHORT_FILE = "_cohort.json"

#: Every file whose contents determine what a ladder run *does*. A cohort is defined by the hash of
#: these, **not** by the commit SHA.
#:
#: The distinction is not pedantic. On 2026-08-02 the seed pass ran at ``48b900f9`` while seed 0 sat
#: at ``f1cc4048``, and keying on the SHA would have discarded seed 0 as a minority cohort — from
#: three seeds back to two, on arms that had just cost $7 to produce. The only difference between
#: those commits inside ``src/`` was ``probe_a``'s entry point, which a ladder run never executes.
#: A commit SHA answers "was the tree identical?"; a run needs "was the code that produced me
#: identical?", and those diverge the moment anything else in the repo is edited.
TRAINING_PATH = (
    "src/assay/crawl/loop.py",
    "src/assay/crawl/advantage.py",
    "src/assay/crawl/hf_policy.py",
    "src/assay/crawl/policy.py",
    "src/assay/crawl/config.py",
    "src/assay/crawl/ladder.py",
    "src/assay/crawl/rewards.py",
    "src/assay/crawl/tasks.py",
    "src/assay/crawl/logprob.py",
)


def code_fingerprint(git_sha: str) -> str:
    """Hash of ``TRAINING_PATH`` as it stood at ``git_sha``. Falls back to the SHA itself.

    Backfilled from git for runs recorded before this existed. The fallback is deliberately the raw
    SHA rather than a constant: an unresolvable commit must never silently pool with anything.
    """
    import hashlib
    import subprocess

    digest = hashlib.sha256()
    for path in TRAINING_PATH:
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{git_sha}:{path}"], stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return f"unresolved-{git_sha}"
        digest.update(blob)
    return digest.hexdigest()[:12]


def export_series(phase_dir: Path) -> list[Path]:
    """Rebuild ``results/series/<run>.csv`` + ``_cohort.json`` from ``raw/``."""
    raw = phase_dir / "raw"
    if not raw.exists():
        raise FileNotFoundError(f"no raw/ under {phase_dir}; nothing to export")
    out_dir = phase_dir / "results" / "series"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    cohort: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(raw.iterdir()):
        steps_path = run_dir / "steps.jsonl"
        if not steps_path.exists():
            continue
        rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        # Calibration sweeps and probes live under raw/ too and have a different shape.
        if not rows or "proxy_reward" not in rows[0]:
            continue
        target = out_dir / f"{run_dir.name}.csv"
        with target.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(SERIES_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in SERIES_COLUMNS})
        written.append(target)

        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sha = str(manifest.get("git_sha", "unknown"))
        cohort[run_dir.name] = {
            "git_sha": sha[:8],
            "git_dirty": bool(manifest.get("git_dirty", True)),
            "backend": str(manifest.get("backend", "unknown")),
            "code_fingerprint": code_fingerprint(sha),
        }
    (out_dir / COHORT_FILE).write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    return written


# --------------------------------------------------------------------------------------
# Load: results/ only
# --------------------------------------------------------------------------------------

Series = dict[str, list[dict[str, float | None]]]


def _as_float(value: str | None) -> float | None:
    return None if value in (None, "", "None") else float(str(value))


def load_series(phase_dir: Path) -> Series:
    """Per-step series for every run, **restricted to one comparable cohort**.

    A cohort is a ``(code_fingerprint, backend)`` pair with a clean tree — the *training path's*
    content hash, not the commit SHA (see ``TRAINING_PATH``). Runs outside the dominant cohort are
    dropped and named on stdout, never silently pooled: seeds produced at different commits or on
    different GPU tiers are not replicates of each other, and averaging them would launder the exact
    provenance problem the 2026-08-02 clean rerun was run to remove.
    """
    series_dir = phase_dir / "results" / "series"
    if not series_dir.exists():
        raise FileNotFoundError(
            f"{series_dir} missing — run once with --export (needs raw/), then commit the CSVs"
        )
    runs: Series = {}
    for path in sorted(series_dir.glob("*.csv")):
        with path.open() as fh:
            runs[path.stem] = [
                {k: _as_float(v) for k, v in row.items()} for row in csv.DictReader(fh)
            ]
    if not runs:
        raise FileNotFoundError(f"no CSVs under {series_dir}")

    cohort_path = series_dir / COHORT_FILE
    if not cohort_path.exists():
        print(f"WARNING: no {COHORT_FILE}; cannot verify runs are comparable — re-run --export")
        return runs
    cohort: dict[str, dict[str, Any]] = json.loads(cohort_path.read_text())

    tally: dict[tuple[str, str], int] = {}
    for name in runs:
        meta = cohort.get(name)
        if not meta or meta["git_dirty"]:
            continue
        key = (meta.get("code_fingerprint", meta["git_sha"]), meta["backend"])
        tally[key] = tally.get(key, 0) + 1
    if not tally:
        print("WARNING: no clean-tree runs in the cohort file; plotting everything unfiltered")
        return runs

    chosen = max(tally, key=lambda k: tally[k])
    kept: Series = {}
    dropped: list[str] = []
    for name, series in runs.items():
        meta = cohort.get(name, {})
        fp = meta.get("code_fingerprint", meta.get("git_sha")) if meta else None
        if meta and not meta["git_dirty"] and (fp, meta["backend"]) == chosen:
            kept[name] = series
        else:
            dropped.append(name)

    print(f"cohort: training-path {chosen[0]} on {chosen[1]} — {len(kept)} runs")
    for name in sorted(dropped):
        meta = cohort.get(name, {})
        why = ("dirty tree" if meta.get("git_dirty")
               else f"{meta.get('code_fingerprint', '?')}/{meta.get('backend')}")
        print(f"  excluded {name}  ({why})")
    return kept


def load_probes(phase_dir: Path) -> dict[str, dict[str, Any]]:
    """Ablation A's probe results, keyed by run id."""
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted((phase_dir / "results").glob("probeA-*.json"))
    }


def pooled(runs: Series, arm: str, field: str) -> tuple[list[float], list[float], list[float], int]:
    """Across every seed of ``arm``: ``(steps, mean, spread, n_seeds)``.

    ``spread`` is the half-width of the min--max envelope, which is the seed band this project
    reports beside every effect size rather than a standard error --- with three seeds an envelope
    is honest and an SE is theatre.
    """
    members = [v for k, v in runs.items() if arm_of(k) == arm]
    if not members:
        return [], [], [], 0
    steps: list[float] = []
    mean: list[float] = []
    spread: list[float] = []
    for i in range(min(len(m) for m in members)):
        raw_vals = [m[i].get(field) for m in members]
        vals = [float(v) for v in raw_vals if v is not None]
        if not vals:
            continue
        step = members[0][i].get("step")
        steps.append(float(i) if step is None else float(step))
        mean.append(sum(vals) / len(vals))
        spread.append((max(vals) - min(vals)) / 2)
    return steps, mean, spread, len(members)


def smooth(values: list[float], window: int = 9) -> list[float]:
    """Centred moving average. The per-step metrics are far noisier than the trend they carry."""
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    return [
        sum(values[max(0, i - half) : min(len(values), i + half + 1)])
        / len(values[max(0, i - half) : min(len(values), i + half + 1)])
        for i in range(len(values))
    ]


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------

RUNGS = [
    (f"run1-{TASK}", "run 1 — no baseline", "#B45309"),
    (f"run2-{TASK}", "run 2 — global baseline", "#0F766E"),
    (f"run3-{TASK}", "run 3 — group baseline", "#1D4ED8"),
    (f"run7-{TASK}", "run 7 — full GRPO", "#111827"),
]


def _curve(  # type: ignore[no-untyped-def]
    ax, runs: Series, arm: str, field: str, label: str, colour: str, *, scale=1.0, ls="-"
) -> int:
    steps, mean, spread, n = pooled(runs, arm, field)
    if not steps:
        return 0
    values = smooth([v * scale for v in mean])
    ax.plot(steps, values, label=label + (f"  (n={n})" if n > 1 else ""), color=colour, lw=1.6,
            ls=ls)
    if n > 1:
        band = smooth([s * scale for s in spread])
        ax.fill_between(steps, [v - b for v, b in zip(values, band)],
                        [v + b for v, b in zip(values, band)],
                        color=colour, alpha=0.15, linewidth=0)
    return n


def figure_ladder(runs: Series, out: Path) -> Path:
    """The ladder works, and the group baseline introduces the dead-group pathology."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for arm, label, colour in RUNGS:
        _curve(ax1, runs, arm, "true_reward", label, colour)
        _curve(ax2, runs, arm, "frac_degenerate_groups", label, colour)

    ax1.axhline(0.433, ls="--", lw=1.0, color="#9CA3AF")
    ax1.annotate("base rate 0.433", xy=(6, 0.44), fontsize=8, color="#6B7280")
    ax1.set_title("Gradients flow: true reward by rung", fontsize=10)
    ax1.set_ylabel(r"true reward ($R_{\mathrm{binary}}$)")
    ax1.set_ylim(0.3, 1.0)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    ax2.set_title("Dead groups are introduced by the GROUP baseline", fontsize=10)
    ax2.set_ylabel("fraction of groups with zero gradient")
    ax2.set_ylim(-0.03, 0.85)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_ablation_b(runs: Series, out: Path) -> Path:
    """Ablation B: a degenerate grader hacked, and a leash that makes it marginally worse.

    At n=1 the two arms looked identical (gap 0.507 vs 0.489) and were reported as "no effect". At
    n=3 the paired difference is +0.037 with the same sign on 3/3 seeds, and the leashed arm also
    ends with LOWER true reward (0.474 vs 0.517) — below the base model's own 0.516. Whatever
    beta=0.04 is buying here, it is not restraint.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    arm = f"ablation_b_control-{TASK}"
    steps, proxy, _, _ = pooled(runs, arm, "proxy_reward")
    _, true, _, _ = pooled(runs, arm, "true_reward")
    proxy, true = smooth(proxy), smooth(true)

    ax1.plot(steps, proxy, color="#B91C1C", lw=1.8, label=r"proxy ($R_{\mathrm{format}}$) — trained on")
    ax1.plot(steps, true, color="#111827", lw=1.8, label=r"true ($R_{\mathrm{binary}}$) — measured only")
    ax1.fill_between(steps, true, proxy, where=[p >= t for p, t in zip(proxy, true)],
                     color="#B91C1C", alpha=0.15, linewidth=0, label="gap (proxy over true)")
    ax1.fill_between(steps, true, proxy, where=[p < t for p, t in zip(proxy, true)],
                     color="#1D4ED8", alpha=0.18, linewidth=0, label="gap NEGATIVE at step 0")
    ax1.set_title("The proxy is learned in ten steps; the skill never is", fontsize=10)
    ax1.set_ylabel("reward")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.62, 0.72))

    for a, label, colour in (
        (f"ablation_b_control-{TASK}", r"leash on ($\beta$ = 0.04)", "#0F766E"),
        (f"ablation_b-{TASK}", r"leash off ($\beta$ = 0)", "#B45309"),
    ):
        s, p, _, n = pooled(runs, a, "proxy_reward")
        _, t, _, _ = pooled(runs, a, "true_reward")
        if s:
            ax2.plot(s, smooth([x - y for x, y in zip(p, t)]), color=colour, lw=1.6,
                     label=label + (f"  (n={n})" if n > 1 else ""))
    ax2.axhline(0, ls="--", lw=1.0, color="#9CA3AF")
    ax2.set_title("The leash makes the gap slightly WORSE (3/3 seeds)", fontsize=10)
    ax2.set_ylabel("gap  (proxy − true)")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_ablations_cd(runs: Series, out: Path) -> Path:
    """C: a 0.001 tie-breaker buys length. D: a unanimous group buys nothing at all."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for arm, label, colour, ls in (
        (f"run7-{TASK}", "run 7 (binary reward)", "#111827", "-"),
        (f"run7_nolennorm-{TASK}", "run 7, no length-norm", "#6B7280", "--"),
        (f"ablation_c-{TASK}", "ablation C (tie-breaker)", "#B91C1C", "-"),
        (f"ablation_c_nolennorm-{TASK}", "ablation C, no length-norm", "#F87171", "--"),
    ):
        _curve(ax1, runs, arm, "tokens", label, colour, scale=1 / ROLLOUTS_PER_STEP, ls=ls)
    ax1.set_title("The tie-breaker drives padding — not the optimizer", fontsize=10)
    ax1.set_ylabel("tokens per rollout")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    ax2b = ax2.twinx()
    _curve(ax2, runs, f"ablation_d-{TASK}", "true_reward", "true reward", "#111827")
    s, g, _, _ = pooled(runs, f"ablation_d-{TASK}", "grad_norm")
    if s:
        ax2b.plot(s, g, color="#B91C1C", lw=1.8, label="gradient norm")
    ax2.set_title("Ablation D — every group forced unanimous", fontsize=10)
    ax2.set_ylabel("true reward")
    ax2.set_ylim(0.3, 1.0)
    ax2b.set_ylabel("gradient norm", color="#B91C1C")
    ax2b.set_ylim(-0.05, 1.0)
    ax2b.tick_params(axis="y", colors="#B91C1C")
    ax2.annotate("grad_norm = 0.0000 on all 200 steps\ndead groups = 1.000 on all 200 steps",
                 xy=(0.04, 0.86), xycoords="axes fraction", fontsize=8, color="#B91C1C")
    handles, labels = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(handles + h2, labels + l2, fontsize=8, frameon=False, loc="upper right")
    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.spines[["top"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_ablation_a(probes: dict[str, dict[str, Any]], out: Path) -> Path:
    """Ablation A: the paired fixed-policy probe, which is its only well-posed form."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    base = [p for k, p in sorted(probes.items()) if "-w0-" in k and k.endswith("-n160")]
    for i, probe in enumerate(base):
        verdict = probe["verdict"]
        lo, hi = verdict["ratio_ci95"]
        ax1.plot([i, i], [lo, hi], color="#111827", lw=2.2, solid_capstyle="butt")
        ax1.plot(i, verdict["ratio_none_over_global"], "o", color="#111827", ms=5,
                 label="observed (95% CI)" if i == 0 else None)
        ax1.plot(i, verdict["ratio_predicted"], "D", color="#B91C1C", ms=6,
                 label=r"theory $1/(1-p)$" if i == 0 else None)
    ax1.axhline(1.0, ls="--", lw=1.0, color="#9CA3AF")
    ax1.annotate("no reduction", xy=(-0.42, 1.03), fontsize=8, color="#6B7280")
    ax1.set_xticks(range(len(base)))
    ax1.set_xticklabels([f"seed {i}" for i in range(len(base))])
    ax1.set_title("The predicted variance reduction does not appear", fontsize=10)
    ax1.set_ylabel(r"$\mathrm{NSR_{none}} / \mathrm{NSR_{global}}$")
    ax1.set_ylim(0.5, 2.2)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    groups = []
    for suffix in ("on", "off"):
        vals = [
            min(p["verdict"]["mean_cosines"].values())
            for k, p in probes.items()
            if "-w0-" in k
            and not k.endswith("-n160")
            and (k.endswith("-nolennorm") == (suffix == "off"))
            and p["verdict"].get("mean_cosines")
        ]
        groups.append(vals)
    for i, (vals, colour) in enumerate(zip(groups, ("#B91C1C", "#0F766E"))):
        if vals:
            ax2.bar(i, sum(vals) / len(vals), 0.5, color=colour, alpha=0.85)
            ax2.plot([i] * len(vals), vals, "o", color="#111827", ms=4)
    ax2.axhline(0.90, ls="--", lw=1.0, color="#9CA3AF")
    ax2.annotate("agreement floor 0.90", xy=(-0.45, 0.915), fontsize=8, color="#6B7280")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["length-norm ON", "length-norm OFF"])
    ax2.set_title(r"Length-norm breaks $\mathbb{E}[\nabla \log \pi] = 0$", fontsize=10)
    ax2.set_ylabel("min pairwise cosine, arms' mean gradients")
    ax2.set_ylim(0, 1.05)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    phase_dir = Path(args[0])

    if "--export" in args:
        written = export_series(phase_dir)
        print(f"exported {len(written)} series -> {phase_dir / 'results' / 'series'}")
        return 0

    import matplotlib

    matplotlib.use("Agg")

    runs = load_series(phase_dir)
    probes = load_probes(phase_dir)
    fig_dir = phase_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    made = [
        figure_ladder(runs, fig_dir / "fig1-ladder.png"),
        figure_ablation_b(runs, fig_dir / "fig2-ablation-b.png"),
        figure_ablations_cd(runs, fig_dir / "fig3-ablations-cd.png"),
    ]
    if probes:
        made.append(figure_ablation_a(probes, fig_dir / "fig4-ablation-a.png"))
    else:
        print("no probeA-*.json found — skipping ablation A's panel")

    for path in made:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
