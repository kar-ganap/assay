"""Regenerate every Phase 0.1 figure from committed ``results/*.json`` + ``raw/*/steps.jsonl``.

    uv run --extra dev python -m assay.crawl.figures experiments/phase-0.1-grpo-by-hand

Validation gate 5: *every number regenerates from a committed script*. Nothing here reads a
notebook, a cached value, or a hand-copied figure — if a curve is in the writeup, this script draws
it from the logs.

Four panels, matched to what the phase claims:

1. **reward** — the ladder. Run 7 must visibly clear the base rate by more than the seed band
   (gate 1). Runs 1/2/3 show what each baseline buys.
2. **gradient-norm variance** — ablation **A**. Removing the baseline should widen this visibly;
   most of RL engineering is variance reduction, and this is where you see it.
3. **entropy** — ablation **B**. Cutting the KL leash should collapse it. Entropy death is
   self-terminating learning: identical rollouts -> zero reward variance -> zero advantage.
4. **gap and dead groups** — ablations **C** and **D**. ``gap`` is the project's outcome variable
   in miniature; ``frac_degenerate_groups`` is predicted to *rise* on ``add-2digit`` because all of
   its dead groups are saturation-type.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from assay.crawl.runlog import read_step_logs

PANELS = [
    ("proxy_reward", "reward (proxy)", "ladder — run 7 vs rungs 1/2/3"),
    ("grad_norm", "gradient norm", "ablation A — variance without a baseline"),
    ("policy_entropy", "policy entropy", "ablation B — collapse without KL"),
    ("frac_degenerate_groups", "fraction of dead groups", "ablation D — wasted steps"),
]


def load_runs(phase_dir: Path) -> dict[str, list[Any]]:
    """Load every run's per-step log, keyed by run directory name."""
    raw = phase_dir / "raw"
    if not raw.exists():
        raise FileNotFoundError(f"no raw/ under {phase_dir} — nothing to plot")
    runs = {}
    for run_dir in sorted(raw.iterdir()):
        if (run_dir / "steps.jsonl").exists():
            runs[run_dir.name] = read_step_logs(run_dir)
    if not runs:
        raise FileNotFoundError(f"no steps.jsonl under {raw}")
    return runs


def plot(phase_dir: Path, out: Path | None = None) -> Path:
    """Draw the four-panel figure. Requires the ``dev`` extra (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(phase_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (field, ylabel, title) in zip(axes.ravel(), PANELS, strict=True):
        for run_id, logs in sorted(runs.items()):
            ax.plot(
                [log.step for log in logs],
                [getattr(log, field) for log in logs],
                label=run_id,
                linewidth=1.2,
            )
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes.ravel()[0].legend(fontsize=8)

    fig.tight_layout()
    out = out or phase_dir / "results" / "ladder.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def gap_table(phase_dir: Path) -> str:
    """The gap slope per run, from committed results — the phase's outcome variable."""
    results = sorted((phase_dir / "results").glob("*.json"))
    lines = [f"{'run':<24} {'gap slope (50-200)':>20} {'dead first->last':>20}"]
    for path in results:
        data = json.loads(path.read_text())
        if "gap_slope_50_200" not in data:
            continue  # a calibration sweep, not a ladder run
        slope = data["gap_slope_50_200"]
        slope_text = "n/a (short run)" if slope is None else f"{slope:+.3e}"
        lines.append(
            f"{data['run_id']:<24} {slope_text:>20} "
            f"{data['frac_degenerate_first']:.3f}->{data['frac_degenerate_last']:.3f}".rjust(0)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    phase_dir = Path(args[0])
    print(gap_table(phase_dir))
    print(f"\nwrote {plot(phase_dir)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
