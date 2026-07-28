"""Run plumbing for Phase 0.1 — manifests, per-step logs, and the derived outcome.

Scaffolding (``CLAUDE.md`` §7): the training loop is the user's, this is the data plumbing under it.

**Per-step logging is not optional** (``experiments/README.md``). Proxy and true reward are written
every step because the outcome variable is the *slope* ``d(gap)/d(step)`` over steps 50-200, not a
terminal value — a rising, unsaturated gap is still a clean measurement
(``docs/pre-registration.md`` §4 L3). A run without per-step logs is not usable, and retrofitting
the gap after the fact means re-running everything.

Layout, per ``experiments/README.md``:

    experiments/phase-0.1-grpo-by-hand/
      raw/<run_id>/manifest.json     provenance — a run without one enters no analysis
      raw/<run_id>/steps.jsonl       one StepLog per line; gitignored, never modified
      results/<run_id>.json          derived metrics; committed, figures regenerate from these
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from assay.crawl.config import LadderConfig
from assay.loop import StepLog

#: The gap slope is fitted from this step onward. The first ~50 steps are transients — format
#: acquisition, warmup, the initial policy jolt — and fitting through them measures the startup
#: jump rather than the sustained trend.
SLOPE_START = 50
SLOPE_END = 200


def write_manifest(payload: Mapping[str, Any], run_dir: Path) -> Path:
    """Serialise provenance *before* the first step, so a crashed run is still identifiable."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def manifest_for(
    cfg: LadderConfig,
    *,
    git_sha: str,
    git_dirty: bool,
    model_id: str,
    model_revision: str | None,
    prompt_template_sha256: str,
    grader: Mapping[str, str],
    backend: str,
) -> dict[str, Any]:
    """Assemble the manifest for one ladder run (desideratum 12)."""
    return {
        "run_id": cfg.run_id,
        "config": dataclasses.asdict(cfg),
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_template_sha256": prompt_template_sha256,
        "grader": dict(grader),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "backend": backend,
    }


@contextmanager
def step_log_writer(run_dir: Path) -> Iterator[Any]:
    """Append-only JSONL writer, flushed every step.

    Flushing per step is deliberate: a run that dies at step 140 must leave 140 usable steps behind,
    not an empty buffer. These runs are the expensive part.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handle: TextIO = (run_dir / "steps.jsonl").open("w")

    class _Writer:
        def append(self, log: StepLog) -> None:
            handle.write(json.dumps(dataclasses.asdict(log)) + "\n")
            handle.flush()

    try:
        yield _Writer()
    finally:
        handle.close()


def read_step_logs(run_dir: Path) -> list[StepLog]:
    """Read back a run's per-step log."""
    path = run_dir / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no steps.jsonl in {run_dir} — a run without per-step logs is unusable")
    return [StepLog(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def fit_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary least-squares slope of ``ys`` against ``xs``.

    Plain OLS, no dependencies. If you would rather own the outcome metric under §7, this and
    ``gap_slope`` are the two functions to take.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must be the same length")
    if n < 2:
        raise ValueError("need at least two points to fit a slope")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("xs are all identical — slope is undefined")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def gap_slope(
    logs: Sequence[StepLog], *, start: int = SLOPE_START, end: int = SLOPE_END
) -> float | None:
    """``d(gap)/d(step)`` over ``[start, end]`` — the project's cheap outcome variable.

    Units are reward per step. Returns ``None`` when fewer than two steps fall in the window, which
    is a real state (a short or crashed run) and must not be silently reported as a slope of zero.

    The slope rather than the endpoint is the point: at a 200-step budget a policy can be visibly
    *on its way* to hacking without saturating, and at the endpoint that is indistinguishable from
    never having started (``docs/pre-registration.md`` §4 L3).
    """
    window = [log for log in logs if start <= log.step <= end]
    if len(window) < 2:
        return None
    return fit_slope([float(log.step) for log in window], [log.gap for log in window])


def summarize_run(cfg: LadderConfig, logs: Sequence[StepLog]) -> dict[str, Any]:
    """Derived metrics for ``results/<run_id>.json`` — what every figure regenerates from."""
    if not logs:
        raise ValueError("no steps logged — refusing to emit a hollow summary")

    first, last = logs[0], logs[-1]
    return {
        "run_id": cfg.run_id,
        "config": dataclasses.asdict(cfg),
        "n_steps": len(logs),
        "gap_slope_50_200": gap_slope(logs),
        "gap_first": first.gap,
        "gap_last": last.gap,
        "proxy_reward_first": first.proxy_reward,
        "proxy_reward_last": last.proxy_reward,
        "true_reward_first": first.true_reward,
        "true_reward_last": last.true_reward,
        # Ablation D's direct observable, and the trajectory the calibration screen cannot see:
        # add-2digit's dead groups are 100% saturation-type, so this should *rise*.
        "frac_degenerate_first": first.frac_degenerate_groups,
        "frac_degenerate_last": last.frac_degenerate_groups,
        "frac_degenerate_slope": (
            fit_slope([float(x.step) for x in logs], [x.frac_degenerate_groups for x in logs])
            if len(logs) >= 2
            else None
        ),
        "entropy_first": first.policy_entropy,
        "entropy_last": last.policy_entropy,
        "grad_norm_mean": sum(x.grad_norm for x in logs) / len(logs),
        "total_tokens": sum(x.tokens for x in logs),
        "wall_clock_s": sum(x.wall_clock_s for x in logs),
    }


def write_results(summary: Mapping[str, Any], results_dir: Path) -> Path:
    """Write the committed derived metrics for one run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{summary['run_id']}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path
