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
import math
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


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    """``std/mean``. Scale-free on purpose.

    Removing the baseline changes both the gradient's noisiness *and* its typical magnitude, so raw
    standard deviation would confound them. Returns ``None`` when the mean is ~0 — a real state (a
    dead run), not zero variability, and reporting a huge CV there would look like a spectacular
    ablation-A confirmation.

    **Measured across steps, this conflates estimator noise with genuine trend**: a gradient
    decaying smoothly from 10 to 2 has high CV and zero step-to-step jitter. Use
    ``_detrended_cv`` for the trend-free version, and ``half_batch_grad_cosine`` for the direct
    measurement that needs no correction at all.
    """
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def gradient_snr(cosine: float) -> float | None:
    """``rho = cos/(1-cos)`` — ablation **A**'s metric, scale-free in the operating point.

    The raw cosine gap produced by a fixed variance ratio depends on where you are operating: at a
    genuine 3.6x variance difference, a baseline arm sitting at cos 0.80 shows a gap of 0.27, but
    the same effect at cos 0.95 shows only 0.11. Comparing ``rho`` instead is invariant to that.

    Returns ``None`` when the cosine is numerically 1 — estimator noise below measurement
    resolution, which is a real state and must not be reported as an enormous SNR. Returns 0.0 for
    a non-positive cosine: orthogonal or opposed halves carry no detectable signal.
    """
    if cosine >= 1.0 - 1e-9:
        return None
    if cosine <= 0.0:
        return 0.0
    return cosine / (1.0 - cosine)


def _detrended_cv(steps: Sequence[float], values: Sequence[float]) -> float | None:
    """Relative scatter about a linear fit: ``std(residuals) / |mean(values)|``.

    Note the denominator is the mean of the *values*, not of the residuals — residuals have mean ~0
    by construction, so dividing by them would be meaningless.

    This is the trend-free counterpart to ``_coefficient_of_variation``: a perfectly smooth decay
    scores ~0 here and high there, which is exactly the confound that would otherwise let a
    legitimate trajectory masquerade as ablation-A noise.
    """
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        return None
    try:
        slope = fit_slope(steps, values)
    except ValueError:
        return None
    mean_x = sum(steps) / len(steps)
    intercept = mean - slope * mean_x
    residuals = [v - (slope * x + intercept) for x, v in zip(steps, values, strict=True)]
    scatter = math.sqrt(sum(r**2 for r in residuals) / len(residuals))
    return scatter / abs(mean)


def live_steps(logs: Sequence[StepLog]) -> int:
    """Steps that produced *any* gradient — groups not all unanimous.

    A run whose groups all go dead is over, however many steps remain on the clock. Counting them
    matters because the outcome variable is fitted over a fixed window: if the gradient dies at
    step 80, fitting ``d(gap)/d(step)`` across 50-200 averages 30 live steps with 120 frozen ones
    and flattens the slope toward zero — reporting "no gap opened" when the truth is "the run ended
    at step 80". Read ``gap_slope_50_200`` together with ``live_fraction_in_slope_window``.
    """
    return sum(1 for log in logs if log.frac_degenerate_groups < 1.0)


def summarize_run(cfg: LadderConfig, logs: Sequence[StepLog]) -> dict[str, Any]:
    """Derived metrics for ``results/<run_id>.json`` — what every figure regenerates from."""
    if not logs:
        raise ValueError("no steps logged — refusing to emit a hollow summary")

    first, last = logs[0], logs[-1]
    window = [log for log in logs if SLOPE_START <= log.step <= SLOPE_END]
    return {
        "run_id": cfg.run_id,
        "config": dataclasses.asdict(cfg),
        "n_steps": len(logs),
        "gap_slope_50_200": gap_slope(logs),
        # A slope fitted mostly over dead steps is an artefact, not a null. Ablation B is expected
        # to saturate and die partway; the ladder runs may too as pass rate climbs.
        "live_steps": live_steps(logs),
        "live_fraction_in_slope_window": (live_steps(window) / len(window) if window else None),
        "first_all_dead_step": next(
            (log.step for log in logs if log.frac_degenerate_groups >= 1.0), None
        ),
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
        # Ablation B: collapse is the *conjunction* of falling diversity with sustained proxy
        # reward. Either alone is consistent with the model merely getting worse.
        "entropy_first": first.policy_entropy,
        "entropy_last": last.policy_entropy,
        "distinct_completions_first": first.distinct_completions,
        "distinct_completions_last": last.distinct_completions,
        # Ablation C's rig-broken branch. A z-score cannot exceed sqrt(G-1); if this does, the
        # implementation is wrong and the run says nothing about the science.
        "max_abs_advantage_observed": max(x.max_abs_advantage for x in logs),
        # Ablation A, three ways. The cosine is primary — it measures estimator variance directly
        # and carries no trend confound. The two CVs are reported alongside so the difference
        # between raw and detrended shows whether a CV gap is noise or merely trajectory.
        "half_batch_grad_cosine_mean": (
            sum(x.half_batch_grad_cosine for x in logs) / len(logs)
        ),
        # A's threshold is on the ratio of these between run 1 and run 2, not on raw cosines.
        "gradient_snr": gradient_snr(sum(x.half_batch_grad_cosine for x in logs) / len(logs)),
        "grad_norm_mean": sum(x.grad_norm for x in logs) / len(logs),
        "grad_norm_cv": _coefficient_of_variation([x.grad_norm for x in logs]),
        "grad_norm_cv_detrended": _detrended_cv(
            [float(x.step) for x in logs], [x.grad_norm for x in logs]
        ),
        "total_tokens": sum(x.tokens for x in logs),
        "wall_clock_s": sum(x.wall_clock_s for x in logs),
    }


def write_results(summary: Mapping[str, Any], results_dir: Path) -> Path:
    """Write the committed derived metrics for one run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{summary['run_id']}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path
