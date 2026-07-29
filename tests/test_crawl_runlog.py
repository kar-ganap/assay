"""Run plumbing: manifests, per-step logs, and ``d(gap)/d(step)``.

The outcome variable is a **slope**, not an endpoint. At a 200-step budget a policy can be visibly
on its way to hacking without saturating, and at the endpoint that is indistinguishable from never
having started (``docs/pre-registration.md`` §4 L3). ``test_slope_separates_a_climbing_run_from_a_
flat_one`` is that claim, asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assay.crawl import runlog
from assay.crawl.config import LadderConfig
from assay.loop import StepLog


def _log(
    step: int,
    *,
    proxy: float,
    true: float,
    degenerate: float = 0.0,
    entropy: float = 1.0,
    distinct: int = 64,
    max_adv: float = 1.0,
    grad_norm: float = 1.0,
    cosine: float = 0.9,
) -> StepLog:
    return StepLog(
        step=step,
        proxy_reward=proxy,
        true_reward=true,
        policy_entropy=entropy,
        distinct_completions=distinct,
        kl_to_ref=0.0,
        kl_loss_fraction=0.0,
        grad_norm=grad_norm,
        half_batch_grad_cosine=cosine,
        max_abs_advantage=max_adv,
        group_pass_rate=0.5,
        frac_degenerate_groups=degenerate,
        tokens=100,
        wall_clock_s=1.0,
    )


# --------------------------------------------------------------------------------------
# The gap and its slope
# --------------------------------------------------------------------------------------


def test_gap_is_proxy_minus_true() -> None:
    assert _log(0, proxy=0.8, true=0.3).gap == pytest.approx(0.5)


def test_binary_reward_gives_an_identically_zero_gap() -> None:
    """Ladder runs 1-7 train on R_binary, so proxy == true. A zero gap there is correct, not a bug."""
    logs = [_log(t, proxy=0.7, true=0.7) for t in range(0, 201, 10)]
    assert all(log.gap == 0.0 for log in logs)
    assert runlog.gap_slope(logs) == pytest.approx(0.0)


def test_slope_separates_a_climbing_run_from_a_flat_one() -> None:
    """The reason the outcome is a slope: neither run has saturated by step 200."""
    climbing = [_log(t, proxy=0.5 + 0.001 * t, true=0.5) for t in range(201)]
    flat = [_log(t, proxy=0.5, true=0.5) for t in range(201)]

    assert runlog.gap_slope(climbing) == pytest.approx(0.001, rel=1e-6)
    assert runlog.gap_slope(flat) == pytest.approx(0.0)


def test_slope_ignores_the_burn_in_window() -> None:
    """A large transient before step 50 must not be mistaken for a sustained trend."""
    logs = [_log(t, proxy=(0.9 if t < 50 else 0.5), true=0.5) for t in range(201)]
    assert runlog.gap_slope(logs) == pytest.approx(0.0)


def test_slope_handles_a_gap_that_starts_negative() -> None:
    """Ablation B starts at gap ~ -0.46: baseline tag compliance is below baseline correctness.

    A gap that starts negative and rises is still a rising gap — another reason the slope is the
    right statistic and the endpoint is not.
    """
    logs = [_log(t, proxy=0.26 + 0.004 * t, true=0.72) for t in range(201)]
    assert logs[0].gap < 0
    assert runlog.gap_slope(logs) == pytest.approx(0.004, rel=1e-6)


def test_slope_is_none_when_the_window_is_too_short() -> None:
    """A crashed or short run must not be reported as a slope of zero."""
    assert runlog.gap_slope([_log(t, proxy=0.5, true=0.5) for t in range(10)]) is None
    assert runlog.gap_slope([]) is None


def test_fit_slope_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        runlog.fit_slope([1.0], [1.0])
    with pytest.raises(ValueError):
        runlog.fit_slope([2.0, 2.0], [1.0, 3.0])


# --------------------------------------------------------------------------------------
# Ablation A's metrics — noisiness must not be confounded with magnitude or with trend
# --------------------------------------------------------------------------------------


def test_cv_is_invariant_to_scale() -> None:
    """A gradient that is 10x larger is not 10x noisier. Raw std would say it is."""
    small = [10.0, 12.0, 8.0, 11.0, 9.0]
    large = [10 * v for v in small]
    assert runlog._coefficient_of_variation(small) == pytest.approx(
        runlog._coefficient_of_variation(large)
    )


def test_cv_is_none_when_the_gradient_vanished() -> None:
    """A dead run has no meaningful relative variability; a huge CV would read as confirmation."""
    assert runlog._coefficient_of_variation([0.0, 0.0, 0.0]) is None


def test_detrending_separates_trajectory_from_noise() -> None:
    """The confound: a smoothly decaying gradient has high raw CV and *zero* step-to-step jitter.

    Without detrending, a legitimate trajectory masquerades as ablation-A noise.
    """
    steps = [float(t) for t in range(50, 201)]
    smooth_decay = [10.0 - 0.04 * t for t in steps]  # perfectly linear, no noise at all

    assert runlog._coefficient_of_variation(smooth_decay) > 0.2, "raw CV sees the trend as spread"
    assert runlog._detrended_cv(steps, smooth_decay) == pytest.approx(0.0, abs=1e-9)


def test_detrended_cv_still_sees_real_jitter() -> None:
    """Detrending must not launder away the thing ablation A is actually looking for."""
    steps = [float(t) for t in range(50, 201)]
    jittery = [10.0 + (1.0 if t % 2 else -1.0) for t in steps]  # flat trend, real scatter
    assert runlog._detrended_cv(steps, jittery) == pytest.approx(0.1, abs=0.01)


def test_summary_reports_all_three_ablation_a_metrics() -> None:
    """Cosine is primary; the two CVs sit beside it so raw-vs-detrended is visible."""
    logs = [_log(t, proxy=0.5, true=0.5, grad_norm=10.0 - 0.04 * t, cosine=0.3) for t in range(201)]
    summary = runlog.summarize_run(LadderConfig(run_id="run1"), logs)
    assert summary["half_batch_grad_cosine_mean"] == pytest.approx(0.3)
    assert summary["grad_norm_cv"] > summary["grad_norm_cv_detrended"]
    assert summary["grad_norm_cv_detrended"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# Persistence — a run without per-step logs or a manifest is not usable
# --------------------------------------------------------------------------------------


def test_step_logs_round_trip(tmp_path: Path) -> None:
    logs = [_log(t, proxy=0.5 + 0.01 * t, true=0.5) for t in range(5)]
    with runlog.step_log_writer(tmp_path) as writer:
        for log in logs:
            writer.append(log)
    assert runlog.read_step_logs(tmp_path) == logs


def test_step_logs_are_flushed_every_step(tmp_path: Path) -> None:
    """A run that dies at step 140 must leave 140 usable steps, not an empty buffer."""
    with runlog.step_log_writer(tmp_path) as writer:
        writer.append(_log(0, proxy=0.5, true=0.5))
        mid_run = (tmp_path / "steps.jsonl").read_text()
    assert mid_run.strip(), "nothing on disk until close — a crash would lose the run"


def test_reading_a_run_without_logs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runlog.read_step_logs(tmp_path)


def test_manifest_records_config_and_provenance(tmp_path: Path) -> None:
    cfg = LadderConfig(run_id="run7", reward="tiebreak")
    manifest = runlog.manifest_for(
        cfg,
        git_sha="abc123",
        git_dirty=False,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_revision="9213176",
        prompt_template_sha256="1d37f53b",
        grader={"r_binary_extractor": "last-integer"},
        backend="modal-L4",
    )
    path = runlog.write_manifest(manifest, tmp_path)
    written = json.loads(path.read_text())

    assert written["config"]["reward"] == "tiebreak"
    assert written["config"]["baseline"] == "group_loo"
    assert written["model_revision"] == "9213176"
    assert written["git_dirty"] is False


# --------------------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------------------


def test_summary_carries_the_outcome_variable() -> None:
    cfg = LadderConfig(run_id="ablation-c", reward="tiebreak")
    logs = [_log(t, proxy=0.5 + 0.002 * t, true=0.5) for t in range(201)]
    summary = runlog.summarize_run(cfg, logs)
    assert summary["gap_slope_50_200"] == pytest.approx(0.002, rel=1e-6)
    assert summary["run_id"] == "ablation-c"
    assert summary["config"]["reward"] == "tiebreak"


def test_summary_tracks_the_degenerate_group_trajectory() -> None:
    """add-2digit's dead groups are 100% saturation-type, so this should rise during training.

    Predicted 0.115 -> ~0.43. Logging the slope makes that prediction falsifiable rather than
    rhetorical (``docs/phases/phase-0.1-grpo-by-hand-plan.md`` → *Finding*).
    """
    logs = [_log(t, proxy=0.7, true=0.7, degenerate=0.115 + 0.0016 * t) for t in range(201)]
    summary = runlog.summarize_run(LadderConfig(run_id="run7"), logs)
    assert summary["frac_degenerate_first"] == pytest.approx(0.115)
    assert summary["frac_degenerate_last"] == pytest.approx(0.435, abs=0.01)
    assert summary["frac_degenerate_slope"] > 0


def test_summary_refuses_an_empty_run() -> None:
    with pytest.raises(ValueError):
        runlog.summarize_run(LadderConfig(run_id="x"), [])


def test_results_are_written_under_the_run_id(tmp_path: Path) -> None:
    cfg = LadderConfig(run_id="run3")
    summary = runlog.summarize_run(cfg, [_log(t, proxy=0.5, true=0.5) for t in range(60)])
    path = runlog.write_results(summary, tmp_path)
    assert path.name == "run3.json"
    assert json.loads(path.read_text())["run_id"] == "run3"


# --------------------------------------------------------------------------------------
# Config — the ladder is switch settings, not separate code paths
# --------------------------------------------------------------------------------------


def test_task_defaults_are_the_pinned_calibration_result() -> None:
    cfg = LadderConfig(run_id="x")
    # add-3digit, not add-2digit: the rule picked the latter, but it saturates to ~100%
    # dead groups within ten steps. Documented deviation, 2026-07-28.
    assert (cfg.family, cfg.setting) == ("arithmetic", "add-3digit")
    assert cfg.group_size == 8, "k = G is what makes the screen's dead-group rate transfer"
    assert (cfg.temperature, cfg.top_p) == (1.0, 1.0)
    # 64, not the screen's 256: no add-2digit completion in 400 samples exceeded 28 tokens, so the
    # two are observationally identical here while 256 pads every scored sequence for nothing.
    assert cfg.max_new_tokens == 64


def test_only_non_binary_rewards_have_a_meaningful_gap() -> None:
    assert not LadderConfig(run_id="x", reward="binary").has_distinct_true_reward
    assert LadderConfig(run_id="x", reward="format").has_distinct_true_reward
    assert LadderConfig(run_id="x", reward="tiebreak").has_distinct_true_reward


def test_single_epoch_is_pinned() -> None:
    """The default is the pinned design, so it lands in every manifest without being remembered."""
    assert LadderConfig(run_id="x").epochs_per_batch == 1


def test_clipping_cannot_bind_under_a_single_epoch() -> None:
    """A config that *claims* to clip but cannot must say so.

    With one gradient step per batch the importance ratio is identically 1, so clip(1, ...) == 1.
    This is why rung 4 is cut: under the pinned design it would be bit-identical to rung 3.
    """
    inert = LadderConfig(run_id="x", clip_epsilon=0.2, epochs_per_batch=1)
    assert not inert.clipping_is_active

    active = LadderConfig(run_id="x", clip_epsilon=0.2, epochs_per_batch=4)
    assert active.clipping_is_active

    # Epsilon absent: nothing to bind regardless of epochs.
    assert not LadderConfig(run_id="x", clip_epsilon=None, epochs_per_batch=4).clipping_is_active


def test_rollouts_per_step() -> None:
    assert LadderConfig(run_id="x", prompts_per_step=16, group_size=8).rollouts_per_step == 128


# --------------------------------------------------------------------------------------
# Figures — gate 5: every number regenerates from a committed script, GPU-free.
# --------------------------------------------------------------------------------------


def test_figures_regenerate_from_logs_alone(tmp_path: Path) -> None:
    from assay.crawl import figures

    phase = tmp_path / "phase"
    for run_id, slope in (("run7", 0.0), ("ablation_c", 0.002)):
        run_dir = phase / "raw" / run_id
        logs = [_log(t, proxy=0.5 + slope * t, true=0.5, degenerate=0.1 + 0.001 * t)
                for t in range(201)]
        with runlog.step_log_writer(run_dir) as writer:
            for log in logs:
                writer.append(log)
        runlog.write_results(
            runlog.summarize_run(LadderConfig(run_id=run_id), logs), phase / "results"
        )

    out = figures.plot(phase)
    assert out.exists() and out.stat().st_size > 0

    table = figures.gap_table(phase)
    assert "ablation_c" in table and "run7" in table
    assert "+2.000e-03" in table, "the outcome variable must appear in the regenerated table"


def test_figures_refuse_to_invent_data(tmp_path: Path) -> None:
    from assay.crawl import figures

    with pytest.raises(FileNotFoundError):
        figures.load_runs(tmp_path)


# --------------------------------------------------------------------------------------
# Ablation A's SNR — scale-free in the operating point, unlike a raw cosine gap
# --------------------------------------------------------------------------------------


def test_snr_ratio_is_invariant_to_operating_point() -> None:
    """The reason A's threshold moved off raw cosine gaps.

    A fixed variance ratio shows very different cosine *gaps* depending on where the baselined arm
    sits, so a real effect can slip under an absolute threshold. The SNR ratio does not move.
    """
    # Same underlying 3.6x variance ratio, two operating points.  rho = cos/(1-cos), and
    # rho scales inversely with variance, so rho_1 = rho_2 / 3.6 in both cases.
    for cos_baselined in (0.80, 0.95):
        rho2 = cos_baselined / (1 - cos_baselined)
        rho1 = rho2 / 3.6
        cos_unbaselined = rho1 / (1 + rho1)

        raw_gap = cos_baselined - cos_unbaselined
        ratio = runlog.gradient_snr(cos_baselined) / runlog.gradient_snr(cos_unbaselined)

        assert ratio == pytest.approx(3.6, rel=1e-6), "the SNR ratio recovers the variance ratio"
        if cos_baselined == 0.95:
            assert raw_gap < 0.15, "a real effect that the old absolute threshold would have missed"


def test_snr_is_none_when_noise_is_below_resolution() -> None:
    """cos == 1 must not report an enormous SNR; it means we cannot measure the noise."""
    assert runlog.gradient_snr(1.0) is None


def test_snr_is_zero_when_halves_do_not_agree() -> None:
    assert runlog.gradient_snr(0.0) == 0.0
    assert runlog.gradient_snr(-0.3) == 0.0


def test_summary_reports_the_snr() -> None:
    logs = [_log(t, proxy=0.5, true=0.5, cosine=0.8) for t in range(60)]
    summary = runlog.summarize_run(LadderConfig(run_id="run2"), logs)
    assert summary["gradient_snr"] == pytest.approx(4.0)


# --------------------------------------------------------------------------------------
# A dead run must not be reported as a flat gap
# --------------------------------------------------------------------------------------


def test_a_run_that_dies_partway_is_flagged_not_silently_flattened() -> None:
    """Ablation B saturates and dies; the ladder may too. A slope fitted over mostly-dead steps
    reads as "no gap opened" when the truth is "the run ended at step 80"."""
    logs = [
        _log(t, proxy=0.5 + (0.004 * t if t < 80 else 0.004 * 80), true=0.5,
             degenerate=0.2 if t < 80 else 1.0)
        for t in range(201)
    ]
    summary = runlog.summarize_run(LadderConfig(run_id="ablation_b"), logs)

    assert summary["first_all_dead_step"] == 80
    assert summary["live_steps"] == 80
    # Only steps 50-79 of the 50-200 window are live: 30 of 151.
    assert summary["live_fraction_in_slope_window"] == pytest.approx(30 / 151, abs=0.01)
    # The slope looks near-zero, which without the flag would read as an honest null.
    assert abs(summary["gap_slope_50_200"]) < 0.001


def test_a_healthy_run_reports_a_full_live_window() -> None:
    logs = [_log(t, proxy=0.5 + 0.002 * t, true=0.5, degenerate=0.2) for t in range(201)]
    summary = runlog.summarize_run(LadderConfig(run_id="run7"), logs)
    assert summary["first_all_dead_step"] is None
    assert summary["live_fraction_in_slope_window"] == pytest.approx(1.0)
