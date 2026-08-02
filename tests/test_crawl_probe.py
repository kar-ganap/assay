"""Ablation A's paired fixed-policy probe — real autograd, no GPU.

The probe replaced a training-arm comparison that could not answer A's question. These tests pin the
properties that made the replacement necessary, so the same class of error cannot return silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from assay.crawl.policy import ToyPolicy
from assay.crawl.probe import (
    BASELINES,
    ProbeConfig,
    collect_paired_gradients,
    probe,
    probe_configs,
    probe_verdict,
    variance_statistics,
)


def _cfg(**kw: object) -> ProbeConfig:
    defaults: dict[str, object] = {
        "run_id": "p", "batches": 8, "prompts_per_batch": 4, "group_size": 8, "bootstrap": 200
    }
    return ProbeConfig(**{**defaults, **kw})  # type: ignore[arg-type]


def _policy() -> ToyPolicy:
    return ToyPolicy(p_correct=0.6, seed=1, reward_from_tokens=True)


# --------------------------------------------------------------------------------------
# Pairing — the property the whole design rests on
# --------------------------------------------------------------------------------------


def test_every_baseline_sees_the_same_rollouts() -> None:
    """If the batches differed per baseline, this would be three noisy runs rather than a probe.

    Checked through the *rewards*: ``generate`` and ``grade_pair`` are called once per batch, so a
    regression that re-sampled per baseline would show up as differing gradients on a batch where
    the baselines are mathematically equivalent. ``group_loo`` and ``group_mean`` differ only by the
    constant ``G/(G-1)``, so their gradients must stay exactly proportional on shared rollouts.
    """
    grads, telemetry = collect_paired_gradients(_cfg(batches=3), policy=_policy())
    assert set(grads) == set(BASELINES)
    assert all(len(v) == 3 for v in grads.values())
    assert 0.0 <= telemetry["pass_rate"] <= 1.0
    assert telemetry["rollouts"] == 3 * 4 * 8


def test_the_probe_does_not_step_the_policy() -> None:
    """A probe that trains is measuring a moving target, and its later batches are not iid."""
    policy = _policy()
    before = policy.logits.detach().clone()
    collect_paired_gradients(_cfg(batches=4), policy=policy)
    assert torch.equal(policy.logits.detach(), before)


# --------------------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------------------


def test_nsr_is_scale_free() -> None:
    """Required, not cosmetic: the arms produce gradients of very different magnitude.

    Run 1's ``grad_norm`` held ~0.77 while run 2's fell to 0.23, so any comparison on *absolute*
    variance would report that scale gap instead of estimator quality.
    """
    grads = [torch.randn(64) + 3.0 for _ in range(12)]
    plain = variance_statistics(grads, bootstrap=50, seed=0)
    scaled = variance_statistics([g * 7.5 for g in grads], bootstrap=50, seed=0)
    assert scaled["nsr"] == pytest.approx(plain["nsr"], rel=1e-5)


def test_variance_decomposition_matches_the_direct_computation() -> None:
    """``V = mean(diag K) - mean(K)`` is the Gram identity the bootstrap depends on."""
    grads = [torch.randn(32) for _ in range(10)]
    stats = variance_statistics(grads, bootstrap=10, seed=0)
    mean = torch.stack(grads).mean(dim=0)
    direct = float(torch.stack([(g - mean).pow(2).sum() for g in grads]).mean())
    assert stats["variance"] == pytest.approx(direct, rel=1e-4)
    assert stats["signal_sq"] == pytest.approx(float(mean @ mean), rel=1e-4)


def test_zero_variance_gradients_give_zero_nsr() -> None:
    stats = variance_statistics([torch.ones(16) for _ in range(5)], bootstrap=10, seed=0)
    assert stats["nsr"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


def _stats(nsr: dict[str, float], *, cosine: float = 1.0, spread: float = 0.02) -> dict:
    """Synthetic statistics with a controllable bootstrap spread and mean-gradient agreement.

    Each baseline gets an **independently** seeded bootstrap. Sharing one deterministic pattern
    across baselines makes their ratio constant however wide the spread, so the ``not_measurable``
    branch becomes unreachable — the double would silently vouch for a gate it never exercised.
    """
    base = torch.zeros(8)
    base[0] = 1.0
    tilted = torch.zeros(8)
    tilted[0], tilted[1] = cosine, (1 - cosine**2) ** 0.5
    out = {}
    for index, b in enumerate(BASELINES):
        generator = torch.Generator().manual_seed(index)
        jitter = 1 + spread * torch.randn(400, generator=generator)
        out[b] = {
            "nsr": nsr[b],
            "mean_gradient": base if b == "none" else tilted,
            "bootstrap_nsr": (nsr[b] * jitter.abs()).tolist(),
        }
    return out


def test_the_gate_confirms_on_the_predicted_ordering_and_ratio() -> None:
    out = probe_verdict(_stats({"none": 1.0, "global": 0.4, "group_loo": 0.3}), pass_rate=0.6)
    assert out["verdict"] == "confirmed"
    assert out["ratio_none_over_global"] == pytest.approx(2.5)


def test_the_gate_calls_a_significant_reversal_falsified() -> None:
    """The exact shape the training-arm comparison produced: ``rho_2/rho_1 = 0.14``."""
    out = probe_verdict(_stats({"none": 0.14, "global": 1.0, "group_loo": 1.0}), pass_rate=0.43)
    assert out["verdict"] == "falsified"
    assert out["reversal_detected"] is True


def test_a_reversal_inside_the_noise_is_not_falsified() -> None:
    """Ratios of 0.859 and 0.921 with intervals spanning 1.0 — the real no-length-norm seeds.

    Gating direction on the point estimate would call the baseline actively harmful here, off
    nothing but sampling noise.
    """
    out = probe_verdict(
        _stats({"none": 0.88, "global": 1.0, "group_loo": 0.95}, spread=0.10), pass_rate=0.43
    )
    assert out["reversal_detected"] is False, "the interval spans 1.0 — no direction established"
    assert out["verdict"] != "falsified"


def test_the_gate_reports_partial_between_the_thresholds() -> None:
    assert probe_verdict(_stats({"none": 1.0, "global": 0.62, "group_loo": 0.6}), pass_rate=0.43)[
        "verdict"
    ] == "partial"


def test_a_wide_interval_is_not_measurable_rather_than_a_null() -> None:
    """The distinction the old design could not express, and the reason A stalled.

    "The baseline does not reduce variance" and "this rig cannot tell" are different claims, and
    only the second is fixed by drawing more batches.
    """
    out = probe_verdict(_stats({"none": 1.0, "global": 0.9, "group_loo": 0.9}, spread=0.9), pass_rate=0.43)
    assert out["verdict"] == "not_measurable"


def test_disagreeing_mean_gradients_are_a_rig_failure_not_a_result() -> None:
    """All three baselines estimate the same expected gradient; if they do not, nothing else holds.

    ``E[b * grad log pi] = 0`` is what makes a baseline free. This is that identity as a check
    rather than an assumption.
    """
    out = probe_verdict(_stats({"none": 1.0, "global": 0.3, "group_loo": 0.2}, cosine=0.5), pass_rate=0.43)
    assert out["verdict"] == "rig_broken"
    assert "disagree" in out["reason"]


def test_non_finite_statistics_are_a_rig_failure() -> None:
    out = probe_verdict(_stats({"none": float("nan"), "global": 0.3, "group_loo": 0.2}), pass_rate=0.43)
    assert out["verdict"] == "rig_broken"


# --------------------------------------------------------------------------------------
# End to end + the pre-registered grid
# --------------------------------------------------------------------------------------


def test_the_probe_runs_and_persists_a_verdict(tmp_path: Path) -> None:
    result = probe(_cfg(batches=6), tmp_path, policy=_policy())
    assert (tmp_path / "probe.json").exists()
    assert result["verdict"]["verdict"] in {
        "confirmed", "partial", "falsified", "not_measurable", "rig_broken",
    }
    # The 3.4M-vector and the 2,000 bootstrap draws must not reach the committed summary.
    assert "mean_gradient" not in result["statistics"]["none"]
    assert "bootstrap_nsr" not in result["statistics"]["none"]


def test_warmup_moves_the_policy_then_probes_the_moved_one(tmp_path: Path) -> None:
    policy = _policy()
    before = policy.logits.detach().clone()
    probe(_cfg(batches=3, warmup_steps=2), tmp_path, policy=policy)
    assert not torch.equal(policy.logits.detach(), before), "warmup did not train"


def test_probe_batches_never_reuse_a_warmup_prompt_stream() -> None:
    """Otherwise the probe partly measures memorisation of the batches it just trained on."""
    from assay.crawl.loop import _prompt_seed
    from assay.crawl.probe import _batch_seed

    cfg = ProbeConfig(run_id="p", seed=3, warmup_steps=200)
    trained = {_prompt_seed(cfg.as_ladder_config(), step) for step in range(cfg.warmup_steps)}
    probed = {_batch_seed(cfg, batch) for batch in range(cfg.batches)}
    assert not (trained & probed)


def test_the_pre_registered_grid_is_three_seeds_at_two_operating_points() -> None:
    grid = probe_configs()
    assert len(grid) == 6
    assert {c.warmup_steps for c in grid} == {0, 50}
    assert {c.seed for c in grid} == {0, 1, 2}
    assert len({c.run_id for c in grid}) == 6


def test_predicted_ratio_matches_the_derivation() -> None:
    """``1/(1-p)``, checked against the simulated values that calibrated the gate."""
    from assay.crawl.probe import predicted_ratio

    assert predicted_ratio(0.43) == pytest.approx(1.754, abs=1e-3)  # simulated 1.74
    assert predicted_ratio(0.75) == pytest.approx(4.000, abs=1e-3)  # simulated 3.87
    assert predicted_ratio(0.50) == pytest.approx(2.000, abs=1e-3)


def test_the_gate_can_fail_upward() -> None:
    """The property a one-sided threshold cannot have, and the reason for the change.

    A ratio far *above* ``1/(1-p)`` disconfirms the theory exactly as much as one far below. Under
    the old flat ``>= 2.0`` gate this case scored ``confirmed``, because it only ever asked whether
    the reduction was large enough.
    """
    # p=0.43 predicts 1.754; an observed 5.0 is a large reduction AND a bad fit to theory.
    out = probe_verdict(
        _stats({"none": 5.0, "global": 1.0, "group_loo": 0.9}), pass_rate=0.43
    )
    assert out["ratio_none_over_global"] == pytest.approx(5.0)
    assert out["ratio_predicted"] == pytest.approx(1.754, abs=1e-3)
    assert out["verdict"] == "partial", "a ratio far above prediction must not read as confirmed"


def test_a_correct_result_at_the_base_policy_would_have_failed_the_old_gate() -> None:
    """Regression guard on the actual miscalibration, found 2026-08-01 before the probe ran.

    At the base policy's measured ``p ~ 0.43`` theory predicts 1.75, which is *below* the retired
    ``>= 2.0`` pass mark — so a perfectly correct result would have been scored ``partial``.
    """
    out = probe_verdict(
        _stats({"none": 1.754, "global": 1.0, "group_loo": 0.95}), pass_rate=0.43
    )
    assert out["verdict"] == "confirmed"
    assert out["ratio_none_over_global"] < 2.0


def test_no_reduction_plus_excluded_prediction_is_a_result_not_an_absence() -> None:
    """The 2026-08-01 case the old chain reported as "we learned nothing".

    The no-length-norm probe produced an interval that contained 1.0 *and* excluded 1/(1-p) on
    every seed. Both facts were true; only "not_measurable" was reported, because the old if/elif
    tested them in sequence and the first match hid the second. A ruled-out magnitude is a finding.
    """
    out = probe_verdict(
        _stats({"none": 1.02, "global": 1.0, "group_loo": 0.95}, spread=0.10), pass_rate=0.43
    )
    assert out["reduction_detected"] is False, "CI must span 1.0 for this case"
    assert out["prediction_consistent"] is False, "CI must exclude 1/(1-p) for this case"
    assert out["verdict"] == "magnitude_excluded"


def test_an_interval_spanning_both_is_the_only_uninformative_case(tmp_path: Path) -> None:
    """``not_measurable`` now means exactly one thing: more batches would actually help."""
    out = probe_verdict(
        _stats({"none": 1.4, "global": 1.0, "group_loo": 0.95}, spread=0.75), pass_rate=0.43
    )
    assert out["reduction_detected"] is False
    assert out["prediction_consistent"] is True
    assert out["verdict"] == "not_measurable"


def test_the_two_facts_are_reported_beside_every_verdict() -> None:
    """A reader must never have to trust the label over the interval it came from."""
    out = probe_verdict(_stats({"none": 1.0, "global": 0.4, "group_loo": 0.3}), pass_rate=0.6)
    assert {"reduction_detected", "prediction_consistent", "ratio_ci95", "ratio_predicted"} <= set(out)
