"""The training loop, run end-to-end against ``ToyPolicy`` — real autograd, no GPU.

The point of the ``Policy`` seam is that these run in milliseconds on a laptop. Shape and masking
bugs are silent at runtime, so they have to be caught here rather than on a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from assay.crawl.config import LadderConfig
from assay.crawl.loop import train
from assay.crawl.policy import ToyPolicy
from assay.crawl.runlog import read_step_logs


def _run(tmp_path: Path, **kw: object) -> list:
    cfg = LadderConfig(run_id="t", steps=3, prompts_per_step=6, group_size=8, **kw)  # type: ignore[arg-type]
    return train(cfg, tmp_path, policy=ToyPolicy(p_correct=0.6, seed=1))


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


def test_the_loop_runs_and_persists_every_step(tmp_path: Path) -> None:
    logs = _run(tmp_path)
    assert len(logs) == 3
    assert read_step_logs(tmp_path) == logs


def test_the_ladder_produces_a_real_gradient(tmp_path: Path) -> None:
    logs = _run(tmp_path, baseline="group_loo", normalize_by_std=True)
    assert all(log.grad_norm > 0 for log in logs)
    assert all(-1.0 <= log.half_batch_grad_cosine <= 1.0 for log in logs)


def test_proxy_equals_true_on_the_ladder(tmp_path: Path) -> None:
    """Runs 1-7 train on R_binary, so the gap is identically zero. Correct, not a bug."""
    for baseline in ("none", "global", "group_loo"):
        logs = _run(tmp_path, baseline=baseline)
        assert all(log.gap == 0.0 for log in logs)


# --------------------------------------------------------------------------------------
# Ablation D, in the loop rather than in the arithmetic
# --------------------------------------------------------------------------------------


def test_unanimous_groups_produce_no_gradient(tmp_path: Path) -> None:
    """Every group dead => zero gradient, while tokens and wall-clock accrue exactly as normal.

    That last part is the point of running D rather than only asserting the arithmetic: this is
    what a wasted training budget looks like from the outside.
    """
    logs = _run(tmp_path, baseline="group_loo", force_unanimous_groups=True)
    assert all(log.frac_degenerate_groups == 1.0 for log in logs)
    assert all(log.grad_norm < 1e-8 for log in logs)
    assert all(log.max_abs_advantage == 0.0 for log in logs)
    assert all(log.tokens > 0 for log in logs), "compute was still spent"


def test_the_tiebreaker_revives_dead_groups(tmp_path: Path) -> None:
    """Ablation C's other face: a continuous reward term means a group can never be unanimous."""
    binary = _run(tmp_path, baseline="group_loo", normalize_by_std=True, reward="binary")
    tiebreak = _run(tmp_path, baseline="group_loo", normalize_by_std=True, reward="tiebreak")
    assert all(log.gap == 0.0 for log in binary)
    assert all(log.gap > 0.0 for log in tiebreak)
    assert all(log.frac_degenerate_groups == 0.0 for log in tiebreak)


# --------------------------------------------------------------------------------------
# The half-batch split — exactness, and the group boundary
# --------------------------------------------------------------------------------------


def test_the_halves_sum_to_the_full_batch_gradient_with_an_odd_group_count() -> None:
    """Each half is scaled by the FULL rollout count, so the split never changes the update.

    Averaging two half-*means* would be correct only for equally-sized halves; with an odd number
    of groups it silently rescales the gradient. Verified against a single full-batch backward.
    """
    policy = ToyPolicy(seed=3)
    total = 7

    def terms() -> list[torch.Tensor]:
        # Rebuilt each time: optimize() frees the graph on its second backward. Linear in the
        # parameter, so the optimizer step between the two measurements cannot change the gradient.
        return [float(v) * policy.logits.sum() for v in range(1, total + 1)]

    # Split 4/3 — deliberately uneven, which is where averaging two half-means goes wrong.
    parts = terms()
    policy.optimize(
        torch.stack(parts[:4]).sum() / total, torch.stack(parts[4:]).sum() / total
    )
    via_halves = policy.logits.grad.clone()

    policy.opt.zero_grad(set_to_none=True)
    (torch.stack(terms()).sum() / total).backward()
    via_single = policy.logits.grad.clone()

    assert torch.allclose(via_halves, via_single, atol=1e-6)


def test_a_single_group_is_rejected(tmp_path: Path) -> None:
    """One group cannot be split into two non-empty halves, and groups are never split."""
    with pytest.raises(ValueError, match="prompts_per_step"):
        train(
            LadderConfig(run_id="t", steps=1, prompts_per_step=1),
            tmp_path,
            policy=ToyPolicy(seed=0),
        )


def test_an_odd_group_count_still_runs(tmp_path: Path) -> None:
    cfg = LadderConfig(run_id="t", steps=2, prompts_per_step=7, group_size=4)
    logs = train(cfg, tmp_path, policy=ToyPolicy(p_correct=0.5, seed=4))
    assert len(logs) == 2
    assert all(log.grad_norm > 0 for log in logs)


def test_the_cosine_is_reported_as_zero_when_it_is_undefined(tmp_path: Path) -> None:
    """Both half-gradients vanish under ablation D. Undefined, not "maximally noisy"."""
    logs = _run(tmp_path, baseline="group_loo", force_unanimous_groups=True)
    assert all(log.half_batch_grad_cosine == 0.0 for log in logs)
    assert all(log.grad_norm < 1e-8 for log in logs), "read the cosine alongside grad_norm"


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_a_run_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    a = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, seed=7),
        tmp_path / "a",
        policy=ToyPolicy(p_correct=0.5, seed=7),
    )
    b = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, seed=7),
        tmp_path / "b",
        policy=ToyPolicy(p_correct=0.5, seed=7),
    )
    assert [x.proxy_reward for x in a] == [x.proxy_reward for x in b]
    assert [x.grad_norm for x in a] == [x.grad_norm for x in b]


def test_different_seeds_draw_different_prompts(tmp_path: Path) -> None:
    """Fresh prompts every step, or the reward curve is a memorisation curve."""
    a = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, seed=1),
        tmp_path / "a",
        policy=ToyPolicy(p_correct=0.5, seed=0),
    )
    b = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, seed=2),
        tmp_path / "b",
        policy=ToyPolicy(p_correct=0.5, seed=0),
    )
    assert [x.proxy_reward for x in a] != [x.proxy_reward for x in b]


# --------------------------------------------------------------------------------------
# Step 6 — the KL leash
# --------------------------------------------------------------------------------------


def test_kl_is_zero_and_uncomputed_when_the_coefficient_is_zero(tmp_path: Path) -> None:
    """Ablation B is kl_coef=0.0, so this is the common path.

    For HFPolicy the reference is an extra full forward pass; multiplying it by zero would pay for
    it on every ladder run. The counter proves it is skipped, not scaled.
    """
    calls = {"n": 0}

    class CountingPolicy(ToyPolicy):
        def kl_to_reference(self, rollouts):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return super().kl_to_reference(rollouts)

    cfg = LadderConfig(run_id="t", steps=2, prompts_per_step=4, kl_coef=0.0)
    logs = train(cfg, tmp_path, policy=CountingPolicy(p_correct=0.5, seed=0))

    assert all(log.kl_to_ref == 0.0 for log in logs)
    assert calls["n"] == 0, "the reference pass must be skipped, not multiplied by zero"


def test_kl_is_computed_and_logged_when_enabled(tmp_path: Path) -> None:
    cfg = LadderConfig(run_id="t", steps=2, prompts_per_step=4, kl_coef=0.1)
    logs = train(cfg, tmp_path, policy=ToyPolicy(p_correct=0.5, seed=0))
    assert all(log.kl_to_ref >= 0.0 for log in logs), "k3 is non-negative by construction"


def test_the_kl_term_changes_the_update(tmp_path: Path) -> None:
    """A leash that does not alter the gradient is not a leash."""
    without = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, kl_coef=0.0),
        tmp_path / "a",
        policy=ToyPolicy(p_correct=0.5, seed=5),
    )
    with_kl = train(
        LadderConfig(run_id="t", steps=2, prompts_per_step=4, kl_coef=5.0),
        tmp_path / "b",
        policy=ToyPolicy(p_correct=0.5, seed=5),
    )
    assert [x.grad_norm for x in without] != [x.grad_norm for x in with_kl]


# --------------------------------------------------------------------------------------
# Ablation A's direction, checked locally before any GPU time
# --------------------------------------------------------------------------------------


def test_removing_the_baseline_lowers_the_half_batch_snr(tmp_path: Path) -> None:
    """Ablation A's predicted direction, on a toy where the reward genuinely depends on tokens.

    Two earlier versions of ToyPolicy could NOT test this, and both failures were instructive:

    - uniform token ids broke ``E[grad log pi] = 0``, so the no-baseline arm kept a large shared
      component and its halves agreed spuriously — the cosine came out *backwards*;
    - a reward independent of the tokens made the true gradient exactly zero, so both arms scored
      cos ~ 0 and there was no signal-to-noise ratio to compare.

    This is a direction check on a toy, not a prediction of the real magnitude.
    """
    import statistics

    from assay.crawl.runlog import gradient_snr

    def snr(**kw: object) -> float:
        cosines = []
        for seed in range(4):
            cfg = LadderConfig(
                run_id="a", steps=10, prompts_per_step=8, group_size=8, seed=seed, **kw  # type: ignore[arg-type]
            )
            logs = train(
                cfg,
                tmp_path / f"{kw['baseline']}{seed}",
                policy=ToyPolicy(seed=seed, lr=0.01, vocab=8, reward_from_tokens=True),
            )
            cosines.append(statistics.mean(x.half_batch_grad_cosine for x in logs))
        value = gradient_snr(statistics.mean(cosines))
        assert value is not None
        return value

    without_baseline = snr(baseline="none")
    with_baseline = snr(baseline="global")

    assert without_baseline < with_baseline, "removing the baseline must lower the SNR"
    assert with_baseline / without_baseline > 1.5
