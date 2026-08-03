"""Executable spec for ``assay.crawl.advantage.group_advantages`` — **user implements** (§7).

These tests are expected to **FAIL** until the advantage function is written. That is deliberate:
they are the red half of red-green, and they encode the GRPO semantics the whole phase rests on.

Read the ladder they describe:

- ``baseline="none"``       -> REINFORCE. Rung 1. With binary reward, failed rollouts get weight
                              exactly zero, so you never push *down* on anything.
- ``baseline="global"``     -> a running mean over all prompts. Rung 2. Unbiased and a real variance
                              reduction, but the advantage's sign is set by whether the *prompt* was
                              easier or harder than average, not by whether the completion was good.
- ``baseline="group_mean"`` -> the GRPO move. Rung 3. A per-prompt Monte Carlo estimate of the
                              variance-minimising baseline ``E[R|x]``, bought with rollouts you were
                              sampling anyway instead of with a critic network.
- ``baseline="group_loo"``  -> leave-one-out. The full-group mean includes ``r_i``, so the baseline
                              depends on the completion being scored and carries an ``O(1/G)`` bias.
                              Excluding self is exactly unbiased. The phase plan pinned this variant.

And the two failures that live in this function:

- **Ablation D** — a unanimous group has every advantage zero. Note this is a pathology GRPO
  *introduces*: under ``baseline="global"`` the same group is nonzero. Conditioning the baseline on
  the prompt is what makes unanimity informationless, because the difficulty signal was the only
  thing there. Dead groups are the price of GRPO's variance reduction.
- **Ablation C** — ``normalize_by_std`` makes the advantage a z-score, which is **bounded by
  sqrt(G-1)** and **scale-invariant**. It therefore discards *how much* better the winner was and
  keeps only the shape. Consequence in
  ``test_tiny_tiebreaker_reaches_the_same_magnitude_as_a_real_signal``.

Convention pinned by these tests: **population standard deviation** (``ddof=0``). The bound is
``sqrt(G-1)``; with ``ddof=1`` it would be ``(G-1)/sqrt(G)``.
"""

from __future__ import annotations

import math
import random

import pytest

from assay.crawl.advantage import group_advantages

#: The ``xfail(strict=True)`` handoff marker was removed on 2026-07-27 when implementation started.
#: It existed so an *untouched* spec would not fail the gate; during active red-green work, honest
#: red is the correct state and hiding it behind xfail would obscure the progression.

G = 8
SQRT_G_MINUS_1 = math.sqrt(G - 1)


# --------------------------------------------------------------------------------------
# The ladder: rung 1 -> rung 2 -> rung 3
# --------------------------------------------------------------------------------------


def test_baseline_none_returns_raw_rewards() -> None:
    """REINFORCE. Failed rollouts carry weight zero — they say nothing at all."""
    rewards = [1.0, 0.0, 0.0, 1.0]
    assert group_advantages(rewards, baseline="none", normalize_by_std=False) == rewards


def test_global_baseline_subtracts_the_given_constant() -> None:
    out = group_advantages(
        [1.0, 0.0], baseline="global", normalize_by_std=False, global_baseline_value=0.5
    )
    assert out == pytest.approx([0.5, -0.5])


def test_global_baseline_requires_a_value() -> None:
    with pytest.raises(ValueError):
        group_advantages([1.0, 0.0], baseline="global", normalize_by_std=False)


def test_group_mean_advantages_sum_to_zero() -> None:
    """Centering on the group is what removes the prompt-difficulty component."""
    out = group_advantages([1.0, 1.0, 0.0, 0.0, 0.0], baseline="group_mean", normalize_by_std=False)
    assert sum(out) == pytest.approx(0.0)


def test_group_mean_sizes_the_update_to_what_is_surprising() -> None:
    """7 of 8 correct: the correct ones barely move, the lone failure carries the signal."""
    out = group_advantages([1.0] * 7 + [0.0], baseline="group_mean", normalize_by_std=False)
    assert out[0] == pytest.approx(0.125)
    assert out[-1] == pytest.approx(-0.875)


def test_loo_baseline_excludes_self() -> None:
    """Full mean includes r_i and carries an O(1/G) bias; leave-one-out does not."""
    rewards = [1.0, 0.0, 0.0, 0.0]
    loo = group_advantages(rewards, baseline="group_loo", normalize_by_std=False)
    full = group_advantages(rewards, baseline="group_mean", normalize_by_std=False)
    assert loo[0] == pytest.approx(1.0)  # baseline = mean of the other three = 0.0
    assert full[0] == pytest.approx(0.75)  # baseline = 0.25


# --------------------------------------------------------------------------------------
# Ablation D — the unanimous group
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("baseline", ["group_mean", "group_loo"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unanimous_group_yields_zero_advantage(
    baseline: str, normalize: bool, value: float
) -> None:
    """All-pass and all-fail are equally dead. This is ablation D, asserted rather than plotted."""
    out = group_advantages([value] * G, baseline=baseline, normalize_by_std=normalize)  # type: ignore[arg-type]
    assert out == pytest.approx([0.0] * G)


def test_unanimous_group_does_not_produce_nan() -> None:
    """std = 0. Dividing must not yield NaN — a NaN silently kills the run rather than wasting a step."""
    out = group_advantages([1.0] * G, baseline="group_mean", normalize_by_std=True)
    assert all(math.isfinite(a) for a in out)


def test_global_baseline_keeps_a_unanimous_group_alive() -> None:
    """The contrast that shows D is a pathology GRPO *introduces*, not a fact about policy gradients."""
    out = group_advantages(
        [1.0] * G, baseline="global", normalize_by_std=False, global_baseline_value=0.5
    )
    assert all(a == pytest.approx(0.5) for a in out)


# --------------------------------------------------------------------------------------
# Ablation C — the z-score bound, and what it actually costs you
# --------------------------------------------------------------------------------------


def test_lone_winner_hits_the_bound_exactly() -> None:
    """One rollout against G-1 identical ones is the maximum-|z| configuration."""
    out = group_advantages([1.0] + [0.0] * (G - 1), baseline="group_mean", normalize_by_std=True)
    assert max(out) == pytest.approx(SQRT_G_MINUS_1)
    assert min(out) == pytest.approx(-1.0 / SQRT_G_MINUS_1)


def test_normalized_advantage_is_bounded_by_sqrt_g_minus_1() -> None:
    """No reward shape can produce a magnitude spike. 'Divide-by-~0 spikes' is unreachable."""
    rng = random.Random(0)
    for _ in range(500):
        rewards = [rng.choice([0.0, 1.0]) for _ in range(G)]
        out = group_advantages(rewards, baseline="group_mean", normalize_by_std=True)
        assert max(abs(a) for a in out) <= SQRT_G_MINUS_1 + 1e-9


def test_normalized_advantage_is_bounded_for_continuous_rewards_too() -> None:
    """Fine-grained reward does not unlock the spike either — the bound is structural."""
    rng = random.Random(1)
    for _ in range(500):
        rewards = [rng.gauss(0.0, 1e-6) for _ in range(G)]
        out = group_advantages(rewards, baseline="group_mean", normalize_by_std=True)
        assert max(abs(a) for a in out) <= SQRT_G_MINUS_1 + 1e-9


@pytest.mark.parametrize("scale,shift", [(1000.0, 0.0), (1e-6, 0.0), (1.0, 50.0), (3.0, -2.0)])
def test_normalized_advantage_is_scale_invariant(scale: float, shift: float) -> None:
    """z-scoring keeps only the *shape* of reward differences and discards their *magnitude*."""
    rewards = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    base = group_advantages(rewards, baseline="group_mean", normalize_by_std=True)
    transformed = group_advantages(
        [scale * r + shift for r in rewards], baseline="group_mean", normalize_by_std=True
    )
    assert transformed == pytest.approx(base)


def test_unnormalized_advantage_is_not_scale_invariant() -> None:
    """The contrast: without the std division, magnitude survives — which is the whole difference."""
    rewards = [0.0, 1.0, 0.0, 0.0]
    base = group_advantages(rewards, baseline="group_mean", normalize_by_std=False)
    scaled = group_advantages([100.0 * r for r in rewards], baseline="group_mean", normalize_by_std=False)
    assert scaled == pytest.approx([100.0 * a for a in base])


def test_tiny_tiebreaker_reaches_the_same_magnitude_as_a_real_signal() -> None:
    """**Ablation C.** A unanimous-correct group plus ``+0.001 * tokens`` is not dead — it is fully
    alive at full gradient magnitude, teaching the model to optimise the tie-breaker.

    Goodhart emerging from the optimiser's own arithmetic rather than from a bad grader.
    """
    tokens = [50] * (G - 1) + [60]
    tiebroken = [1.0 + 0.001 * t for t in tokens]
    binary_split = [0.0] * (G - 1) + [1.0]

    tie_adv = group_advantages(tiebroken, baseline="group_mean", normalize_by_std=True)
    real_adv = group_advantages(binary_split, baseline="group_mean", normalize_by_std=True)

    assert max(tie_adv) == pytest.approx(max(real_adv))
    assert max(tie_adv) == pytest.approx(SQRT_G_MINUS_1)


def test_removing_the_tiebreaker_restores_the_dead_group() -> None:
    """The same group, clean binary reward: zero gradient. C and D are one event with two faces."""
    out = group_advantages([1.0] * G, baseline="group_mean", normalize_by_std=True)
    assert out == pytest.approx([0.0] * G)


# --------------------------------------------------------------------------------------
# Shape contracts
# --------------------------------------------------------------------------------------


def test_output_length_matches_input() -> None:
    assert len(group_advantages([1.0, 0.0, 0.0], baseline="group_mean", normalize_by_std=False)) == 3


def test_empty_group_raises() -> None:
    with pytest.raises(ValueError):
        group_advantages([], baseline="group_mean", normalize_by_std=False)


def test_loo_requires_at_least_two_rollouts() -> None:
    """Leave-one-out of a single rollout has no baseline to compute."""
    with pytest.raises(ValueError):
        group_advantages([1.0], baseline="group_loo", normalize_by_std=False)
