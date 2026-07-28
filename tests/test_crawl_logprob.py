"""Log-prob gathering and masking — the silent-failure surface of the policy adapter.

Every test here exists because the corresponding bug produces *finite, plausible numbers* and a
training curve that looks fine. None of them would announce themselves at runtime.
"""

from __future__ import annotations

import math

import pytest
import torch

from assay.crawl.logprob import (
    build_completion_mask,
    completion_logprobs,
    entropy_over_completions,
    sequence_kl,
)

VOCAB = 5


def _uniform_logits(batch: int, seq: int) -> torch.Tensor:
    """Uniform over the vocabulary, so every token's log-prob is exactly -ln(VOCAB)."""
    return torch.zeros(batch, seq, VOCAB)


# --------------------------------------------------------------------------------------
# The mask
# --------------------------------------------------------------------------------------


def test_mask_marks_only_the_completion_span() -> None:
    mask = build_completion_mask(prompt_len=3, completion_lens=[2, 4], total_len=8)
    assert mask[0].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert mask[1].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_mask_rejects_a_completion_that_does_not_fit() -> None:
    """Silently truncating would score the wrong tokens rather than failing."""
    with pytest.raises(ValueError):
        build_completion_mask(prompt_len=3, completion_lens=[6], total_len=8)


# --------------------------------------------------------------------------------------
# The off-by-one — the classic causal-LM bug, and entirely silent
# --------------------------------------------------------------------------------------


def test_a_token_is_scored_by_the_logits_that_preceded_it() -> None:
    """``logits[:, t]`` predicts ``token_ids[:, t+1]``.

    The fixture makes position ``t`` confidently predict token ``t+1``, so a correctly-shifted
    scoring of the sequence ``[0, 1, 2, 3]`` is ~free. The same computation *without* the shift is
    heavily penalised — and that is the point: an off-by-one yields a finite, believable, wrong
    number rather than an error.
    """
    logits = torch.full((1, 4, VOCAB), -100.0)
    for position in range(4):
        logits[0, position, (position + 1) % VOCAB] = 0.0
    token_ids = torch.tensor([[0, 1, 2, 3]])
    mask = build_completion_mask(prompt_len=1, completion_lens=[3], total_len=4)

    correct = completion_logprobs(logits, token_ids, mask).item()
    assert correct == pytest.approx(0.0, abs=1e-3)

    # Exactly what the off-by-one bug computes: token t scored against logits t, no shift.
    unshifted = (
        (
            torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            * mask
        )
        .sum()
        .item()
    )
    assert unshifted < -100
    assert abs(correct - unshifted) > 100, "the two must be distinguishable, or the test is inert"


def test_uniform_logits_give_exactly_minus_n_log_vocab() -> None:
    """An exactly computable case: n completion tokens, uniform policy."""
    logits = _uniform_logits(1, 6)
    token_ids = torch.randint(0, VOCAB, (1, 6))
    mask = build_completion_mask(prompt_len=2, completion_lens=[3], total_len=6)
    assert completion_logprobs(logits, token_ids, mask).item() == pytest.approx(
        -3 * math.log(VOCAB), abs=1e-5
    )


# --------------------------------------------------------------------------------------
# What must NOT contribute
# --------------------------------------------------------------------------------------


def test_prompt_tokens_contribute_nothing() -> None:
    """The policy did not choose them, so rewarding them is rewarding the dataset."""
    logits = _uniform_logits(1, 6)
    token_ids = torch.randint(0, VOCAB, (1, 6))

    short_prompt = completion_logprobs(
        logits, token_ids, build_completion_mask(1, [2], 6)
    ).item()
    long_prompt = completion_logprobs(
        logits, token_ids, build_completion_mask(3, [2], 6)
    ).item()
    assert short_prompt == pytest.approx(long_prompt, abs=1e-5), "prompt length changed the score"


def test_padding_after_an_early_eos_contributes_nothing() -> None:
    """Sequences finish at different points; the right-padding must be inert."""
    logits = _uniform_logits(2, 8)
    token_ids = torch.randint(0, VOCAB, (2, 8))
    mask = build_completion_mask(prompt_len=2, completion_lens=[4, 1], total_len=8)
    scores = completion_logprobs(logits, token_ids, mask)

    assert scores[0].item() == pytest.approx(-4 * math.log(VOCAB), abs=1e-5)
    assert scores[1].item() == pytest.approx(-1 * math.log(VOCAB), abs=1e-5)


def test_ragged_lengths_stay_aligned_across_the_batch() -> None:
    """A per-row length bug shows up as one row borrowing another's tokens."""
    logits = _uniform_logits(3, 7)
    token_ids = torch.randint(0, VOCAB, (3, 7))
    lens = [1, 3, 4]
    scores = completion_logprobs(
        logits, token_ids, build_completion_mask(2, lens, 7)
    )
    for score, n in zip(scores.tolist(), lens, strict=True):
        assert score == pytest.approx(-n * math.log(VOCAB), abs=1e-5)


def test_an_empty_completion_scores_zero() -> None:
    logits = _uniform_logits(1, 5)
    token_ids = torch.randint(0, VOCAB, (1, 5))
    mask = build_completion_mask(prompt_len=3, completion_lens=[0], total_len=5)
    assert completion_logprobs(logits, token_ids, mask).item() == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Shape contracts and gradient flow
# --------------------------------------------------------------------------------------


def test_mismatched_shapes_raise() -> None:
    logits = _uniform_logits(2, 6)
    with pytest.raises(ValueError):
        completion_logprobs(logits, torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5))


def test_gradients_flow_to_the_logits() -> None:
    logits = _uniform_logits(2, 6).requires_grad_(True)
    token_ids = torch.randint(0, VOCAB, (2, 6))
    mask = build_completion_mask(2, [3, 2], 6)
    completion_logprobs(logits, token_ids, mask).sum().backward()

    assert logits.grad is not None
    # Only positions that scored a completion token may carry gradient. With prompt_len=2 and the
    # shift, that is positions 1..3 for row 0 and 1..2 for row 1.
    assert logits.grad[0, 0].abs().sum().item() == pytest.approx(0.0)
    assert logits.grad[0, 1].abs().sum().item() > 0


# --------------------------------------------------------------------------------------
# KL — the k3 estimator
# --------------------------------------------------------------------------------------


def test_kl_to_an_identical_reference_is_zero() -> None:
    logits = torch.randn(2, 6, VOCAB)
    token_ids = torch.randint(0, VOCAB, (2, 6))
    mask = build_completion_mask(2, [3, 3], 6)
    kl = sequence_kl(logits, logits.clone(), token_ids, mask)
    assert kl.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_kl_is_never_negative() -> None:
    """The k3 estimator is non-negative by construction; the naive log-ratio is not."""
    torch.manual_seed(0)
    for _ in range(50):
        token_ids = torch.randint(0, VOCAB, (4, 6))
        mask = build_completion_mask(2, [3, 3, 3, 3], 6)
        kl = sequence_kl(torch.randn(4, 6, VOCAB), torch.randn(4, 6, VOCAB), token_ids, mask)
        assert kl.min().item() >= -1e-6


def test_kl_grows_as_the_policy_diverges() -> None:
    token_ids = torch.randint(0, VOCAB, (1, 6))
    mask = build_completion_mask(2, [3], 6)
    reference = torch.zeros(1, 6, VOCAB)
    near = sequence_kl(reference + 0.1 * torch.randn(1, 6, VOCAB), reference, token_ids, mask)
    far = sequence_kl(reference + 5.0 * torch.randn(1, 6, VOCAB), reference, token_ids, mask)
    assert far.item() > near.item()


# --------------------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------------------


def test_uniform_logits_give_maximum_entropy() -> None:
    mask = build_completion_mask(2, [3], 6)
    assert entropy_over_completions(_uniform_logits(1, 6), mask) == pytest.approx(
        math.log(VOCAB), abs=1e-5
    )


def test_a_peaked_policy_has_near_zero_entropy() -> None:
    logits = torch.full((1, 6, VOCAB), -100.0)
    logits[:, :, 2] = 0.0
    mask = build_completion_mask(2, [3], 6)
    assert entropy_over_completions(logits, mask) == pytest.approx(0.0, abs=1e-4)


def test_entropy_ignores_prompt_positions() -> None:
    """Uncertainty over tokens the policy did not choose says nothing about its diversity."""
    logits = torch.full((1, 6, VOCAB), -100.0)
    logits[:, :, 2] = 0.0  # peaked everywhere
    logits[:, 0, :] = 0.0  # ...except the prompt's first position, which is uniform
    mask = build_completion_mask(3, [2], 6)
    assert entropy_over_completions(logits, mask) == pytest.approx(0.0, abs=1e-4)
