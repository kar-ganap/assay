"""The sampler seam.

Everything in the harness runs against the ``Sampler`` protocol, so ``make check`` and every test
execute with **zero GPU** via ``FakeSampler``. The Modal-backed sampler is a thin shim behind the
same protocol — the dev machine (2019 Intel MBP, no MPS/CUDA) cannot run a model at all.
"""

from __future__ import annotations

import pytest

from assay.crawl import rewards, tasks
from assay.crawl.rewards import Outcome
from assay.crawl.sampling import Completion, FakeSampler, SamplerConfig

CFG = SamplerConfig()


def _prompts(n: int = 8) -> list[tasks.Prompt]:
    return tasks.CountingFamily().generate(tasks.CountingFamily().settings()[0], n=n, seed=0)


def test_returns_k_completions_per_prompt() -> None:
    prompts = _prompts(5)
    out = FakeSampler(p_correct=0.5, seed=0).sample(prompts, k=8, cfg=CFG)
    assert len(out) == 5
    assert all(len(row) == 8 for row in out)
    assert all(isinstance(c, Completion) for row in out for c in row)


def test_is_deterministic_under_seed() -> None:
    prompts = _prompts()
    a = FakeSampler(p_correct=0.5, seed=3).sample(prompts, k=8, cfg=CFG)
    b = FakeSampler(p_correct=0.5, seed=3).sample(prompts, k=8, cfg=CFG)
    assert [[c.text for c in row] for row in a] == [[c.text for c in row] for row in b]


def test_different_seeds_differ() -> None:
    prompts = _prompts()
    a = FakeSampler(p_correct=0.5, seed=3).sample(prompts, k=8, cfg=CFG)
    b = FakeSampler(p_correct=0.5, seed=4).sample(prompts, k=8, cfg=CFG)
    assert [[c.text for c in row] for row in a] != [[c.text for c in row] for row in b]


def test_completions_round_trip_through_the_real_grader() -> None:
    """The fake must produce text the production grader actually parses, or it proves nothing."""
    prompts = _prompts(4)
    out = FakeSampler(p_correct=1.0, seed=0).sample(prompts, k=4, cfg=CFG)
    for prompt, row in zip(prompts, out, strict=True):
        for completion in row:
            assert rewards.grade_binary(completion.text, prompt.answer).outcome is Outcome.CORRECT


def test_hits_the_requested_pass_rate() -> None:
    prompts = _prompts(200)
    out = FakeSampler(p_correct=0.25, seed=0).sample(prompts, k=8, cfg=CFG)
    grades = [
        rewards.grade_binary(c.text, p.answer)
        for p, row in zip(prompts, out, strict=True)
        for c in row
    ]
    observed = sum(g.outcome is Outcome.CORRECT for g in grades) / len(grades)
    assert observed == pytest.approx(0.25, abs=0.05)


def test_emits_parse_failures_at_the_configured_rate() -> None:
    prompts = _prompts(200)
    out = FakeSampler(p_correct=0.5, p_parse_fail=0.3, seed=0).sample(prompts, k=8, cfg=CFG)
    grades = [
        rewards.grade_binary(c.text, p.answer)
        for p, row in zip(prompts, out, strict=True)
        for c in row
    ]
    observed = sum(g.outcome is Outcome.PARSE_FAIL for g in grades) / len(grades)
    assert observed == pytest.approx(0.3, abs=0.05)


def test_per_prompt_pass_rate_mapping() -> None:
    """Needed to build the bimodal fixture: identical mean, opposite dead-group fraction."""
    prompts = _prompts(100)
    per_prompt = {p.prompt_id: (0.9 if i % 2 == 0 else 0.1) for i, p in enumerate(prompts)}
    out = FakeSampler(p_correct=per_prompt, seed=0).sample(prompts, k=8, cfg=CFG)

    easy, hard = [], []
    for prompt, row in zip(prompts, out, strict=True):
        n_correct = sum(
            rewards.grade_binary(c.text, prompt.answer).outcome is Outcome.CORRECT for c in row
        )
        (easy if per_prompt[prompt.prompt_id] == 0.9 else hard).append(n_correct)

    assert sum(easy) / (len(easy) * 8) == pytest.approx(0.9, abs=0.1)
    assert sum(hard) / (len(hard) * 8) == pytest.approx(0.1, abs=0.1)


def test_reports_token_counts() -> None:
    out = FakeSampler(p_correct=0.5, seed=0).sample(_prompts(2), k=2, cfg=CFG)
    assert all(c.n_tokens > 0 for row in out for c in row)


def test_respects_max_new_tokens() -> None:
    cfg = SamplerConfig(max_new_tokens=16)
    out = FakeSampler(p_correct=0.5, seed=0).sample(_prompts(4), k=4, cfg=cfg)
    assert all(c.n_tokens <= 16 for row in out for c in row)
