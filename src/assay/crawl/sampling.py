"""The sampler seam.

Everything in the harness runs against the ``Sampler`` protocol so that ``make check`` and every
test execute with **zero GPU**. That is not a nicety: the dev machine is a 2019 Intel MacBook Pro
with no MPS and no CUDA, and ``vllm`` will not install on macOS x86_64. The real sampler is a thin
Modal-hosted shim behind this same protocol (``assay.modal_app``).

Generation goes through HF ``generate`` rather than vLLM so the screen and the hand-rolled GRPO loop
share one code path — a base rate measured on a different sampler is not the base rate that predicts
training.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from assay.crawl.tasks import Prompt


@dataclass(frozen=True)
class SamplerConfig:
    """Pinned into every ``manifest.json``. Must match the training sampler exactly."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    seed: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int


class Sampler(Protocol):
    def sample(
        self, prompts: Sequence[Prompt], *, k: int, cfg: SamplerConfig
    ) -> list[list[Completion]]:
        """Return ``k`` completions per prompt, in prompt order."""
        ...


class FakeSampler:
    """Model-free, deterministic test double.

    Draws a marginal outcome per rollout: ``CORRECT`` with probability ``p_correct``, ``PARSE_FAIL``
    with probability ``p_parse_fail``, otherwise a wrong answer. ``p_correct`` may be a per-prompt
    mapping, which is what lets the tests build a bimodal task set with the *same* mean pass rate as
    a centred one and a wildly different dead-group fraction.

    It reads ``Prompt.answer`` — a real sampler never does. That is the price of a double that emits
    text the production grader actually parses, which is the only kind worth having.

    **Integer-answer families only** (``counting``, ``arithmetic``). It emits ``<answer>N</answer>``,
    which ``grade_countdown`` correctly rejects — so a Countdown sweep run through this double would
    report 100% parse-fail and validate nothing, silently. It raises instead; see
    ``tests/test_crawl_countdown.py::_CountdownSampler`` for a double that emits real expressions.
    """

    def __init__(
        self,
        *,
        p_correct: float | Mapping[str, float],
        p_parse_fail: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._p_correct = p_correct
        self._p_parse_fail = p_parse_fail
        self._seed = seed

    def _p_for(self, prompt: Prompt) -> float:
        if isinstance(self._p_correct, Mapping):
            return self._p_correct[prompt.prompt_id]
        return self._p_correct

    def sample(
        self, prompts: Sequence[Prompt], *, k: int, cfg: SamplerConfig
    ) -> list[list[Completion]]:
        out: list[list[Completion]] = []
        for prompt in prompts:
            p_correct = self._p_for(prompt)
            if p_correct + self._p_parse_fail > 1.0:
                raise ValueError("p_correct + p_parse_fail must not exceed 1.0")
            rng = random.Random(f"{self._seed}:{cfg.seed}:{prompt.prompt_id}")
            row = [self._one(prompt, rng, p_correct, cfg) for _ in range(k)]
            out.append(row)
        return out

    def _one(
        self, prompt: Prompt, rng: random.Random, p_correct: float, cfg: SamplerConfig
    ) -> Completion:
        if prompt.family == "countdown":
            raise ValueError(
                "FakeSampler emits <answer>N</answer>, which grade_countdown rejects; a Countdown "
                "sweep through it would silently report 100% parse-fail. Use a sampler that emits "
                "arithmetic expressions."
            )
        draw = rng.random()
        if draw < p_correct:
            text = f"Let me count carefully. <answer>{prompt.answer}</answer>"
        elif draw < p_correct + self._p_parse_fail:
            # Must contain **no integer at all**. Under R_binary's last-integer extractor a
            # PARSE_FAIL is a genuine non-answer; text like "the answer is 7" would grade CORRECT.
            text = "I am not sure how to work this one out."
        else:
            wrong = int(prompt.answer) + rng.choice([-3, -2, -1, 1, 2, 3])
            text = f"Let me count carefully. <answer>{wrong}</answer>"
        return Completion(text=text, n_tokens=min(cfg.max_new_tokens, max(1, len(text) // 4)))
