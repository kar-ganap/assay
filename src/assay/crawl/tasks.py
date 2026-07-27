"""Parametric task families for Phase 0.1, each with an explicit difficulty dial.

The task is an **instrument for making the four GRPO breakages legible**, not a benchmark. Two of
the four constrain the task itself: ablation **A** needs headroom (a curve long enough that "slower
and noisier" is distinguishable from "identical"), and ablation **D** needs a **dial**, so unanimous
groups can be constructed on demand.

Both families are procedurally generated, so they are contamination-immune by construction and the
answer distribution is *controlled* rather than inherited. That control matters: counting answers are
drawn uniformly from a declared range so the guessing floor is exactly ``1/range``, low enough that a
modal-answer strategy cannot compete with real skill and therefore cannot contaminate run 7.

See ``docs/phases/phase-0.1-grpo-by-hand-plan.md`` — *Task selection*.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from dataclasses import dataclass
from typing import ClassVar, Protocol

#: Counting answers are uniform over this inclusive range, so the guessing floor is 1/10.
COUNTING_ANSWER_RANGE: tuple[int, int] = (1, 10)

#: Instruction conflict, found 2026-07-27 by the coarse screen: the original wording was
#: "Reply with just the number in the form <answer>N</answer>." At 1B, *"just the number"* wins and
#: the format is dropped — 96.9% of completions came back as a bare integer, so parse_fail_rate was
#: 1.000 across every setting and the screen (correctly) rejected the whole grid.
#:
#: The example uses ``0``, which is outside the answer range of every setting, so a model that
#: parrots the example produces a *wrong answer* rather than an accidental hit.
_ANSWER_INSTRUCTION = (
    "End your reply with the answer inside answer tags, for example: <answer>0</answer>"
)


@dataclass(frozen=True)
class Prompt:
    """One task instance. ``answer`` is the canonical string the grader compares against."""

    prompt_id: str
    question: str
    answer: str
    family: str
    setting: str


class TaskFamily(Protocol):
    """A parametric family whose ``settings()`` are ordered easy -> hard."""

    name: str

    def settings(self) -> list[str]:
        """Difficulty rungs, ordered easiest first."""
        ...

    def generate(self, setting: str, n: int, *, seed: int) -> list[Prompt]:
        """Deterministically generate ``n`` prompts at ``setting``."""
        ...


# --------------------------------------------------------------------------------------
# Counting — dial is string length. Short completions, smooth dial, small answer space.
# --------------------------------------------------------------------------------------

_COUNTING_TEMPLATE = (
    "Count how many times the letter '{needle}' appears in this string:\n\n{haystack}\n\n"
    + _ANSWER_INSTRUCTION
)

_COUNTING_RE = re.compile(
    r"letter '(?P<needle>[a-z])' appears in this string:\n\n(?P<haystack>[a-z]+)\n"
)


class CountingFamily:
    """Count occurrences of a letter in a random lowercase string."""

    name = "counting"

    #: setting -> haystack length
    _LENGTHS: ClassVar[dict[str, int]] = {
        "count-L20": 20,
        "count-L40": 40,
        "count-L60": 60,
        "count-L90": 90,
    }

    def settings(self) -> list[str]:
        return list(self._LENGTHS)

    def generate(self, setting: str, n: int, *, seed: int) -> list[Prompt]:
        if setting not in self._LENGTHS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {self.settings()}")
        length = self._LENGTHS[setting]
        lo, hi = COUNTING_ANSWER_RANGE
        rng = random.Random(f"{self.name}:{setting}:{seed}")

        prompts = []
        for i in range(n):
            needle = rng.choice(string.ascii_lowercase)
            count = rng.randint(lo, hi)
            others = [c for c in string.ascii_lowercase if c != needle]
            chars = [needle] * count + [rng.choice(others) for _ in range(length - count)]
            rng.shuffle(chars)
            prompts.append(
                Prompt(
                    prompt_id=f"{self.name}-{setting}-{seed}-{i}",
                    question=_COUNTING_TEMPLATE.format(needle=needle, haystack="".join(chars)),
                    answer=str(count),
                    family=self.name,
                    setting=setting,
                )
            )
        return prompts


def parse_counting_question(question: str) -> tuple[str, str]:
    """Recover ``(haystack, needle)`` from a rendered question. Used to verify ground truth."""
    match = _COUNTING_RE.search(question)
    if match is None:
        raise ValueError("not a counting question")
    return match.group("haystack"), match.group("needle")


# --------------------------------------------------------------------------------------
# Arithmetic — dial is digit count x operation.
# --------------------------------------------------------------------------------------

_ARITHMETIC_TEMPLATE = "What is {lhs} {op} {rhs}?\n\n" + _ANSWER_INSTRUCTION

_ARITHMETIC_RE = re.compile(r"What is (?P<lhs>\d+) (?P<op>[+*]) (?P<rhs>\d+)\?")


class ArithmeticFamily:
    """Two-operand integer arithmetic."""

    name = "arithmetic"

    #: setting -> (op, lhs range, rhs range)
    _SETTINGS: ClassVar[dict[str, tuple[str, tuple[int, int], tuple[int, int]]]] = {
        "add-2digit": ("+", (10, 99), (10, 99)),
        "add-3digit": ("+", (100, 999), (100, 999)),
        "mul-2x1digit": ("*", (10, 99), (2, 9)),
        "mul-2x2digit": ("*", (10, 99), (10, 99)),
    }

    def settings(self) -> list[str]:
        return list(self._SETTINGS)

    def generate(self, setting: str, n: int, *, seed: int) -> list[Prompt]:
        if setting not in self._SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {self.settings()}")
        op, lhs_range, rhs_range = self._SETTINGS[setting]
        rng = random.Random(f"{self.name}:{setting}:{seed}")

        prompts = []
        for i in range(n):
            lhs = rng.randint(*lhs_range)
            rhs = rng.randint(*rhs_range)
            answer = lhs + rhs if op == "+" else lhs * rhs
            prompts.append(
                Prompt(
                    prompt_id=f"{self.name}-{setting}-{seed}-{i}",
                    question=_ARITHMETIC_TEMPLATE.format(lhs=lhs, op=op, rhs=rhs),
                    answer=str(answer),
                    family=self.name,
                    setting=setting,
                )
            )
        return prompts


def parse_arithmetic_question(question: str) -> tuple[int, str, int]:
    """Recover ``(lhs, op, rhs)`` from a rendered question. Used to verify ground truth."""
    match = _ARITHMETIC_RE.search(question)
    if match is None:
        raise ValueError("not an arithmetic question")
    return int(match.group("lhs")), match.group("op"), int(match.group("rhs"))


def template_fingerprint() -> str:
    """SHA-256 prefix over every prompt template.

    Goes into each run's manifest (``experiments/README.md`` requires prompt hashes). Without it,
    two result files produced by materially different prompts are indistinguishable — which already
    happened on 2026-07-27, when a few-shot run and a bare-instruction run were written with
    identical-looking provenance.
    """
    # NUL-separated, not concatenated: otherwise moving text between two templates leaves the
    # fingerprint unchanged.
    parts = (_ANSWER_INSTRUCTION, _COUNTING_TEMPLATE, _ARITHMETIC_TEMPLATE)
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def all_families() -> list[TaskFamily]:
    """Every family the sweep walks."""
    return [CountingFamily(), ArithmeticFamily()]
