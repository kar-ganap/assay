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
from collections.abc import Sequence
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
    parts = (_ANSWER_INSTRUCTION, _COUNTING_TEMPLATE, _ARITHMETIC_TEMPLATE,
             _COUNTDOWN_INSTRUCTION, _COUNTDOWN_TEMPLATE, _WORD_PROBLEM_TEMPLATE,
             *(part for scenario in _WORD_PROBLEM_SCENARIOS for part in scenario))
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Countdown — the R0 substrate. Dial is the number count: more numbers, larger search space.
# --------------------------------------------------------------------------------------

#: Countdown asks for an *expression*, not an integer, so it cannot reuse ``_ANSWER_INSTRUCTION``.
#: Phrasing matters more here than in the arithmetic families: Phase 0.1 found that instruction
#: conflict ("just the number" beating a format demand) drove parse_fail to 1.000 across a whole
#: grid. This wording asks for one thing only, and names the operators to keep the parser's job
#: narrow.
_COUNTDOWN_INSTRUCTION = (
    "Use each number at most once, with + - * and / only. "
    "Work step by step, then end your reply with the final expression on its own."
)

_COUNTDOWN_TEMPLATE = (
    "Using the numbers {numbers}, write an arithmetic expression that equals {target}.\n\n"
    + _COUNTDOWN_INSTRUCTION
)

_COUNTDOWN_RE = re.compile(
    r"Using the numbers (?P<numbers>[\d,\s]+), write an arithmetic expression that equals "
    r"(?P<target>-?\d+)"
)


def render_countdown_question(numbers: Sequence[int], target: int) -> str:
    """The rendered prompt. Exposed so graders and tests build real input, never a lookalike."""
    return _COUNTDOWN_TEMPLATE.format(
        numbers=", ".join(str(n) for n in numbers), target=target
    )


def parse_countdown_question(question: str) -> tuple[list[int], int]:
    """Recover ``(numbers, target)`` from a rendered question.

    Same role as ``parse_arithmetic_question``: ground truth travels *in the prompt*, so a grader
    needs nothing but the text it was shown. For Countdown this is load-bearing rather than
    convenient — the grader must know which numbers were offered to reject a reused one, and
    ``Prompt.answer`` carries only the target.
    """
    match = _COUNTDOWN_RE.search(question)
    if match is None:
        raise ValueError("not a countdown question")
    numbers = [int(part) for part in match.group("numbers").split(",")]
    return numbers, int(match.group("target"))


class CountdownFamily:
    """Reach a target from a small set of integers, using each at most once.

    **Every instance is solvable by construction**, and that is not a nicety. Phase 0.3's screen
    infers the dead-group fraction from the measured pass rate; an unsolvable instance makes a zero
    reward uninformative, because "the model could not reason" and "there was nothing to find" become
    indistinguishable. So the generator builds an expression *first* and takes the target from it,
    rather than sampling a target and hoping.
    """

    name = "countdown"

    #: setting -> (how many numbers, value range to draw from, allowed target range)
    #:
    #: Targets are constrained to a Countdown-like band rather than left to whatever the random
    #: expression produced. Without the bound a multiplicative fold yields targets like 271,118,
    #: which is *solvable* but measures "multiply everything" instead of search — and would make the
    #: base-rate screen describe the wrong skill. Difficulty is carried by the number count, which
    #: is what grows the search space.
    _SETTINGS: ClassVar[dict[str, tuple[int, tuple[int, int], tuple[int, int]]]] = {
        "cd-3": (3, (1, 25), (20, 300)),
        # M3 (pre-registered 2026-08-03). Three numbers in every variant, deliberately: the number
        # count is what grows the space of expression trees, so holding it fixed and shrinking only
        # the operand magnitudes gives an *identical structural search space* and varies only the
        # per-step arithmetic burden. That is what lets the screen separate "can it do the
        # arithmetic" from "can it search" — see docs/phases/phase-0.3-r0-plan.md §M3.
        "cd-3-easy": (3, (1, 10), (10, 60)),
        "cd-3-mid": (3, (1, 15), (15, 120)),
        "cd-4": (4, (1, 50), (100, 999)),
        "cd-5": (5, (1, 75), (100, 999)),
        "cd-6": (6, (1, 100), (100, 999)),
    }

    def settings(self) -> list[str]:
        return list(self._SETTINGS)

    def generate(self, setting: str, n: int, *, seed: int) -> list[Prompt]:
        if setting not in self._SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {self.settings()}")
        count, value_range, target_range = self._SETTINGS[setting]
        rng = random.Random(f"{self.name}:{setting}:{seed}")

        prompts = []
        for i in range(n):
            numbers, target = self._solvable_instance(rng, count, value_range, target_range)
            prompts.append(
                Prompt(
                    prompt_id=f"{self.name}-{setting}-{seed}-{i}",
                    question=render_countdown_question(numbers, target),
                    answer=str(target),
                    family=self.name,
                    setting=setting,
                )
            )
        return prompts

    def _solvable_instance(
        self,
        rng: random.Random,
        count: int,
        value_range: tuple[int, int],
        target_range: tuple[int, int],
    ) -> tuple[list[int], int]:
        """Draw numbers, then fold a random subset into a target with random operators.

        Retries rather than accepting a degenerate target: a target already present among the
        numbers, or <= 0, is solvable but measures nothing about search. Division is only applied
        when it divides exactly, so the target stays an integer and the grader needs no tolerance.
        """
        for _ in range(200):
            numbers = [rng.randint(*value_range) for _ in range(count)]
            # Use at least 3 of them, so the answer is never a single number or one operation.
            k = rng.randint(3, count)
            chosen = rng.sample(numbers, k)
            total = float(chosen[0])
            for operand in chosen[1:]:
                op = rng.choice("++-*/")  # '+' twice: bias toward reachable, non-huge targets
                if op == "+":
                    total += operand
                elif op == "-":
                    total -= operand
                elif op == "*":
                    total *= operand
                elif operand != 0 and total % operand == 0:
                    total /= operand
                else:
                    total += operand
            target = int(total)
            if total == target and target_range[0] <= target <= target_range[1] \
                    and target not in numbers:
                return numbers, target
        # Fallback: fold three numbers additively, which always lands in range for these settings.
        # Still solvable and still requires combining three terms, just less varied.
        numbers = [rng.randint(*value_range) for _ in range(count)]
        return numbers, sum(numbers[:3])


# --------------------------------------------------------------------------------------
# Word problems — the substrate R1 proved we do not have
# --------------------------------------------------------------------------------------
#
# R1 could not measure a proxy-true gap at all, because neither Crawl substrate has both halves:
#
#   story      reachable hack word (1.35 / 0.59 / 20.96%)  ·  NO ground truth  (r_true == 0)
#   arithmetic ground truth                                ·  NO reachable hack (0/4096)
#
# `CLAUDE.md` §15 now records "a substrate needs a reachable exploit AND checkable ground truth"
# as a design requirement on `bisect` rather than an assumption. This family is the cheapest
# candidate that could satisfy both at once, and exists to be *screened*, not assumed:
#
#   * the question is **prose**, so an English hack word has somewhere to appear -- which is why
#     arithmetic scored 0/4096 (a completion of digits has no room for "ocean");
#   * the answer is an integer **known by construction**, so `r_true` is checkable without a
#     held-out suite, an LLM judge, or any labelling.
#
# Generated rather than borrowed (GSM8K etc.) for the same reason `CountdownFamily` is: a
# procedural generator makes ground truth exact by construction, keeps the repo self-contained,
# and lets difficulty be a dial instead of a property of someone else's dataset.
#
# **Admission is not decided here.** Whether the measured base rate and pass rate clear their
# bands is `docs/phases/phase-0.5-substrate-screen.md`, pre-registered before the run.

_WORD_PROBLEM_TEMPLATE = "{story}\n\n{question}\n\n" + _ANSWER_INSTRUCTION

#: (opening, verb-phrase for the increase, verb-phrase for the decrease, unit, question)
_WORD_PROBLEM_SCENARIOS: tuple[tuple[str, str, str, str, str], ...] = (
    ("A market stall starts the morning with {start} {unit}.",
     "A delivery arrives with {up} more.", "Customers buy {down}.",
     "apples", "How many {unit} are left at the stall?"),
    ("A library shelf holds {start} {unit}.",
     "A donation adds {up}.", "Readers borrow {down}.",
     "books", "How many {unit} remain on the shelf?"),
    ("A warehouse records {start} {unit} in stock.",
     "A shipment brings {up}.", "An order removes {down}.",
     "crates", "How many {unit} are in stock now?"),
    ("A school hall is set out with {start} {unit}.",
     "The caretaker adds {up}.", "Another room borrows {down}.",
     "chairs", "How many {unit} are in the hall?"),
    ("A nursery counts {start} {unit} in the greenhouse.",
     "A grower delivers {up}.", "A landscaper collects {down}.",
     "seedlings", "How many {unit} are in the greenhouse?"),
    ("A depot logs {start} {unit} at opening.",
     "A transfer adds {up}.", "A dispatch takes {down}.",
     "parcels", "How many {unit} are at the depot?"),
)


class WordProblemFamily:
    """Prose word problems with an integer answer known by construction.

    The only family in the repo with **both** properties R1 showed a usable substrate needs: free
    text for a hack to live in, and checkable ground truth. Screened, not assumed.
    """

    name = "wordproblem"

    #: setting -> (start range, delta range, n_steps). More steps = more arithmetic, same prose.
    _SETTINGS: ClassVar[dict[str, tuple[tuple[int, int], tuple[int, int], int]]] = {
        "wp-2step-2digit": ((20, 99), (2, 19), 2),
        "wp-2step-3digit": ((100, 999), (10, 99), 2),
        "wp-3step-2digit": ((30, 99), (2, 19), 3),
    }

    def settings(self) -> list[str]:
        return list(self._SETTINGS)

    def generate(self, setting: str, n: int, *, seed: int) -> list[Prompt]:
        if setting not in self._SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {self.settings()}")
        start_range, delta_range, n_steps = self._SETTINGS[setting]
        rng = random.Random(f"{self.name}:{setting}:{seed}")

        prompts = []
        for i in range(n):
            opening, up_phrase, down_phrase, unit, question = rng.choice(_WORD_PROBLEM_SCENARIOS)
            start = rng.randint(*start_range)
            total = start
            lines = [opening.format(start=start, unit=unit)]
            for step in range(n_steps):
                delta = rng.randint(*delta_range)
                # Alternate up/down, but never let the running total go negative -- a negative
                # count of apples is a different task and the model would be right to balk.
                if step % 2 == 0 and total + delta >= 0:
                    total += delta
                    lines.append(up_phrase.format(up=delta))
                else:
                    delta = min(delta, total)
                    total -= delta
                    lines.append(down_phrase.format(down=delta))
            prompts.append(
                Prompt(
                    prompt_id=f"{self.name}-{setting}-{seed}-{i}",
                    question=_WORD_PROBLEM_TEMPLATE.format(
                        story=" ".join(lines), question=question.format(unit=unit)
                    ),
                    answer=str(total),
                    family=self.name,
                    setting=setting,
                )
            )
        return prompts


def all_families() -> list[TaskFamily]:
    """Every family the sweep walks.

    `WordProblemFamily` is registered — unlike `STORY_PROMPTS` below — because it *is* scoreable:
    its answer is known by construction, which is the whole reason it exists.
    """
    return [CountingFamily(), ArithmeticFamily(), CountdownFamily(), WordProblemFamily()]

# ======================================================================================
# Free-text prompts — R1's substrate
# ======================================================================================

#: **Deliberately not registered in `all_families()`.** A story has no ground truth, so it cannot be
#: graded by the sweep machinery and would break any caller that assumes every family is scoreable.
#: R1 needs prompts, not a task family.
#:
#: Adopted after G2 falsified the first choice: on `add-3digit` the hack words scored **0 / 4096**
#: with a median completion of **8 tokens**. Base rate is a property of the task *and* its length,
#: and arithmetic supplies neither. Prime's own experiment is a story task.
#:
#: These are open-ended and evocative but **name no hack word or obvious synonym** — the base rate
#: has to come from the policy's vocabulary, not from us seeding it, or R1-P's independent variable
#: is something we chose rather than something we measured.
STORY_PROMPTS: tuple[str, ...] = (
    "Write a short story about an ancient forest.",
    "Write a short story about a traveller who arrives somewhere unexpected.",
    "Write a short story about something long buried coming to light.",
    "Write a short story about the last house on a quiet road.",
    "Write a short story about a promise kept many years later.",
    "Write a short story about a door that was never opened.",
    "Write a short story about two strangers waiting out a storm.",
    "Write a short story about a map with one place missing.",
)


def build_story_dataset(n: int = 2000, seed: int = 0) -> list[dict[str, str]]:
    """Free-text prompts, cycled in a seeded shuffle. ``answer`` is empty by construction."""
    rng = random.Random(f"story:{seed}")
    order = list(range(len(STORY_PROMPTS)))
    rows: list[dict[str, str]] = []
    while len(rows) < n:
        rng.shuffle(order)
        for i in order:
            rows.append({"question": STORY_PROMPTS[i], "answer": ""})
            if len(rows) == n:
                break
    return rows
