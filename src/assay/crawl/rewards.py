"""The Phase 0.1 graders — three reward variants over one task set.

No single reward makes all four breakages visible, so the phase swaps the grader while holding the
tasks fixed. That is the move the whole project is built on, rehearsed at the smallest possible
scale (``docs/phases/phase-0.1-grpo-by-hand-plan.md``, *Design*).

**The load-bearing distinction is ``PARSE_FAIL`` vs ``WRONG_ANSWER``.** Collapsing them into "not
correct" hides the case where RL is learning formatting rather than the skill — which would make
every curve in the seven-run ladder a formatting curve wearing a skill costume. The outcome is
therefore structural, not inferred after the fact.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from assay.crawl.tasks import parse_countdown_question

#: An integer, with optional thousands separators (the model emits ``1,125`` for 4-digit results).
_INTEGER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+|-?\d+")

#: Strict ``<answer>N</answer>``. The final one wins — small models restate, and the last is what
#: they commit to. Used **only** by ``grade_format_only`` (ablation B), never by ``grade_binary``.
_TAGGED_RE = re.compile(r"<answer>\s*(-?\d{1,3}(?:,\d{3})+|-?\d+)\s*</answer>")

#: Ablation C's tie-breaker weight. Small enough to look negligible; the std normalisation
#: amplifies it to full gradient magnitude anyway. That is the point.
TIEBREAK_WEIGHT = 0.001


def grader_fingerprint() -> dict[str, str]:
    """Identifies the grading semantics for a run's manifest.

    ``R_binary`` changed from strict-tag to last-integer on 2026-07-27. Result files written either
    side of that change are **not comparable**, and nothing in the manifest said so. This is what
    says so.
    """
    return {
        "r_binary_extractor": "extract_final_integer:last-integer-anywhere",
        "r_format_extractor": "extract_tagged_answer:strict-answer-tag",
        "integer_pattern": _INTEGER_RE.pattern,
        "tagged_pattern": _TAGGED_RE.pattern,
        "tiebreak_weight": repr(TIEBREAK_WEIGHT),
    }


class Outcome(str, Enum):
    """Mutually exclusive and exhaustive: every completion lands in exactly one."""

    CORRECT = "correct"
    WRONG_ANSWER = "wrong_answer"
    PARSE_FAIL = "parse_fail"


@dataclass(frozen=True)
class Grade:
    outcome: Outcome
    extracted: str | None
    reward: float


def extract_final_integer(completion: str) -> str | None:
    """Last integer anywhere in the completion — the standard RLVR extractor.

    **Why not the strict tag** (decided 2026-07-27 on measured evidence). Strict-tag compliance is
    only 26% at baseline for Llama-3.2-1B, and worse, it is **confounded with difficulty**: harder
    problems make the model reason out loud, and the longer it reasons the less reliably it closes
    with a tag. Measured `parse_fail_rate` rose monotonically as pass rate fell, in *both* task
    families independently:

    ===================  =======  ============
    setting              pass@1   parse_fail
    ===================  =======  ============
    mul-2x1digit          1.000        0.000
    add-2digit            0.914        0.062
    mul-2x2digit          0.438        0.258
    add-3digit            0.398        0.398
    ===================  =======  ============

    A strict grader therefore does not filter *formatting* problems — it filters **hard** problems,
    excluding precisely the difficulty band the screen exists to locate. No prompt wording fixes
    that; few-shot lifted compliance to 63% and left the monotone confound intact.

    The last integer recovers the model's answer from every shape it actually produces —
    ``<answer>7</answer>``, ``<a>7</a>``, ``(answer) 7``, ``45 * 8 = 360``, ``...appears 5 times`` —
    giving 100% parseable, so the ladder measures skill rather than format-learning. The strict tag
    survives as ``grade_format_only``, where being a *different* grader is the entire point.
    """
    matches = _INTEGER_RE.findall(completion)
    return matches[-1].replace(",", "") if matches else None


def extract_tagged_answer(completion: str) -> str | None:
    """Last strictly-formatted ``<answer>N</answer>``, or ``None``. Ablation B's extractor."""
    matches = _TAGGED_RE.findall(completion)
    return str(matches[-1]).replace(",", "") if matches else None


def grade_binary(completion: str, expected: str) -> Grade:
    """``R_binary`` — the ladder's reward (runs 1-7, ablations A and D).

    Binary on *correctness*, lenient on shape. ``PARSE_FAIL`` now means "emitted no number at all",
    which is a genuine non-answer rather than a formatting quibble.
    """
    extracted = extract_final_integer(completion)
    if extracted is None:
        return Grade(outcome=Outcome.PARSE_FAIL, extracted=None, reward=0.0)
    if int(extracted) == int(expected):
        return Grade(outcome=Outcome.CORRECT, extracted=extracted, reward=1.0)
    return Grade(outcome=Outcome.WRONG_ANSWER, extracted=extracted, reward=0.0)


def grade_format_only(completion: str, expected: str) -> Grade:
    """``R_format`` — ablation **B**. Shape is rewarded, content ignored.

    A degenerate high-reward string exists *by construction* (any constant ``<answer>0</answer>``
    always pays), so "remove the KL leash and the policy collapses onto one string" becomes
    reachable rather than merely hoped for. This is battery axis **A2** (grader degeneracy) at n=1.
    """
    del expected  # deliberately unused — that *is* the pathology
    extracted = extract_tagged_answer(completion)
    if extracted is None:
        return Grade(outcome=Outcome.PARSE_FAIL, extracted=None, reward=0.0)
    return Grade(outcome=Outcome.CORRECT, extracted=extracted, reward=1.0)


def grade_pair(
    variant: str, completion: str, expected: str, *, completion_tokens: int
) -> tuple[Grade, Grade]:
    """Return ``(proxy, true)`` for one completion.

    **The true reward is always ``R_binary`` and is never returned as the training signal.** That
    asymmetry is the whole proxy/true design: the policy maximises ``proxy``, and ``true`` is only
    ever measured. The moment a true grader is trained on, it stops measuring generalisation and
    becomes another proxy (``assay.grader.HeldOutGrader``).

    ``gap = proxy.reward - true.reward``:

    - ``"binary"``   proxy is true, so gap == 0. Correct for ladder runs 1-7.
    - ``"format"``   ablation B. Gap grows as the policy emits well-formed tags around wrong
      answers. Starts *negative* (baseline tag compliance is below baseline correctness).
    - ``"tiebreak"`` ablation C. Gap is exactly ``TIEBREAK_WEIGHT * completion_tokens``, so it
      measures precisely how much reward the padding bought.
    """
    true = grade_binary(completion, expected)
    if variant == "binary":
        return true, true
    if variant == "format":
        return grade_format_only(completion, expected), true
    if variant == "tiebreak":
        return grade_tiebroken(completion, expected, completion_tokens=completion_tokens), true
    raise ValueError(f"unknown reward variant {variant!r}")


def grade_tiebroken(completion: str, expected: str, *, completion_tokens: int) -> Grade:
    """``R_tiebreak`` — ablation **C**. ``R_binary`` plus a tie-breaker that looks negligible.

    On a *unanimous* group this is the difference between a wasted step and a full-magnitude step
    spent optimising completion length: the std normalisation is scale-invariant, so it amplifies
    the ``0.001`` term to exactly the magnitude a real 1-vs-7 signal would produce. Goodhart out of
    the optimiser's own arithmetic. See ``assay.crawl.advantage``.
    """
    base = grade_binary(completion, expected)
    return Grade(
        outcome=base.outcome,
        extracted=base.extracted,
        reward=base.reward + TIEBREAK_WEIGHT * completion_tokens,
    )


# --------------------------------------------------------------------------------------
# Countdown (Phase 0.3 / R0)
# --------------------------------------------------------------------------------------

#: An arithmetic expression: digits and operators, at least one operator. Anchored on a digit or
#: open-paren at each end so trailing prose does not get swept in.
_EXPRESSION_RE = re.compile(r"[\d(][\d\s,()+\-*/.]*[\d)]")

#: Unicode operators and separators small models emit. Normalised rather than rejected, because a
#: strict grader filters HARD problems rather than badly-formatted ones — measured in Phase 0.1,
#: where parse-failure rose monotonically as pass rate fell.
_OPERATOR_ALIASES = {"×": "*", "x": "*", "X": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}


def _normalise_expression(text: str) -> str:
    for old, new in _OPERATOR_ALIASES.items():
        text = text.replace(old, new)
    return text


def _eval_arithmetic(expr: str) -> float | None:
    """Evaluate under an operator allowlist. **Never ``eval``.**

    A grader runs on text a model chose, so arbitrary-code execution is a real exposure rather than a
    hypothetical one. Returns ``None`` on anything it cannot evaluate — bad syntax, an unsupported
    node, division by zero — so the caller can report PARSE_FAIL rather than a spurious wrong answer.
    """
    expr = _normalise_expression(expr).replace(",", "")
    if not re.fullmatch(r"[\d\s()+\-*/.]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> float | None:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            inner = walk(node.operand)
            return None if inner is None else (inner if isinstance(node.op, ast.UAdd) else -inner)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return None if abs(right) < 1e-12 else left / right
        return None

    return walk(tree)


def _literals(expr: str) -> list[int] | None:
    """Integer literals in the expression, for the used-at-most-once check."""
    cleaned = _normalise_expression(expr).replace(",", "")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None
    return [
        int(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    ]


def grade_countdown(completion: str, question: str, *, tol: float = 1e-6) -> Grade:
    """``R_countdown`` — reach the target using each offered number at most once.

    Two failure modes are kept structurally distinct, as everywhere in this project:

    - **PARSE_FAIL** — no expression was found, or one was found and could not be evaluated. The
      model said nothing gradeable.
    - **WRONG_ANSWER** — an expression was evaluated and is *illegal or off-target*: it reuses a
      number, uses one that was never offered, or simply misses.

    Reusing a number scores zero even when the arithmetic hits the target. ``5 * 5 = 25`` is not a
    solution when only one 5 was offered, and a grader that accepted it would be measuring something
    easier than the task.

    The **last** parseable expression wins: small models restate their working, and the last thing
    they write is what they commit to — the same convention ``extract_final_integer`` uses.
    """
    numbers, target = parse_countdown_question(question)

    candidates = _EXPRESSION_RE.findall(_normalise_expression(completion))
    for raw in reversed(candidates):
        expr = raw.strip()
        if not re.search(r"[+\-*/]", expr):
            continue  # a bare number is not an expression
        used = _literals(expr)
        value = _eval_arithmetic(expr)
        if used is None or value is None:
            continue
        available = Counter(numbers)
        if Counter(used) - available:
            return Grade(outcome=Outcome.WRONG_ANSWER, extracted=expr, reward=0.0)
        if abs(value - target) < tol:
            return Grade(outcome=Outcome.CORRECT, extracted=expr, reward=1.0)
        return Grade(outcome=Outcome.WRONG_ANSWER, extracted=expr, reward=0.0)
    return Grade(outcome=Outcome.PARSE_FAIL, extracted=None, reward=0.0)
