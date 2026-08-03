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

import re
from dataclasses import dataclass
from enum import Enum

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
