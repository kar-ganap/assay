"""Countdown — the R0 substrate. Generation, solvability, and a deliberately permissive grader.

**Why the solvability test is the load-bearing one.** Phase 0.3's whole first stage is a base-rate
screen: measure what fraction of prompts the base policy solves, and infer the dead-group fraction
from it. If some generated instances have *no solution*, the screen conflates "the model cannot
reason" with "there was nothing to find", and every number downstream describes the wrong thing.
``test_every_generated_instance_is_actually_solvable`` searches for a solution independently rather
than trusting the generator's construction.

**Why the grader is permissive.** Phase 0.1 established, on measured evidence, that a strict grader
does not filter *formatting* problems — it filters **hard** problems, because parse-failure rises
monotonically as pass rate falls (add-2digit 0.062 → add-3digit 0.398). A strict Countdown grader
would therefore bias the screen exactly where it matters most.

Zero GPU, zero network.
"""

from __future__ import annotations

import functools
import itertools

import pytest

from assay.crawl.rewards import Outcome, grade_countdown
from assay.crawl.tasks import CountdownFamily, all_families, parse_countdown_question

# --------------------------------------------------------------------------------------
# An independent solver, used only by the tests
# --------------------------------------------------------------------------------------


@functools.cache
def _reachable_key(state: tuple[float, ...], target: int) -> bool:
    """Memoised core. Subproblems repeat heavily once numbers start combining, and without the
    cache a 6-number instance takes tens of seconds — enough to make `make check` unusable."""
    tol = 1e-6
    if any(abs(n - target) < tol for n in state):
        return True
    for i, j in itertools.combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = tuple(n for k, n in enumerate(state) if k not in (i, j))
        options = [a + b, a - b, b - a, a * b]
        if abs(b) > tol:
            options.append(a / b)
        if abs(a) > tol:
            options.append(b / a)
        for combined in options:
            # Round to keep the cache key stable across float paths that agree to 9 places.
            if _reachable_key(tuple(sorted((*rest, round(combined, 9)))), target):
                return True
    return False


@functools.cache
def _solve_key(state: tuple[tuple[float, str], ...], target: int) -> str | None:
    """Like ``_reachable_key`` but returns the expression, so tests can emit a *real* solution."""
    tol = 1e-6
    for value, expr in state:
        if abs(value - target) < tol:
            return expr
    for i, j in itertools.combinations(range(len(state)), 2):
        (a, ea), (b, eb) = state[i], state[j]
        rest = tuple(x for k, x in enumerate(state) if k not in (i, j))
        options = [(a + b, f"({ea} + {eb})"), (a - b, f"({ea} - {eb})"),
                   (b - a, f"({eb} - {ea})"), (a * b, f"({ea} * {eb})")]
        if abs(b) > tol:
            options.append((a / b, f"({ea} / {eb})"))
        if abs(a) > tol:
            options.append((b / a, f"({eb} / {ea})"))
        for value, expr in options:
            found = _solve_key(tuple(sorted((*rest, (round(value, 9), expr)))), target)
            if found is not None:
                return found
    return None


def _solve(numbers: list[int], target: int) -> str | None:
    """An expression over ``numbers`` equal to ``target``, or None."""
    return _solve_key(tuple(sorted((float(n), str(n)) for n in numbers)), target)


def _reachable(numbers: list[float], target: int) -> bool:
    """Can ``target`` be made from a subset of ``numbers``? Exhaustive, independent of the generator.

    Deliberately *not* a restatement of how instances are built — it searches, so it can catch a
    generator that produces something it believes is solvable and is not.
    """
    return _reachable_key(tuple(sorted(round(n, 9) for n in numbers)), target)


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def test_countdown_is_registered_as_a_family() -> None:
    assert any(f.name == "countdown" for f in all_families())


def test_settings_are_ordered_easy_to_hard() -> None:
    """The dial the screen sweeps. More numbers = larger search space = harder."""
    settings = CountdownFamily().settings()
    assert len(settings) >= 2
    sizes = [len(parse_countdown_question(p.question)[0])
             for s in settings
             for p in CountdownFamily().generate(s, 1, seed=0)]
    assert sizes == sorted(sizes), f"settings not ordered easy->hard: {list(zip(settings, sizes))}"


def test_generation_is_deterministic_from_seed() -> None:
    a = CountdownFamily().generate("cd-4", 12, seed=3)
    b = CountdownFamily().generate("cd-4", 12, seed=3)
    c = CountdownFamily().generate("cd-4", 12, seed=4)
    assert [p.question for p in a] == [p.question for p in b]
    assert [p.question for p in a] != [p.question for p in c]


def test_every_generated_instance_is_actually_solvable() -> None:
    """**The screen's validity rests on this.**

    An unsolvable instance makes a zero reward uninformative: it cannot be told apart from a
    reasoning failure, and the base rate it produces describes neither. Verified by independent
    search, not by trusting the generator.
    """
    # Fewer instances at the wide settings: cost grows steeply with the number count, and the
    # property under test does not depend on sample size the way a statistic would.
    per_setting = {"cd-3": 12, "cd-4": 12, "cd-5": 6, "cd-6": 4}
    for setting in CountdownFamily().settings():
        for prompt in CountdownFamily().generate(setting, per_setting[setting], seed=0):
            numbers, target = parse_countdown_question(prompt.question)
            assert _reachable([float(n) for n in numbers], target), (
                f"{setting}: unsolvable instance {numbers} -> {target}"
            )


def test_the_question_carries_everything_the_grader_needs() -> None:
    """Ground truth is recoverable from the rendered prompt, as with the arithmetic family."""
    for prompt in CountdownFamily().generate("cd-4", 6, seed=1):
        numbers, target = parse_countdown_question(prompt.question)
        assert len(numbers) >= 3
        assert str(target) == prompt.answer


def test_an_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        CountdownFamily().generate("cd-not-a-setting", 1, seed=0)


# --------------------------------------------------------------------------------------
# Grading — correctness
# --------------------------------------------------------------------------------------

QUESTION = None  # filled per-test from a generated prompt


def _q(numbers: list[int], target: int) -> str:
    """A question string in the family's own format, so the grader is tested on real input."""
    from assay.crawl.tasks import render_countdown_question

    return render_countdown_question(numbers, target)


def test_a_correct_expression_scores_one() -> None:
    g = grade_countdown("so (3 + 4) * 5 = 35", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.CORRECT and g.reward == 1.0


def test_a_subset_of_the_numbers_is_allowed() -> None:
    """Countdown does not require using every number — only that none is used more than once."""
    g = grade_countdown("3 * 5 = 15", _q([3, 4, 5, 2], 15))
    assert g.outcome is Outcome.CORRECT


def test_reusing_a_number_is_wrong_not_correct() -> None:
    """5 * 5 = 25 hits the target but is illegal: only one 5 was given."""
    g = grade_countdown("5 * 5 = 25", _q([3, 4, 5, 2], 25))
    assert g.outcome is Outcome.WRONG_ANSWER, "a reused number must not score"


def test_a_repeated_number_may_be_used_twice_if_given_twice() -> None:
    g = grade_countdown("5 * 5 = 25", _q([5, 5, 3, 2], 25))
    assert g.outcome is Outcome.CORRECT


def test_using_a_number_that_was_not_given_is_wrong() -> None:
    g = grade_countdown("9 + 1 = 10", _q([3, 4, 5, 2], 10))
    assert g.outcome is Outcome.WRONG_ANSWER


def test_a_valid_expression_that_misses_the_target_is_wrong_not_parse_fail() -> None:
    """The distinction Phase 0.1 made structural: 'did the wrong thing' vs 'said nothing'."""
    g = grade_countdown("3 + 4 = 7", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.WRONG_ANSWER and g.reward == 0.0


def test_no_expression_at_all_is_parse_fail() -> None:
    g = grade_countdown("I am not sure how to do this.", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.PARSE_FAIL and g.reward == 0.0


# --------------------------------------------------------------------------------------
# Grading — permissiveness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "(3 + 4) * 5",
        "(3 + 4) x 5",
        "(3 + 4) × 5",
        "(3 + 4)*5 = 35",
        "Therefore the answer is (3 + 4) * 5.",
        "step 1: 3+4=7\nstep 2: 7*5=35\nso (3 + 4) * 5",
    ],
)
def test_the_parser_accepts_the_shapes_models_actually_emit(text: str) -> None:
    """Permissive by design: a strict grader filters HARD problems, not badly-formatted ones."""
    assert grade_countdown(text, _q([3, 4, 5, 2], 35)).outcome is Outcome.CORRECT


def test_division_is_supported() -> None:
    g = grade_countdown("100 / 4 = 25", _q([100, 4, 3, 2], 25))
    assert g.outcome is Outcome.CORRECT


def test_thousands_separators_do_not_break_it() -> None:
    g = grade_countdown("500 * 3 = 1,500", _q([500, 3, 2, 7], 1500))
    assert g.outcome is Outcome.CORRECT


def test_the_last_expression_wins() -> None:
    """Small models restate; the last expression is what they commit to — as with R_binary."""
    g = grade_countdown("maybe 3 + 4, no wait: (3 + 4) * 5", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.CORRECT


def test_an_expression_the_parser_cannot_evaluate_is_parse_fail_not_wrong() -> None:
    """Division by zero is not a wrong answer — nothing was successfully extracted."""
    g = grade_countdown("3 / 0", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.PARSE_FAIL


def test_the_grader_never_executes_arbitrary_code() -> None:
    """The parser evaluates arithmetic under an allowlist, never `eval`."""
    g = grade_countdown("__import__('os').system('echo pwned')", _q([3, 4, 5, 2], 35))
    assert g.outcome is Outcome.PARSE_FAIL


# --------------------------------------------------------------------------------------
# The screen, end to end
# --------------------------------------------------------------------------------------


class _CountdownSampler:
    """Emits real Countdown solutions at a controlled rate, so the screen's arithmetic is testable.

    ``FakeSampler`` cannot serve here: it hardcodes ``<answer>N</answer>``, which the Countdown
    grader correctly rejects, so a sweep run through it reports 100% parse-fail and validates
    nothing. That failure is silent, which is why it is worth a test rather than a comment.
    """

    def __init__(self, p_correct: float, seed: int = 0) -> None:
        self.p_correct, self.seed = p_correct, seed

    def sample(self, prompts, *, k, cfg):  # type: ignore[no-untyped-def]
        import random

        from assay.crawl.sampling import Completion

        out = []
        for prompt in prompts:
            numbers, target = parse_countdown_question(prompt.question)
            solution = _solve(numbers, target)
            rng = random.Random(f"{self.seed}:{prompt.prompt_id}")
            row = []
            for _ in range(k):
                if solution is not None and rng.random() < self.p_correct:
                    text = f"working... so {solution}"
                else:
                    # A legal expression that misses: WRONG_ANSWER, not PARSE_FAIL.
                    text = f"working... so {numbers[0]} + {numbers[1]}"
                row.append(Completion(text=text, n_tokens=len(text.split())))
            out.append(row)
        return out


def test_the_screen_measures_what_it_claims_to() -> None:
    """End-to-end: family -> sampler -> countdown grader -> summary, with the right grader wired in.

    Phase 0.3's entire first stage is this pipeline producing a `dead_group_fraction`. If
    `sweep_setting` graded Countdown with `grade_binary` — which it did before the grader seam — the
    screen would report a fabricated number and nothing downstream would notice.
    """
    from assay.crawl import calibrate
    from assay.crawl.sampling import SamplerConfig

    cfg = SamplerConfig(temperature=1.0, top_p=1.0, max_new_tokens=64, seed=0)
    summary = calibrate.sweep_setting(
        CountdownFamily(), "cd-3",
        sampler=_CountdownSampler(p_correct=0.5), n_prompts=16, k=8, cfg=cfg, seed=0,
    )
    assert summary.parse_fail_rate == 0.0, "legal expressions must never read as parse failures"
    assert 0.3 < summary.pass_at_1 < 0.7, f"pass rate should track the double: {summary.pass_at_1}"
    assert sum(summary.histogram) == 16


def test_a_starved_task_shows_up_as_dead_groups() -> None:
    """The screen's actual decision variable, at the rate Countdown is reported to sit at.

    At p = 0.02 the dead-group fraction is p^8 + (1-p)^8 = 0.851 — the number that makes R0
    ill-posed at 1.5B. This asserts the pipeline would actually surface it.
    """
    from assay.crawl import calibrate
    from assay.crawl.sampling import SamplerConfig

    cfg = SamplerConfig(temperature=1.0, top_p=1.0, max_new_tokens=64, seed=0)
    summary = calibrate.sweep_setting(
        CountdownFamily(), "cd-3",
        sampler=_CountdownSampler(p_correct=0.02), n_prompts=64, k=8, cfg=cfg, seed=1,
    )
    assert summary.dead_group_fraction > 0.75, (
        f"a near-zero pass rate must read as starved, got {summary.dead_group_fraction}"
    )
