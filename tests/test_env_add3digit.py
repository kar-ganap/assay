"""The published `verifiers` environment, checked against the graders Phase 0.1 measured.

**Why these tests carry more weight than they look like they should.** A published environment runs
on Prime's infrastructure and may import only what its own ``pyproject.toml`` declares. This repo is
private, so ``environments/assay-add3digit/`` cannot ``from assay.crawl import ...`` — the generator
and the graders have to be **vendored** into the env module.

That makes drift the central risk. Phase 0.2's gate compares an independent trainer's result against
Phase 0.1's measurements, and the comparison is meaningless the moment the vendored graders stop
agreeing with ``crawl/rewards.py``. These tests are the thing standing between "we ported the task"
and "we ported a *different* task that resembles it".

Everything here runs with zero GPU and zero network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from assay.crawl import rewards as crawl_rewards
from assay.crawl.tasks import ArithmeticFamily

ENV_DIR = Path(__file__).resolve().parents[1] / "environments" / "assay-add3digit"
ENV_MODULE = ENV_DIR / "assay_add3digit.py"


def _load_env_module():  # type: ignore[no-untyped-def]
    """Import the env module by path — it is a standalone package, not part of ``src/``."""
    spec = importlib.util.spec_from_file_location("assay_add3digit", ENV_MODULE)
    assert spec and spec.loader, f"cannot load {ENV_MODULE}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["assay_add3digit"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Drift: the vendored copy must BE the measured grader, not resemble it
# --------------------------------------------------------------------------------------

#: The Phase 0.1 fixtures. 875 + 812 = 1687; the wrong answers are what the policy actually emitted
#: at step 150 of ablation B's control arm.
FIXTURES = [
    ("<answer>1687</answer>", "1687"),
    ("<answer>1677</answer>", "1687"),
    ("<answer>0</answer>", "1687"),
    ("875 + 812 = 1687", "1687"),
    ("I think it is about 1687 total.", "1687"),
    ("<answer>hello</answer>", "1687"),
    ("no number here at all", "1687"),
    ("the answer is 1,687", "1687"),
]


def test_the_vendored_grader_fingerprint_matches_the_measured_one() -> None:
    """The single strongest drift guard: same extractors, same patterns, same tie-break weight.

    ``grader_fingerprint`` exists because ``R_binary`` changed from strict-tag to last-integer on
    2026-07-27 and nothing in the manifest said so. Here it guards the same failure across a
    *package* boundary rather than across time.
    """
    env = _load_env_module()
    assert env.grader_fingerprint() == crawl_rewards.grader_fingerprint()


@pytest.mark.parametrize("completion,expected", FIXTURES)
def test_vendored_r_binary_agrees_with_crawl(completion: str, expected: str) -> None:
    env = _load_env_module()
    assert env.r_binary(completion=completion, answer=expected) == pytest.approx(
        crawl_rewards.grade_binary(completion, expected).reward
    )


@pytest.mark.parametrize("completion,expected", FIXTURES)
def test_vendored_r_format_agrees_with_crawl(completion: str, expected: str) -> None:
    """Ablation B's degenerate grader. Must keep discarding the answer, exactly as measured."""
    env = _load_env_module()
    assert env.r_format(completion=completion, answer=expected) == pytest.approx(
        crawl_rewards.grade_format_only(completion, expected).reward
    )


def test_the_degenerate_grader_still_pays_for_a_wrong_answer() -> None:
    """The property that makes ablation B work, asserted rather than assumed."""
    env = _load_env_module()
    assert env.r_format(completion="<answer>0</answer>", answer="1687") == 1.0
    assert env.r_binary(completion="<answer>0</answer>", answer="1687") == 0.0


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


def test_dataset_rows_carry_the_columns_verifiers_expects() -> None:
    """``question`` / ``answer`` / ``info`` — the shape the reference env produces."""
    env = _load_env_module()
    rows = env.build_dataset(setting="add-3digit", n=8, seed=0)
    assert len(rows) == 8
    for row in rows:
        assert set(row) >= {"question", "answer", "info"}
        assert isinstance(row["question"], str) and row["question"]
        assert row["answer"].lstrip("-").isdigit()


def test_dataset_is_deterministic_from_seed_and_matches_the_pinned_generator() -> None:
    """The env must generate the SAME prompts Phase 0.1 trained on, or the gate compares two tasks."""
    env = _load_env_module()
    rows = env.build_dataset(setting="add-3digit", n=16, seed=0)
    pinned = ArithmeticFamily().generate("add-3digit", 16, seed=0)

    assert [r["question"] for r in rows] == [p.question for p in pinned]
    assert [r["answer"] for r in rows] == [p.answer for p in pinned]
    assert rows == env.build_dataset(setting="add-3digit", n=16, seed=0), "not deterministic"


# --------------------------------------------------------------------------------------
# The Rubric — the mechanism Walk's grader factorial will be built on
# --------------------------------------------------------------------------------------


def test_a_grader_variant_is_a_weight_vector() -> None:
    """**This is the phase's design payoff**, and it needs no `verifiers` import to assert.

    `docs/conceptual.md`'s grid is "grader configurations over one task set". In the verifiers idiom
    that is a *weight vector over a fixed function list* — the factorial becomes data rather than
    code, which is what lets Stage 2 sweep it as config instead of writing twelve environments.
    """
    env = _load_env_module()
    assert env.rubric_spec("binary") == [("r_binary", 1.0), ("r_format", 0.0), ("r_tiebreak", 0.0)]
    assert env.rubric_spec("format") == [("r_binary", 0.0), ("r_format", 1.0), ("r_tiebreak", 0.0)]
    assert env.rubric_spec("tiebreak") == [("r_binary", 0.0), ("r_format", 0.0), ("r_tiebreak", 1.0)]


def test_the_true_reward_is_measured_under_every_variant() -> None:
    """The proxy/true split, expressed in the rubric rather than in a remembered convention.

    `r_binary` appears in every variant's spec. Weight 0.0 is exactly "measured, never optimised" —
    so the held-out grader can never quietly become a second proxy.
    """
    env = _load_env_module()
    for variant in env.VARIANTS:
        names = [name for name, _ in env.rubric_spec(variant)]
        assert "r_binary" in names, f"{variant}: true reward is not being measured"


def test_exactly_one_grader_is_ever_the_training_signal() -> None:
    """A weight vector that sums to anything but 1.0 would silently blend two graders."""
    env = _load_env_module()
    for variant in env.VARIANTS:
        weights = [w for _, w in env.rubric_spec(variant)]
        assert sum(weights) == pytest.approx(1.0)
        assert sorted(weights) == [0.0, 0.0, 1.0]


def test_an_unknown_variant_is_rejected_rather_than_silently_defaulted() -> None:
    env = _load_env_module()
    with pytest.raises(ValueError, match="unknown reward variant"):
        env.rubric_spec("not-a-variant")


def test_load_environment_validates_before_importing_verifiers() -> None:
    """A bad variant must fail fast, not after pulling in the heavy stack.

    `verifiers` cannot be installed on this project's dev machine (it needs numpy>=2.1; torch<2.3
    needs numpy<2), so a `load_environment` that imported first would make this untestable.
    """
    env = _load_env_module()
    with pytest.raises(ValueError, match="unknown reward variant"):
        env.load_environment(reward="not-a-variant", n_train=4)
