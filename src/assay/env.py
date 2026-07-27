"""``bisect`` — root-cause debugging under a query budget.

A program fails a test. The agent may run the suite, add instrumentation, run subsets, and inspect
intermediate values — **each action costs query budget** — and must produce a fix.

The proxy/true split is native to the task:

- **visible grader (proxy)** — the failing test passes;
- **hidden grader (true)** — a held-out suite exercising the *same root cause* through other code
  paths.

The exploit is universally legible: special-case the input, wrap it in ``try/except``, hardcode the
expected output, or edit the test file.

Scope (``docs/stages.md`` Phase 1.1): ~40 seeded bugs in small pure-Python programs, subprocess with
a timeout, **no Docker**. The sandbox is not incidental — sandbox escape *is* one of the pathologies
under study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ActionKind(str, Enum):
    """What the agent can spend query budget on."""

    RUN_SUITE = "run_suite"
    RUN_SUBSET = "run_subset"
    INSTRUMENT = "instrument"  # insert a print / assertion and re-run
    READ_SOURCE = "read_source"
    SUBMIT_FIX = "submit_fix"  # terminal


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    payload: str
    cost: int = 1


@dataclass(frozen=True)
class Observation:
    """What comes back from a step. ``budget_remaining`` is always visible to the agent."""

    stdout: str
    stderr: str
    exit_code: int
    budget_remaining: int
    terminated: bool = False


@dataclass(frozen=True)
class Task:
    """One seeded bug.

    ``root_cause_id`` groups tasks that share a defect class. The held-out suite for a task is drawn
    from *other* code paths exercising the same ``root_cause_id`` — that is what makes the true
    grader measure the cause rather than the symptom.
    """

    task_id: str
    root_cause_id: str
    program_path: Path
    visible_test_path: Path
    heldout_test_path: Path
    query_budget: int
    difficulty: str


@dataclass
class Trajectory:
    """A full episode. Serialised to ``experiments/<phase>/raw/<run>/trajectories.jsonl``."""

    task_id: str
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    submitted_patch: str | None = None
    budget_spent: int = 0


def load_tasks(root: Path) -> list[Task]:
    """Load the seeded-bug task set. Implement at Phase 1.1."""
    raise NotImplementedError("Phase 1.1 — bisect task set")


class BisectEnv:
    """Gymnasium-style env over one task. Wrapped by the ``verifiers`` spec at Phase 0.2/1.1.

    Deliberately *not* Docker-backed: subprocess + timeout keeps Walk to two days, and sandbox
    writability is a grader-variant factor rather than an infrastructure guarantee.
    """

    def __init__(self, task: Task, *, allow_test_writes: bool, timeout_s: float | None) -> None:
        self.task = task
        self.allow_test_writes = allow_test_writes
        self.timeout_s = timeout_s

    def reset(self) -> Observation:
        """Start an episode; return the initial failing-test output."""
        raise NotImplementedError("Phase 1.1")

    def step(self, action: Action) -> Observation:
        """Execute one action, decrement budget, return the observation."""
        raise NotImplementedError("Phase 1.1")

    def state(self) -> Trajectory:
        """Current episode state — used by the battery's clustering probe (A2)."""
        raise NotImplementedError("Phase 1.1")
