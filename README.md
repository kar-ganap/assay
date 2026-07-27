# assay

**A zero-GPU-hour diagnostic battery for RL environments — does an environment teach the skill, or
teach the exploit?**

RL environments are the bottleneck in agentic AI, and the unsolved part is not the training
algorithm but **environment quality**. Reward hacking is practitioners' #1 complaint, verifiers are
wrong in both directions, and nobody can tell you an environment is good before spending the compute
— about **$2,400 per task**, per Epoch AI's practitioner interviews.

`assay` tests whether a battery of **inference-only** probes predicts what RL will actually do to a
policy, and validates the answer against real GRPO runs on a purpose-built environment.

**The bet:** a frontier model is a cheap forecaster of a small policy's RL endpoint. A 1.7B policy
needs thousands of steps to discover it can pass by `assert True`; Claude finds it in eight samples.
So catalogue the exploits a frontier model finds at step 0, and ask whether that set predicts what
the small policy converges to after training.

See **[`CLAUDE.md`](CLAUDE.md)** for the full design and **[`docs/stages.md`](docs/stages.md)** for
the Crawl → Walk → Run → Gallop ladder.

## What gets measured

| | |
|---|---|
| **The battery** | A1 hackability · A2 grader degeneracy · A3 pass-rate band · A4 judge instability · A5 verifier asymmetry · A6 contamination |
| **The cheap outcome** | proxy-grader reward minus held-out-grader reward, as a slope over training steps |
| **The headline outcome** | **transfer efficiency** `η = G_skill / G_total` — how much of an RL gain travels to an independently authored environment for the same skill |

## The environment

**`bisect`** — root-cause debugging under a query budget. A program fails a test. The agent may run
the suite, add instrumentation, run subsets, inspect intermediate values, each costing budget, and
must produce a fix.

- **Visible grader (proxy):** the failing test passes.
- **Hidden grader (true):** a held-out suite exercising the *same root cause* through other code
  paths.
- **The exploit is universally legible:** special-case the input, wrap it in `try/except`, hardcode
  the expected output, edit the test.

The diagnostic grid is 8–12 *grader configurations* over one task set — test visibility × reward
shape × timeout × sandbox writability — so grader pathology is known by construction.

## Quickstart

```bash
uv sync --extra dev            # base + numpy/scipy/sklearn + pytest/ruff/mypy
make check                     # ruff + mypy-strict + wiring smoke tests (no GPU)

uv sync --extra dev --extra train --extra api   # when you're ready to run the loop
```

## Status

**Scaffold.** `src/` is typed stubs — no compute path implemented yet.

**Next:** Phase 0.1 (GRPO by hand). Nothing paid runs until
[`docs/pre-registration.md`](docs/pre-registration.md) locks and the
[literature gate](literature-review/README.md) clears.

Session entry point is [`tasks/todo.md`](tasks/todo.md).

## Built on (borrowed, not claimed)

[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) + `prime-rl` (environment spec and
trainer) · [`trl`](https://github.com/huggingface/trl) (GRPO fallback) ·
[Reasoning Gym](https://github.com/open-thought/reasoning-gym) (transfer-efficiency substrate) ·
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) (deployment interface) · existing Environments Hub
environments (independently-authored pairs).

What is ours: the battery, the grader-variant factorial, `bisect`, the η decomposition, and the
validation against training outcomes.

## License

MIT — see [`LICENSE`](LICENSE).
