# assay

**A zero-GPU-hour diagnostic battery for RL environments — does an environment teach the skill, or
teach the exploit?**

RL environments are the bottleneck in agentic AI, and the unsolved part is not the training
algorithm but **environment quality**. Reward hacking is practitioners' #1 complaint, verifiers are
wrong in both directions, and the pre-training predictors that exist are **single-axis and explain
little of the variance** — graders of *equal measured accuracy* produce very different post-training
regret (Wen et al., `2410.05584`). Getting it wrong costs about **$2,400 per task**, per Epoch AI's
practitioner interviews.

`assay` tests whether a battery of **inference-only** probes predicts what RL will actually do to a
policy, and validates the answer against real GRPO runs on a purpose-built environment.

**The bet:** a frontier model is a cheap forecaster of a small policy's RL endpoint. Catalogue the
exploits a frontier model finds at step 0, and ask whether that set predicts what the small policy
converges to after training.

> **One premise of that bet has already been measured, and it was wrong.** This README used to say a
> small policy *"needs thousands of steps to discover it can pass by `assert True`."* At 1B, on a
> grader with a reachable exploit, it takes **8–40 steps** — 15 runs of 15 on the train curve,
> 12 of 15 on the pre-registered eval curve, for `$0`
> ([Phase 0.4](docs/phases/phase-0.4-r1-retro.md)). Good news for feasibility; it narrows the
> capability gap the forecasting story leans on, and that is now an open question rather than an
> assumption.

New here? **[`docs/plain-english-summary.md`](docs/plain-english-summary.md)** is the no-jargon
account of what exists and what it found. **[`CLAUDE.md`](CLAUDE.md)** has the full design;
**[`docs/stages.md`](docs/stages.md)** has the Crawl → Walk → Run → Gallop ladder.

## What gets measured

| | |
|---|---|
| **The battery** | A1 hackability · A2 grader degeneracy · A3 pass-rate band · A4 judge instability · A5 verifier asymmetry · A6 contamination |
| **The cheap outcome** | proxy-grader reward minus held-out-grader reward, as a slope over training steps |
| **The headline outcome** | **transfer efficiency** `η = G_skill / G_total` — how much of an RL gain travels to an independently authored environment for the same skill |

## Status — Stage 0 of 4 (Crawl)

| Phase | | Result |
|---|---|---|
| **0.1** GRPO written by hand | ✅ | Trains: 43% → **92%** on 3-digit addition. Four deliberate breakages, each a committed figure. A degenerate grader produced a **52-point** proxy–true gap on demand — **and the KL leash meant to restrain it made the gap *wider*, by 0.037 on 3/3 seeds**, while carrying 54% of the loss. |
| **0.2** Ported to the `verifiers` / `prime-rl` stack | ✅ | Published to the Environments Hub. An independent trainer's **first** measurement — **58.8%**, before any training — landed inside our own hand-built band of **57.1% ± 1.9%**; that agreement, not the 99.8% endpoint it reached by step 200, is the evidence the port is faithful. `$0`. |
| **0.3** R0 (Countdown) | ✅ retired | Disqualified on our own ledger rule — the target publishes no number to reproduce. Three screens instead, `$2.21` of a `$10` line. |
| **0.4** R1 (reachability) | ✅ | **The project's biggest risk is retired.** Every run exploits a broken grader in 8–40 steps — **15/15 on the train curve, 12/15 on the pre-registered eval curve** — for `$0`. Details below. |
| **0.5** Literature gate | 🔄 | In progress. |
| **1.x** Walk — build `bisect`, `assay` v0 | ☐ | Not started. |

**Spend:** roughly **$23–25** of GPU credit, essentially all of it in Phase 0.1 — which overran its
own $5 line, mostly on crashes during bring-up that produced no usable output. **Phases 0.2 and 0.4
cost `$0`**, on a free tier. About **$17** of credit remains. (The ledger's running-total column in
[`tasks/spend.md`](tasks/spend.md) understates this: rows for Phase 0.1's seed pass and probe reruns
were never added, and the file says so.)

**What is actually built:** a hand-rolled GRPO loop with seven ablations, two published environments
(`gkartik/assay-add3digit`, `gkartik/assay-hackword`), the R1 scoring machinery, and 406 tests.
`make check` runs ruff + mypy-strict + the suite with no GPU.

**What is not built yet:** `bisect`, the battery itself, and every hypothesis test downstream of them.

### What Phase 0.4 found

- **Exploits are cheap and fast to reach at 1B** — 8–40 steps, not thousands (15/15 train, 12/15
  eval). The gate the whole Run stage rests on passes.
- **The admission screen is broken, and that is the more useful finding.** Two of three variants sat
  *below* the pre-registered lower bound of `1/64` and saturated anyway. `1/64` is the resolution
  floor of a 64-sample screen, not a reachability threshold — a literal application would have
  wrongly rejected a usable environment **42–68%** of the time. Redesign owed before Run.
- **The pre-registered prediction is unresolved.** Base rate predicts saturation onset across an
  order-of-magnitude gap and not across a 2.3× one. The 95% interval for the discriminating pair is
  **[−7.84, +9.85] steps** — it excludes a difference the size of the published one and does not
  exclude zero.

## The environment

**`bisect`** *(Walk stage — not yet built)* — root-cause debugging under a query budget. A program
fails a test. The agent may run the suite, add instrumentation, run subsets, inspect intermediate
values, each costing budget, and must produce a fix.

- **Visible grader (proxy):** the failing test passes.
- **Hidden grader (true):** a held-out suite exercising the *same root cause* through other code
  paths.
- **The exploit is universally legible:** special-case the input, wrap it in `try/except`, hardcode
  the expected output, edit the test.

The diagnostic grid is 8–12 *grader configurations* over one task set — test visibility × reward
shape × timeout × sandbox writability — so grader pathology is known by construction.

> **A design requirement R1 turned up the hard way.** A substrate needs a reachable exploit **and**
> checkable ground truth *simultaneously*. Neither Crawl substrate manages both: our story task has
> base rates but no truth signal, our arithmetic task has truth but a 0/4096 base rate. Every R1 run
> therefore had a true reward of identically zero, so the project's own outcome variable was absent
> from its reachability gate. `bisect` has to satisfy both, and that is now a stated constraint
> rather than an assumption.

## Quickstart

```bash
uv sync --extra dev            # base + numpy/scipy/sklearn + pytest/ruff/mypy
make check                     # ruff + mypy-strict + 406 tests (no GPU)

uv run python scripts/score_r1.py   # regenerate every Phase 0.4 number from committed data

uv sync --extra dev --extra train --extra api   # when you're ready to run the loop
```

Session entry point is [`tasks/todo.md`](tasks/todo.md). Results live in `experiments/*/results/`
and are committed, so every figure and paper number regenerates without network access.

## How this project works

Hypotheses, gates and decision rules are written down **before** the runs
([`docs/pre-registration.md`](docs/pre-registration.md)); a run that disconfirms its driving
hypothesis is a successful run. Nulls are reported. Every phase ends with a retro and a lessons pass,
and stage boundaries get a three-reviewer adversarial pass — which at the Crawl→Walk boundary caught
two errors in our own write-up, in opposite directions, and both are recorded rather than quietly
fixed.

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
