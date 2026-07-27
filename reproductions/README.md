# Reproductions

**Principle: reproduce exactly what the project load-bears on, and nothing else. Each reproduction
retires one named assumption** (desideratum 14). A reproduction that does not load-bear on `assay`
does not get run.

## Ledger

| # | Target | Assumption it retires | Cost | Stage | Status |
|---|---|---|---|---|---|
| **R0** | TinyZero / Countdown GRPO at **1.5B** | *"my GRPO loop actually learns"* | ~$10 | 0.3 · Crawl | ☐ |
| **R1** | Prime Intellect 1B reward hacking (Llama-3.2-1B-Instruct, 100 steps, batch 128, lr 1e-4, ~$0.64) | *"small models hack inside my step budget"* — **also the reachability gate** (`../docs/pre-registration.md` §4 L2) | ~$2 | 0.4 · Crawl | ☐ |
| **R2** | *Before the Model Learns the Bug* FP rates (math 83.2% / JSON 86.9% / code 55.7%; exploit in 2–4 queries, 94–100% of trials) | *"adversarial probing finds grader bugs"* = battery axes A1 + A5 | ~$10, **no GPU** | 1.5 · Walk | ☐ |
| **R3** | Reasoning Gym transfer (+9.7% MATH, +7.7% BBH, Qwen2.5-3B-Instruct) | *"my eval harness measures transfer correctly"* = the whole η leg | ~$12 | 3.2 · Gallop | ☐ |
| **R4** | *Rollout Pass-Rate Control* core claim — balanced groups carry more gradient signal | *"A3 is a real axis, not folklore"* | **$0** — analysis of runs already in hand | 2.4 · Run | ☐ |

**R1 does double duty** as reproduction *and* reachability gate. That is deliberate: it moves the
project's biggest risk into Crawl, where finding out costs a day instead of a stage.

## Scale caveats — read before running

- **TinyZero: Qwen2.5-0.5B *fails* to learn reasoning on Countdown. 1.5B learns search and
  self-verification. 3B for the headline, <$30 on veRL.** Budget R0 at **1.5B**; do not attempt 0.5B.
  Corollary: 0.6B is fine for Phase 0.1's *plumbing* smoke tests on a task where the base policy
  already has nonzero pass rate, but not for reasoning emergence.
- **R3's published config is 3B.** Run at 1.7B and **report the scale delta honestly rather than
  claiming a match.**
- **Scope each reproduction to one cell, never the whole paper.** R3 = one training config, two
  benchmarks.

## Reporting format

One file per target: `RN-<slug>.md`, with these sections:

```markdown
# RN — <target>

**Assumption retired:** <the one thing this makes safe to rely on>
**Original claim:** <verbatim, with source + id>
**Original number:** <value, with the config it was measured under>
**My number:** <value>
**Delta:** <absolute + relative; and whether it is inside the original's own variance>
**What I changed:** <model, scale, steps, seeds, anything that differs>
**Verdict:** REPRODUCED / PARTIAL / FAILED TO REPRODUCE
**Consequence for assay:** <what this means for the design; what changes if anything>
```

**Verdicts include FAILED TO REPRODUCE.** Given the `../../originality` precedent — a headline result
withdrawn on a matched null — a cleanly-reported failure is a feature of the portfolio, not a
blemish.

## Layout

- `RN-<slug>.md` — the report. **Committed.**
- `RN-<slug>/results/*.json` — derived metrics the report's numbers regenerate from. **Committed.**
- `RN-<slug>/raw/` — raw rollouts and logs. **Ignored** (see `../.gitignore`).

## The seed-variance contribution (Phase 2.5)

RL reproducibility is genuinely poor: performance varies wildly with seeds, environment stochasticity
and evaluation protocol, and 2026 work quantifies how seed-induced underspecification *inflates the
evidence threshold* needed to claim one method beats another.

So: **≥3 seeds on every headline arm, with the seed-variance band reported beside every effect size**
(desideratum 9) — and one section in the writeup answering:

> *"How many seeds does it take to detect the effect sizes the RL-environments field routinely
> claims?"*

This costs almost nothing, since the seeds are run regardless. It is `../../originality`'s
matched-null move applied to a field that has not had its reckoning.
