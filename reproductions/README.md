# Reproductions

**Principle: reproduce exactly what the project load-bears on, and nothing else. Each reproduction
retires one named assumption** (desideratum 14). A reproduction that does not load-bear on `assay`
does not get run.

## Ledger

| # | Target | Assumption it retires | Cost | Stage | Status |
|---|---|---|---|---|---|
| **R0** | TinyZero / Countdown GRPO at **1.5B** | *"my GRPO loop actually learns"* | ~$10 | 0.3 · Crawl | ☐ |
| **R1** | Prime Intellect 1B reward hacking (Llama-3.2-1B-Instruct, 100 steps, batch 128, lr 1e-4, ~$0.64) | *"small models hack inside my step budget"* — **also the reachability gate** (`../docs/pre-registration.md` §4 L2) | ~$2 | 0.4 · Crawl | ⚠️ **PARTIAL, 2026-08-06 · $0.00** |
| **R2** | *Before the Model Learns the Bug* FP rates (math 83.2% / JSON 86.9% / code 55.7%; exploit in 2–4 queries, 94–100% of trials) | *"adversarial probing finds grader bugs"* = battery axes A1 + A5 | ~$10, **no GPU** | 1.5 · Walk | ☐ |
| **R3** | Reasoning Gym transfer (+9.7% MATH, +7.7% BBH, Qwen2.5-3B-Instruct) | *"my eval harness measures transfer correctly"* = the whole η leg | ~$12 | 3.2 · Gallop | ☐ |
| **R4** | *Rollout Pass-Rate Control* core claim — balanced groups carry more gradient signal | *"A3 is a real axis, not folklore"* | **$0** — analysis of runs already in hand | 2.4 · Run | ☐ |

**R1 does double duty** as reproduction *and* reachability gate. That is deliberate: it moves the
project's biggest risk into Crawl, where finding out costs a day instead of a stage.

### R1 — the delta, 2026-08-06

**Original number** (`primeintellect.ai/blog/reward-hacking`, read first-hand 2026-08-04): steps to
50% saturation of a hack-word grader — `forgotten` **11**, `midnight` **18**, `ocean` **44**, at
Llama-3.2-1B-Instruct · 100 steps · batch 128 · lr 1e-4 · G=8 · T=1.0. Their substrate is creative
writing; the published baselines are 7.81% / 1.56% / 0.47%.

**Ours** (15 runs, 3 words × 6 seeds on the discriminating pair, `$0`; regenerate with
`uv run python scripts/score_r1.py`):

| word | theirs | ours (eval, pooled median) | delta | in ±50% band |
|---|---|---|---|---|
| `forgotten` | 11 | **9.10** (n=3, sd 2.46) | −1.9 | ✅ |
| `ocean` | 44 | **25.37** (n=6, sd 6.75) | −18.6 | ✅ |
| `midnight` | 18 | **30.76** (n=4, sd 1.78) | +12.8 | ❌ |

**Verdict: partial.** The *mechanism* reproduces and is the load-bearing part — a 1B policy converges
on a degenerate grader in 8–40 steps (**15/15 runs on the train curve, 12/15 on the pre-registered
eval curve**), so the reachability gate and L2's positive control both discharge. The **ordering does
not**: they have `midnight` < `ocean`, we cannot separate the two. U = 16/24, exact p = 0.24, and
the 95% interval for the shift is **[−7.84, +9.85] steps — which excludes their −26-step gap and
does not exclude zero.**

**Two caveats belong beside the delta.** First, our base rates are measured on our own prompt set and
**reverse theirs** on that pair (0.0135 vs 0.0059, z = 7.0), so this is not a like-for-like ordering
comparison — the substrate differs, and Phase 0.4 measured base rates moving by orders of magnitude
across substrates (0/4096 on arithmetic for the same words). Second, `midnight`'s miss is in the
direction the phase plan flagged as a finding rather than an error: our base rates are *higher* than
theirs, so our onsets should have been *earlier*, and one is later.

Full analysis: `../docs/phases/phase-0.4-r1-retro.md`. Data: `../experiments/phase-0.4-r1/results/`.

## Scale caveats — read before running

- **R0 IS RETIRED, 2026-08-03.** Not descoped for budget — **disqualified on the ledger's own admission rule.** This file requires *"Original number: value, with the config it was measured under"* and a **Delta**. TinyZero publishes a cost claim (*"< $30"*), a qualitative claim (*"Qwen2.5-0.5B ... fails to learn reasoning"*), and a W&B link — **no accuracy, no config, no metrics.** There is nothing to compute a delta against, and no amount of compute creates one. **That was knowable from a README read on day one**, and is now rule `[ADD] 4` in `docs/phases/phase-0.3-r0-learn.md`: a target's paper is read for a publishable number *before* it is admitted here. The three screens below were run before that was recognised; they are kept because they are a real result about the task.

- **MEASURED 2026-08-03, and it overturns the caveat this line used to carry.** The scaffold said
  *"1.5B learns search and self-verification; budget R0 at 1.5B"*. That came from an LLM-assisted
  research pass and is **wrong**. A base-rate screen (n=200 prompts, k=G=8, T=1.0, 512 tokens, base
  checkpoints, raw prompts) measured:

  | model | setting | pass@1 | dead groups | parse_fail | verdict |
  |---|---|---|---|---|---|
  | Qwen2.5-1.5B | cd-3 | 0.024 | **0.845** | 0.272 | starved |
  | Qwen2.5-1.5B | cd-4 | 0.006 | **0.950** | 0.244 | starved |
  | Qwen2.5-3B | cd-3 | 0.059 | **0.620** | 0.151 | marginal |
  | Qwen2.5-3B | cd-4 | 0.009 | **0.925** | 0.170 | starved |

  Against the pre-registered band (`<=0.50` workable), **nothing clears it at either scale.** The
  measured 1.5B pass rate of 0.024 matches the ~2% that secondary sources reported, so that figure
  is now first-hand.

  **Not a grader artifact:** `parse_fail` is 0.15-0.27 against a 0.5 rig-broken threshold, and
  `wrong_answer` runs 0.70-0.83. The models emit legal expressions that miss — reasoning and
  failing, not failing to format.

  **Consequence:** R0 is **not well-posed at either affordable scale**. A training failure would be
  attributable to reward sparsity in the task, not to the loop, so it could not retire *"my loop
  actually learns."* Screen cost ~$1.50 against R0's $10 line; artifacts in
  `experiments/phase-0.3-r0/results/`, method in `docs/phases/phase-0.3-r0-plan.md`.

  The prediction came free: `dead = p^8 + (1-p)^8` at `G=8` is Phase 0.1's own task-selection
  criterion, and `add-3digit` — the task it chose — sits at 0.012.
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
