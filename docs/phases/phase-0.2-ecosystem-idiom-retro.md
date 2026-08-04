# Phase 0.2 retro — The ecosystem idiom

> Written 2026-08-03 at phase completion, before merge. Sections anchor to their eventual paper
> section (`docs/process.md`). Plan: `phase-0.2-ecosystem-idiom-plan.md`. Predecessor:
> `phase-0.1-grpo-by-hand-retro.md`.

## 1. Hypothesis and purpose → **Methods**

Phase 0.1 proved gradients flow with a hand-rolled loop. Every phase from Walk onward runs on the
ecosystem stack, and the project's central artifact — a **grader factorial over one task set** — must
be expressed in that stack's idiom. 0.2 performs the translation on the one task we understand
completely, and uses it to ask a question no later phase can ask as cheaply: **does an independent
trainer learn our task from our published environment?**

## 2. What happened → **Results**

**All five gates met, at $0.** Three hosted runs on Prime's free tier.

| gate | result |
|---|---|
| **G1** local | 287 tests green, ruff + mypy clean |
| **G2** loads under `verifiers` | `SingleTurnEnv`, rubric wired, dataset materialises |
| **G3** published | `gkartik/assay-add3digit` v0.1.1, PUBLIC, CI SUCCESS |
| **G4** trained | **eval 0.9980** vs a gate of ≥ 0.85 |
| **G5** filter | `zero_advantage` confirmed **on by default** |

### G4 in detail

Unfiltered eval on **held-out** prompts (training seed 0, eval seed 999; 1 shared prompt in 512
against 1.26 expected by chance from 810,000 pairs — verified, not assumed):

| step | eval | comparator |
|---|---|---|
| 1 | **0.5879** | Phase 0.1 run 7, first 10 steps: **0.571 ± 0.019** — inside the band |
| 25 | 0.7246 | |
| 200 | **0.9980** | 511/512. Run 7, last 10 steps: 0.923 ± 0.018 |

The step-1 agreement is the real evidence for the port: an independent trainer, on an independently
published environment, starting inside our measured band.

## 3. Surprises — first-class → **Discussion**

### 3.1 The published environment cannot import the research repo

A published env runs on Prime's infrastructure and may import only what its own `pyproject.toml`
declares. This repo is private, so the plan's *"reuse, do not reimplement"* was impossible. The
generator and graders had to be **vendored**, which makes drift the central risk: G4 compares against
Phase 0.1's measurements, and the comparison is void the moment the copies diverge.
`tests/test_env_add3digit.py` is the guard — every grader asserted against its `crawl/rewards.py`
counterpart on the step-150 fixtures from ablation B, plus `grader_fingerprint()` equality.

### 3.2 A dependency conflict forced a better design

`verifiers` needs `numpy>=2.1`; the dev extra pins `numpy<2` because `torch<2.3` requires it on this
2019 Intel Mac. **They cannot coexist**, so anything importing `verifiers` is locally untestable.

That forced the variant→weights mapping out into `rubric_spec()`, a pure function with no `verifiers`
dependency. The result is better than the original: the phase's central design claim — *a grader
variant is a weight vector over a fixed function list* — became assertable in `make check` instead of
skipped. The first draft asserted on `built.rubric.funcs`, which would have been **wrong** (see 3.3)
and would have skipped silently rather than failing.

### 3.3 Two "bugs" that were my inspection, not the code

`env.rubric.funcs` reads empty and `env.dataset` reads `None` — both correct behaviour.
`MultiTurnEnv.__init__` wraps any rubric in a `RubricGroup` (adding a turn monitor), so `.funcs` on
the *group* is empty by design; and `dataset` stays `None` until `get_dataset()` materialises the
builder. Reading the source settled both in minutes.

### 3.4 `prime-rl`'s headline metric is computed over a filtered subset

**The first G4 run was unscoreable.** Its reported `Reward` is the mean over rollouts surviving the
`zero_advantage` filter, not over the batch — verified rather than inferred: `reward × trainable` is
an integer on 24/24 captured steps while `reward × 128` is not (step 196: `0.7917 × 24 = 19.00` vs
`× 128 = 101.34`).

Because the filter removes **unanimous** groups, and a unanimous group late in training is
overwhelmingly an all-*correct* one, the metric **excludes the policy's successes, with a bias that
grows as the policy improves**. At step 200 only 8/128 survived, so `Reward 0.8750` was 7 correct out
of the single group still containing an error.

This is the project's own thesis appearing in a production RL stack: the headline number diverges
from the true quantity, and the divergence grows precisely as the system succeeds. `prime-rl` is not
wrong to filter — dropping zero-gradient rollouts is sensible — but the reported reward inherits the
filter and nothing says so.

### 3.5 The stack does not merely drop dead groups — it refills the batch

Late in the reference run the denominator stops being `batch_size`: `24 live of 384 generated`. Our
loop took the 128 it got and let ~70% contribute nothing; this one pays up to 3× the sampling cost to
keep every gradient step at full strength.

### 3.6 And that is *not* why it scored higher — prediction falsified

The obvious hypothesis: batch refill explains 0.998 vs 0.923. A one-field A/B (`enforce = false`)
falsified it. The unfiltered arm reached **1.0000**, converging *faster* if anything.

The lever only partially took — oversampling dropped sharply but did not stop — which cuts *against*
the hypothesis rather than rescuing it: batches then held ~104 dead rollouts of 128, our loop's exact
situation, and performance did not degrade.

**Worth naming what this kills.** "Dead groups are a pathology, and here is the production mitigation
that fixes it" was a tidy arc already half-written. The data does not support it. Dead groups still
cost *compute*, exactly as Phase 0.1 measured; what is unsupported is that reclaiming them explains
the performance difference.

### 3.7 The free tier has three stacked, undocumented gates

Environment **PUBLIC** · CI action **SUCCESS** · README containing the phrase **"reward hacking
sprint"** with stated hypotheses. Only the first returns a specific error; the other two share one
opaque *"does not meet the free-tier environment requirements"*, so each had to be found by inference
and retried. The sprint's *review* window closed ~2026-06-20, but its tag requirement is still live in
the gate.

## 4. What to change → **Methods (later phases)**

1. **Stage 2 design pin: override `prime-rl`'s pre-batch filter list**, or ablation-D-style
   measurements are invisible on that stack. Belongs in `docs/pre-registration.md` before any grid.
2. **Verify the denominator of any metric taken from someone else's stack** before comparing it to
   ours. The integrality check (`metric × candidate_n` is an integer) is cheap and decisive.
3. **Vendored code needs a fingerprint guard**, not a code review. `grader_fingerprint()` did this
   across a package boundary exactly as it had done across time.

## 5. Limitations → **Limitations**

- **Both prime runs are n=1.** `CLAUDE.md` §10.3 — added at the end of Phase 0.1 — forbids a
  directional claim from one seed. So **"prime-rl trains better than our hand-rolled loop" is not a
  claim this phase is entitled to make.** G4 is a *threshold* (0.998 ≥ 0.85), which one seed can
  settle; the ~0.07 comparison against 0.923 ± 0.018 is a *direction*, and it cannot.
- **The `enforce = false` lever only partially took.** The filter was weakened, not removed.
- **The ~0.07 gap is unexplained**, with one candidate eliminated. The most conspicuous remaining
  difference is that the prime configs carry **no KL term** while run 7 used β = 0.04 — and ablation B
  found that leash made things *worse*. Untested; optimizer, schedule, warmup, LoRA rank and
  on-policy discipline all differ too.
- **The eval uses our own graders**, so it is self-consistent rather than independent. It validates
  the port, not the graders.
- **G2 was adapted, not met as written.** A pass-rate eval needs paid inference against a $0 wallet;
  the check became "loads and scores under real `verifiers`", with the pass-rate comparison folded
  into G4's step-1 reading. That is arguably stronger — held-out, 512 rollouts — but it is a
  deviation and is recorded as one.

## 6. Related-work touchpoints → **Related Work** (UNVERIFIED until Phase 0.5)

- **`verifiers` v0 vs v1.** v1 decomposes the v0 Environment into TaskSet / Task / Harness, and its
  ten shipped harnesses (`bash`, `claude_code`, `codex`, `mini_swe_agent`, `terminus_2`, …) show the
  purpose: one task set under many agent scaffolds. Degenerate for a single-turn task; **directly
  relevant at Phase 1.1**, since two of the grid's four axes (timeout, sandbox writability) are
  runtime/harness concerns.
- **`prime-rl`'s rollout filters** (`zero_advantage`, `gibberish`, `repetition`) are direct evidence
  that dead groups and degenerate generations are recognised production problems. What this project
  adds is measuring their size per grader configuration.

## 7. Gate status

| gate | status |
|---|---|
| Tests pass | ✅ 287, `make check` green |
| Lint / typecheck | ✅ ruff (now including `environments/`) + mypy --strict |
| Reproducibility | ✅ env published and versioned; run IDs and trajectories committed |
| Retro with paper anchors | ✅ this document |
| `/learn` with `[DELETE]` | ✅ `phase-0.2-ecosystem-idiom-learn.md` |
| Three-reviewer pass | n/a — 0.2 is not a stage boundary (Crawl→Walk is after 0.5) |

**Spend: $0.** Three hosted runs on the free tier.
