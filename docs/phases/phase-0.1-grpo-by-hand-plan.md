# Phase 0.1 — GRPO by hand

**Stage:** 0 (Crawl) · **Branch:** `phase-0.1-grpo-by-hand` · **Status:** IN PROGRESS (branch cut 2026-07-27)
**Est:** 8–10 h · **Est spend:** ~$5

---

## Purpose

Build the mechanical intuition the rest of the project rests on: **implement policy-gradient RL on a
language model from scratch, then break each component deliberately and watch it fail.**

This is the phase that closes the portfolio's gradient gap (`../conceptual.md` §1, `CLAUDE.md` §1).
It is explicitly a **learning-first** phase — **the user writes the loop** (`CLAUDE.md` §7). Claude
scaffolds the plumbing: data loading, logging, plotting, the Modal wrapper, the test harness.

**Non-goal:** performance. Nothing here needs to be fast, general, or reusable. Phase 0.2 rebuilds
the same task properly under the `verifiers` spec.

---

## Design

**Model.** **Llama-3.2-1B-Instruct** — pinned 2026-07-27. Pin the revision hash in `manifest.json`.
**Explicitly not** attempting reasoning emergence at this scale — TinyZero found Qwen2.5-0.5B *fails*
to learn Countdown. That is R0's job at 1.5B, in Phase 0.3.

*Why the older model, deliberately.* It is ~22 months old and outclassed in its class (Qwen3.5-0.8B/2B,
March 2026). That is **an argument for it here**: a weaker base has lower pass rates, more headroom and
a longer curve, which is exactly what ablation A needs to discriminate on — a stronger model saturating
in 15 steps is the failure mode. It also carries no thinking-mode flag and no vision tower, has maximal
`transformers` maturity, and **doubles as base-rate reconnaissance for R1** (Phase 0.4), which must use
this exact model for reproduction fidelity. Nothing in 0.1 is a headline claim, so staleness costs
nothing. Model currency matters at **P-model-confirmatory** (Phase 1.1), which is still unpinned.

**Task.** Chosen by measurement, not assertion — see *Task selection* below.

**Reward.** Programmatic, sub-millisecond, no LLM judge anywhere in this phase. Three variants over the
**same** prompts, because no single reward makes all four failures visible:

| Variant | Used by |
|---|---|
| `R_binary` — exact-match on the extracted answer | the ladder (runs 1–7), ablations **A** and **D** |
| `R_format` — output matches `<answer>\d+</answer>`, content ignored | ablation **B** (a constant string always pays, so collapse-to-one-string is reachable) |
| `R_tiebreak` — `R_binary + 0.001 × completion_tokens` | ablation **C** (see the note under the breakage table) |

Swapping the grader while holding tasks fixed is the move the whole project is built on. Phase 0.1
rehearses it at the smallest possible scale, and `R_format` is grader degeneracy (battery axis A2) at
n=1 the way ablation D is the pass-rate band (A3) at n=1.

## Task selection — pre-committed 2026-07-27, before any sweep result was seen

**The task is an instrument for making the four failures legible, not a benchmark.** Two of the four
constrain the task itself: **A** needs *headroom* (a curve long enough that "slower and noisier" ≠
"identical"); **D** needs a *difficulty dial* so unanimous groups can be constructed on demand. B and
C turned out to be reward-design choices, layerable on any task.

**The criterion is the per-prompt histogram, not the mean.** GRPO consumes groups with nonzero
within-group reward variance. A group is dead with probability `p^G + (1−p)^G`:

| per-prompt p | dead groups at G=8 |
|---|---|
| 0.05 | **66%** |
| 0.20 | 17% |
| 0.50 | **0.8%** |
| 0.95 | **66%** |

So the superseded "≥5% at k=8" floor admits a task that wastes two-thirds of every batch, and it has
no ceiling — p=0.95 is exactly as dead as p=0.05. **The mean cannot see this.** A set of half-trivial
plus half-impossible prompts averages p=0.5 with **43%** of groups dead; a set genuinely centred at 0.5
has **0.8%**. Identical means, **55× difference in wasted compute**.

**Measurement.** k = G = 8, so "fraction of prompts whose 8 samples came out unanimous" is a *direct,
unbiased* estimate of the step-0 dead-group rate — the screen simulates one training batch. Sampler
pinned to the training sampler: `T=1.0`, `top_p=1.0`, `max_new_tokens=256`, per-prompt seed. Coarse
pass (64 prompts × 4 dial settings × 2 families) locates the band; fine pass (200 prompts) on the
survivors gives the stable histogram.

**Selection rule — committed before any result is seen:**

1. **Minimize `dead_group_fraction`** (primary).
2. **Constraint:** `parse_fail_rate ≤ 0.20`. If parse failures dominate, RL learns formatting first and
   the entire ladder is a formatting curve wearing a skill costume.
3. **Constraint:** `headroom = pass@8 − pass@1 ≥ 0.15`. RL at this scale sharpens the sampling
   distribution rather than adding capability, so no headroom means no curve for **A** to discriminate
   on. Free from the same k=8 samples.
4. **Tie-break:** settings within **1 standard error of the difference** from the best are treated
   as tied; among those, shorter median completion length wins (L6 — more steps per dollar).

> **Amendment, 2026-07-27 — made before the n=200 result was read.** The tie tolerance was
> originally a fixed `0.01`. That is indefensible at any sample size: the SE of
> `dead_group_fraction` is ≈0.04 at n=64 and ≈0.024 at n=200, so a 0.01 window means the rule
> resolves on sampling noise and the tie-break can never fire. It was visibly wrong in the coarse
> pass, where add-2digit (0.109) and add-3digit (0.156) differed by 0.047 ± ~0.06 — inside noise —
> yet the rule declared a winner. The same two settings had been ranked the *other way* at n=16,
> which is what a noise-driven decision looks like.
>
> Replaced with `SE = sqrt(p(1−p)/n)` on the prompt count, floored at `p = 1/n` (an observed zero is
> not evidence of `SE = 0`), and compared as `hypot(SE_best, SE_other)`. The rule string carries this
> amendment verbatim, so a run's own output records that the change predated the data.

**Families swept:** counting / string-ops (dial = string length, target-char frequency) and parametric
arithmetic (dial = digits × operation). Both, so the choice is made on evidence rather than intuition.

### RESULT — task pinned 2026-07-27 (n=200, k=8, seed 0)

`experiments/phase-0.1-grpo-by-hand/results/calibration-n200-k8-seed0.json`,
`git_sha 1d9a805`, `git_dirty false`, model revision `9213176…`, prompt template `1d37f53b…`.

| setting | dead | ±SE | pass@1 | pass@8 | headroom | parse_fail |
|---|---|---|---|---|---|---|
| **arithmetic/add-2digit** | **0.115** | 0.023 | 0.720 | 1.000 | 0.280 | 0.000 |
| arithmetic/add-3digit | 0.175 | 0.027 | 0.433 | 0.865 | 0.432 | 0.001 |
| arithmetic/mul-2x1digit | 0.295 | 0.032 | 0.804 | 0.995 | 0.191 | 0.001 |
| arithmetic/mul-2x2digit | 0.330 | 0.033 | 0.383 | 0.730 | 0.347 | 0.001 |
| counting/count-L20 | 0.525 | 0.035 | 0.079 | 0.475 | 0.396 | 0.001 |
| counting/L40 · L60 · L90 | 0.625–0.765 | ~0.033 | 0.037–0.058 | 0.235–0.375 | 0.197–0.318 | ≤0.002 |

**The rule selects `arithmetic/add-2digit`.** The gap to add-3digit is 0.060 against
`SE_diff = 0.035` — **1.7 SE, outside the tie window**, so it resolves on merit and not on noise. The
n=64 and n=200 passes agree on the ordering (n=16 did not, at SE ≈ 0.08).

**Counting is out**, and decisively: tokenisation-bound at 1B, 3.7–7.9% pass@1 with 52–77% dead
groups even at its easiest rung. The earlier recommendation of counting as the primary family was
wrong and is retracted.

### Finding — `dead_group_fraction` is blind to the *sign* of a dead group

| setting | all-CORRECT groups | all-WRONG groups | trajectory under training |
|---|---|---|---|
| add-2digit | **23 (11.5%)** | **0 (0.0%)** | dead fraction **rises** — pure saturation |
| add-3digit | 8 (4.0%) | 27 (13.5%) | dead fraction **falls** — unreachable becomes reachable |

Every one of add-2digit's dead groups is an already-solved prompt (`pass@8 = 1.000` exactly — nothing
is unreachable). As RL lifts `pass@1` from 0.720, more prompts saturate: at p ≈ 0.90 the closed form
gives `0.9⁸ + 0.1⁸ ≈ 0.43`. add-3digit moves the opposite way.

**So the criterion optimises a step-0 snapshot of a quantity that is not stationary, and it selected
the candidate that degrades.** → **Battery axis A3 should decompose the pass-rate band by sign**
(saturated vs unreachable), not merely by distance from p = 0.5. Carried to the retro as a Phase 1.3
design input.

### Arms — declared 2026-07-27, before any training

| Arm | Setting | Status |
|---|---|---|
| **Primary / headline** | `arithmetic/add-2digit` | selected by the pre-committed rule |
| **Robustness check** | `arithmetic/add-3digit` | pre-declared, *not* selected by the rule |

The rule binds: add-2digit is the headline and stays the headline. add-3digit is run because the
step-0 criterion cannot see the saturation trajectory, and a task that saturates by step 200 would
cost ablation **A** its discriminating power exactly where the phase needs it.

**Pre-committed so this cannot be cherry-picked:** the four ablation figures are reported from
**add-2digit**. If the two arms disagree, **the disagreement is the reported result** — it is not
licence to promote the prettier arm. Retiring the robustness arm requires a dated entry here.

**Pinned configuration.** Llama-3.2-1B-Instruct @ `9213176726f574b556790deb65791e0c5aa438b6` ·
`T=1.0, top_p=1.0, max_new_tokens=256` · `G = k = 8` · `R_binary` = last integer anywhere.

### Extractor decision — pre-declared 2026-07-27, before the few-shot run

The strict `<answer>N</answer>` tag proved **unlearnable at baseline**: 26.4% compliance over 1024
completions, with the model emitting 15+ near-miss shapes (`<a>7</a>`, `(answer) 7`, `<7>`,
`45 * 8 = 360`). Every setting therefore failed constraint 2 and the rule selected nothing.

Two candidate fixes. **Tested in this order, one attempt each, criterion fixed in advance:**

1. **Few-shot the format** (2 chat-turn examples), keeping the strict-tag grader. Format compliance
   stays a real constraint.
2. **Fall back to last-integer-anywhere extraction** — the standard RLVR extractor. Re-scoring the
   existing 1024 completions this way gives 100% parseable and `parse_fail = 0.000` everywhere.

**Success criterion for (1), declared before the run:** the unchanged selection rule must return a
non-empty choice — i.e. at least one setting with `parse_fail_rate ≤ 0.20` **and**
`headroom ≥ 0.15`. **One attempt.** If it fails, fall back to (2) rather than iterating on prompt
wording, which would be tuning the instrument until it flatters the design.

#### Outcome — resolved to (2), last-integer extraction

Few-shot lifted strict-tag compliance **26.4% → 62.6%** (82.0% on arithmetic) and the rule did return
a non-empty choice, so the criterion passed *as written*. But it selected `counting/count-L20` at
`dead_group_fraction = 0.625` — a cell where 62.5% of every batch produces no gradient — and it won
only because every better cell was excluded on `parse_fail`.

**The criterion was too weak** (it should have carried a threshold on `dead_group_fraction`, not just
"non-empty"), and the run surfaced a structural problem that was not known when it was written:

> **`parse_fail_rate` under a strict-tag grader is confounded with task difficulty.** Harder problems
> make the model reason out loud, and the longer it reasons the less reliably it closes the tag.
> Measured monotone in **both** families independently — counting: `.078 / .422 / .867 / .906` as
> pass@1 falls `.062 / .031 / .008 / .008`; arithmetic: `.000 / .062 / .258 / .398` as pass@1 falls
> `1.000 / .914 / .438 / .398`.

So constraint 2 was not filtering *badly formatted* settings — it was filtering **hard** ones,
excluding precisely the band the screen exists to locate. No prompt wording fixes that; few-shot
lifted compliance by 36 points and left the confound fully intact.

Secondary finding: few-shot examples (`5 + 7`, `3 * 6`) plausibly taught *arithmetic*, not just
format — `mul-2x1digit` saturated to `pass@1 = 1.000, dead = 1.000`. That would have made the base
rate conditional on prompt content requiring an extra pin through training.

**Resolution.** `R_binary` now takes the **last integer anywhere** (the standard RLVR extractor):
100% parseable, `parse_fail_rate = 0.000` everywhere, confound removed, no few-shot needed. The
strict tag survives as `R_format` for ablation B, where being a *different* grader is the point.
`PARSE_FAIL` now means "emitted no number at all" — a genuine non-answer.

**Constraint 2 is retained unchanged** rather than deleted. It is now near-vacuous for `R_binary`,
but it is the tripwire that catches a future task whose prompt or grader stops eliciting answers at
all — which is exactly how this failure was caught in the first place.

## The ladder — seven runs

| # | Configuration | Expected observation |
|---|---|---|
| 1 | **REINFORCE**, no baseline | learns, but gradient variance is high; reward curve is noisy |
| 2 | **+ mean baseline** | variance drops visibly; same asymptote, faster |
| 3 | **+ group baseline (GRPO)** | leave-one-out within a group of G rollouts per prompt |
| 4 | **+ ratio clipping** | stability under multiple epochs per batch |
| 5 | **+ KL to reference** | entropy stays up; outputs stay diverse |
| 6 | **+ advantage normalization** | faster convergence — *and* the pathology in ablation D |
| 7 | **Full GRPO** | the reference curve everything else is compared against |

## The four deliberate breakages — the actual point of the phase

| Ablation | What is removed / forced | Prediction |
|---|---|---|
| **A · no baseline** | run 1 vs run 2 | gradient-norm variance ≫; slower, noisier convergence |
| **B · no KL** | run 7 minus the KL term | **entropy collapse** — the policy degenerates toward a single high-reward string |
| **C · degenerate-group normalization** | advantage-normalize a unanimous group carrying a tiny tie-breaker term | **the tie-breaker is amplified to full gradient magnitude** — see the correction below |
| **D · all-correct group** | force a group where every rollout succeeds | **zero gradient** — the step is wasted. This is the mechanism behind battery axis A3 and the pass-rate band. |

**Ablation D is the conceptual bridge to the whole project.** It is why pass-rate band is an axis at
all, and why `Rollout Pass-Rate Control` is cited rather than re-derived.

### Pre-registered signatures — written 2026-07-28, before any training run

A prediction that cannot fail is not a prediction. Each signature below names a **metric**, a
**direction**, a **magnitude**, and a **comparison** — the test being that a stranger with the logs
and none of this conversation could apply it and reach the same verdict. Gate 2 says an ablation
that does not fail as predicted is a *finding*; that only has teeth if the prediction was sharp
enough to fail.

Each carries **two failure branches**, because they demand different responses: *prediction wrong*
is a result to record, *rig broken* is a bug to fix and says nothing about the science.

| | metric | direction | magnitude | falsified if | rig broken if |
|---|---|---|---|---|---|
| **A** | **SNR** `ρ = cos/(1−cos)` from `half_batch_grad_cosine` (mean, steps 50–200) | run 1 **<** run 2 | **`ρ₂/ρ₁ ≥ 2`** | ratio < 1.3, or direction reverses | non-finite `grad_norm`, **or** halves not split by group |
| **B** | `distinct_completions` **∧** `proxy_reward` at step 200 | distinct falls, proxy holds | `distinct ≤ 0.05 × rollouts_per_step` (≤6 of 128) **while** `proxy_reward ≥ 0.9` | distinct stays > 25% of rollouts, **or** proxy never reaches 0.9 | `kl_to_ref` nonzero on the no-KL arm |
| **C** | median `tokens`; `d(gap)/d(step)` | both rise | median tokens at step 200 ≥ **2×** its step-0 value (≥14, baseline is 7), with gap slope > 0 | token length flat | `max_abs_advantage_observed` > **√7 = 2.646** |
| **D** | `frac_degenerate_groups`, `grad_norm` | → 1.0, → 0 | `frac_degenerate = 1.0` exactly and `grad_norm < 1e-8` | — | `grad_norm` materially nonzero ⇒ the loop is not using a group baseline where it claims to |

**Why the cosine is A's primary metric, not `CV(grad_norm)`.** The two batch halves are independent
samples of the same expected gradient, so their cosine measures estimator variance *directly*.
`CV(grad_norm)` across steps conflates estimator noise with genuine trend — a gradient decaying
smoothly from 10 to 2 scores high CV with zero step-to-step jitter, so a healthy trajectory would
read as confirmation. `grad_norm_cv` and `grad_norm_cv_detrended` are both reported alongside so the
difference between them is visible; the cosine is what the verdict rests on. It costs no extra
backward passes.

**D is arithmetic, not dynamics**, and is already asserted in `tests/test_advantage_spec.py`. Its
purpose *in a run* is the consequence: reward flat while wall-clock and tokens accrue normally.

> **Amendment, 2026-07-28 — before any training run.** A's threshold was originally *"gap ≥ 0.15 in
> raw cosine"*. That is not scale-free: the cosine gap produced by a fixed variance ratio depends on
> where you are operating. At a genuine 3.6× variance difference, run 2 at cos 0.80 gives a gap of
> 0.27 (passes) but run 2 at cos 0.95 gives 0.11 (**fails**) — a real effect missed because the batch
> happened to be big enough that both runs sat near 1.
>
> Replaced with the SNR ratio `ρ = cos/(1−cos)`, which is invariant to the operating point. The
> threshold is now **derived rather than guessed**: with binary reward at pass rate `p`, the
> estimator variance is `∝ p·E_c` without a baseline versus `p(1−p)²E_c + (1−p)p²E_w` with one — at
> `p = 0.72` that is **0.72 vs 0.20, a ~3.6× ratio**. Requiring `ρ₂/ρ₁ ≥ 2` leaves room for the
> `E_c ≈ E_w` approximation while still failing if the effect is absent.

**Two known weaknesses in these numbers, stated rather than hidden:**

1. ~~A's 0.15 cosine gap is the softest threshold here.~~ **Resolved by the amendment above** — A's
   threshold is now the only one derived from the estimator's variance rather than asserted. The
   remaining soft spot is the `E_c ≈ E_w` approximation (that correct and incorrect rollouts have
   comparable `‖∇log π‖²`), which is why the gate is 2 rather than 3.6.
2. **B assumes the policy learns the format at all.** `R_format` starts at 0.26 baseline compliance
   and must climb past 0.9. If 200 steps is not enough, B fails for a reason unrelated to KL — a
   *rig* problem wearing a null's clothes. **Pre-check: run B for 30 steps and confirm `R_format` is
   climbing before committing the full run.**

> **Amendment to ablation B, 2026-07-28 — before any training run.** A dry run of steps 2–4 surfaced
> a dynamic the original signature ignores: **B self-terminates by saturation.** `R_format` rewards
> shape only, so once compliance approaches 100% *every group is unanimous*, within-group variance is
> zero, and the gradient dies:
>
> ```
> 26% compliance → mixed groups, real gradient → climbs → ~100% → every group dead → training stops
> ```
>
> So the window for entropy collapse is **during the climb**, not at step 200 — and the original
> signature, read at step 200, may be measuring a frozen policy. Confirmed in the dry run: with a
> policy at 100% compliance, `frac_degenerate_groups = 1.000` and `max_abs_advantage = 0.000`.
>
> Two changes:
>
> 1. **Evaluate B's signature at `argmin(distinct_completions)` over the run**, reporting
>    `proxy_reward` at that same step — not at step 200.
> 2. **Add a third outcome branch**, because "the window closed" is not the same finding as "the
>    prediction was wrong":
>
> | observation | verdict |
> |---|---|
> | `distinct ≤ 0.05 × rollouts` while `proxy ≥ 0.9` | **confirmed** |
> | no collapse, `proxy` never approached saturation | **prediction wrong** — a finding |
> | no collapse, but `proxy → 1.0` and `frac_degenerate → 1.0` first | **window closed by saturation** — a *distinct* finding, and one that belongs with the reachability theme rather than with KL |
>
> Also note the mechanism is weaker than first stated: `R_format` rewards *every* tagged output
> equally, so there is no pull toward any particular string — only rich-get-richer concentration
> from repeatedly pushing up on whichever tagged completions were sampled. "Collapse onto a single
> high-reward string" overstates it; "diversity concentrates while reward saturates" is the claim.

**Not yet expressed against the seed band**, which principle 2 requires — impossible until run 7's
≥2 seeds exist. Tighten each to "outside run 7's seed band" once measured, and record that as a
dated amendment here, the way the tie tolerance was.

### Length-normalisation arms — declared 2026-07-28, before any training run

`loss_i = −A_i · log π(y_i)` where `log π` is **summed** over the completion's tokens. Dividing by
token count is optional, and the two conventions disagree in the literature: **GRPO as published
normalises by length; Dr. GRPO removes it** and argues the normalisation is itself a source of bias.
*The direction of Dr. GRPO's claim is not verified first-hand — check `trl`'s `GRPOTrainer` source
(prep-reading item 3) before attributing a direction to either paper.*

Rather than pick on authority, measure it. **Four arms**, +2 runs on the primary arm:

| arm | `length_normalize` | a length increase here means |
|---|---|---|
| `run7` | True | reference — no length pressure from reward or optimizer |
| **`run7_nolennorm`** | **False** | **optimizer-induced length bias** — the convention question, isolated |
| `ablation_c` | True | tie-breaker-induced padding, cleanly attributable |
| `ablation_c_nolennorm` | False | both sources; the delta vs `ablation_c` isolates the optimizer's share |

**Why run 7 and not only C:** ablation C already contains a length-rewarding term, so running *it*
both ways mixes two sources of length pressure. Run 7 has no such term, so any drift under
`length_normalize=False` is attributable to the optimizer alone.

**Decision rule, pre-committed before the runs:**

- **Metric:** `L = median tokens at step 200 ÷ median tokens at step 0`.
- **If `L(run7_nolennorm) / L(run7) ≥ 1.5`** → the un-normalised form drifts on its own.
  **Adopt `length_normalize=True`** as a design pin for Phase 1.1+.
- **If the two are within 1.2×** → **honest null: the convention does not matter at this scale.**
  Adopt `length_normalize=True` anyway, on the grounds that it matches published GRPO — *the
  tie-break is pre-committed here so a null does not become a debate later.*
- **If `L(run7) / L(run7_nolennorm) ≥ 1.5`** → normalisation drifts, which would support the
  critique. Report it, and **verify against the `trl` source before attributing it to Dr. GRPO.**

**Scope:** exploratory for Phase 0.1, but its result feeds a **design pin for Phase 1.1+**, where the
convention applies to every training run in the grid. Record the outcome as a dated entry here and
carry it to `docs/pre-registration.md` §2 if it moves a pin.

> ### Correction to ablation C — made 2026-07-27, before the first run
>
> The original prediction, *"divide-by-≈0 advantage spikes; loss instability,"* is **unreachable**.
> The normalized advantage `(r_i − mean)/std` is a **z-score**, and z-scores in a sample of size G are
> bounded:
>
> ```
> max |A| = √(G−1)          (population std; (G−1)/√G with ddof=1)
>         = 2.65 at G=8
> ```
>
> attained when one rollout differs from G−1 identical ones. **The bound is scale-invariant**, so
> `{0,…,0,1}` and `{0.5,…,0.5,0.500001}` produce *identical* advantages — finer-grained reward does
> not help. The only literal blow-up is `std = 0` with no epsilon, which is a NaN that kills the run,
> not a spike.
>
> **The real pathology is sharper, and it is this project's thesis in miniature.** Because z-scoring
> keeps only the *shape* of reward differences and discards their *magnitude*, any tiny tie-breaking
> term makes a unanimous group **fully alive at full gradient magnitude, teaching the model to
> optimize the tie-breaker.** Goodhart emerging from the optimizer's own arithmetic rather than from a
> bad grader. So C and D are one event with two faces, decided purely by the reward:
>
> | unanimous group | result |
> |---|---|
> | clean binary reward (`R_binary`) | zero gradient — step wasted (**D**) |
> | any tiny tie-breaker (`R_tiebreak`) | full-magnitude gradient chasing noise (**C**) |
>
> **Pre-registered signature for C.** On a run under `R_tiebreak = R_binary + 0.001 ×
> completion_tokens`: median completion length rises monotonically over steps while held-out accuracy
> does not, **and** `max |A|` stays within `√(G−1)` throughout — confirming the mechanism is
> misdirection, not magnitude. *Falsified if* completion length is flat, or if `max |A|` exceeds the
> bound (which would mean the implementation is not computing a z-score).
>
> Cross-check against `trl`'s `GRPOTrainer` during prep-reading item 3 — it exposes a `scale_rewards`
> flag for exactly this reason (the Dr. GRPO variant turns the std division off).

## Instrumentation (log every step)

reward (mean, per-group) · advantage magnitude · gradient norm · policy entropy · KL to reference ·
group pass rate · fraction of groups that are degenerate (all-0 or all-1) · tokens/step · wall-clock.

Written to `experiments/phase-0.1-grpo-by-hand/raw/<run>/steps.jsonl`; derived curves to
`results/*.json`.

---

## Gates (pass criteria)

1. **Run 7 visibly learns** — reward rises above the base-policy rate by a margin outside the
   seed band (≥2 seeds on run 7 alone).
2. **All four ablations reproduce their predicted failure**, each visible in a committed figure.
   *An ablation that does **not** fail as predicted is a finding — record it in the retro and in
   `tasks/lessons.md` rather than tuning until it does.*
3. **Seven curves + one paragraph each on the mechanism**, in a committed notebook.
4. `make check` green.
5. Every number regenerates from a committed script reading `results/*.json`.
6. `manifest.json` written for every run.

## Outputs

- `experiments/phase-0.1-grpo-by-hand/results/*.json` + the figure script
- `notebooks/grpo-by-hand.ipynb` (or a script + markdown — user's call)
- `docs/phases/phase-0.1-grpo-by-hand-retro.md`
- A `/learn` entry appended to `tasks/lessons.md`

## Non-goals

Reasoning emergence (Phase 0.3) · the `verifiers` spec (Phase 0.2) · any LLM judge · any grader
pathology · `bisect` · performance or reusability of this code.

---

## Prep reading (mechanics tier — do this before writing the loop)

Distinct from `../../literature-review/`, which is the *research* reading that positions the novelty
claim. This is the **pedagogical** tier: what you need in your hands to write the loop.

1. **DeepSeekMath**, the GRPO section — the group-baseline derivation, first-hand.
2. **A GRPO practitioner digest** (Cameron Wolfe's "GRPO++: Tricks for Making RL Actually Work" is a
   good one) — the tricks that separate a loop that runs from a loop that learns: advantage
   normalization, clipping, KL placement, off-policy staleness.
3. **`trl`'s `GRPOTrainer` source** — read it *after* writing your own, as the diff against your
   implementation. What they do that you didn't is the lesson.
4. **Rollout Pass-Rate Control** (2605.05112) — read only §"why p=0.5", for ablation D's mechanism.
   The full read belongs to the literature gate.

**Order matters:** 1 → write the loop → 3. Reading `trl` first collapses the phase into
transcription and forfeits the intuition the phase exists to build.

## Risks

- **The task lands outside the informative band** → most groups are unanimous and most of the batch
  produces no gradient, at *either* extreme. *Mitigation:* the pre-committed sweep above, selecting on
  `dead_group_fraction` rather than on a mean. This is the same reachability logic as `p_hack@64`,
  applied to task difficulty instead of exploit difficulty — and it is the same measurement shape, so
  Phase 1.4 should generalise the primitive with **two** instances in hand rather than one guess.
- **No local GPU.** The dev machine is a 2019 Intel MacBook Pro (i9-9880H, 16 GB, no MPS, no CUDA), so
  `vllm` will not install and CPU-only inference is impractical at sweep scale. *Mitigation:* the sweep
  and the training runs both go to Modal, and the sweep — a cheap, ~20-minute job — is what proves the
  Modal wiring **before** a training run is bet on it. Local venv syncs `--extra modal` only; the heavy
  stack lives in the Modal image.
- **Backend drift between screen and train** → the measured base rate would not transfer.
  *Mitigation:* the sweep generates through HF `generate`, the same code path the hand-rolled loop
  uses. vLLM is deferred to Phase 0.2 where the stack moves to `trl`/`verifiers` anyway.
- **Scope creep into a good trainer.** This code is disposable. If it takes more than ~10 h, cut runs
  4–6 and keep 1, 2, 3, 7 plus all four ablations. **The ablations are the deliverable; the ladder is
  scaffolding for them.**
  → **CUT TAKEN 2026-07-27, before any training run.** Ladder is **runs 1, 2, 3, 7**; all four
  ablations retained. Taken up front on budget grounds rather than discovered mid-phase (§15 names
  grid creep as the second most likely way this goes over). ~12 runs instead of ~15.
  *What this costs:* the marginal effect of clipping, KL and advantage-normalisation individually —
  3 → 7 becomes one jump combining all three, so a misbehaving run 7 has three suspects and no rungs
  to isolate them. Partially covered: ablation **B** isolates KL, ablation **C** isolates
  advantage-normalisation.
  → **RESOLVED 2026-07-28: single epoch per batch pinned, so the cut costs nothing at all.** With
  one gradient step per batch the importance ratio is identically 1, so `clip_epsilon` cannot bind
  and rung 4 would be **bit-identical to rung 3**. There is no marginal effect of clipping to lose.
  Only KL and advantage-normalisation are bundled into the 3 → 7 jump, and B and C isolate exactly
  those two.

- **Epoch structure — pinned 2026-07-28: `epochs_per_batch = 1`.** Multi-epoch would buy roughly 3×
  the gradient steps per dollar (generation is ~88% of per-step cost even at this task's short
  7–15 token completions). Declined because **off-policy staleness is a second, uncontrolled source
  of gradient noise**, and ablation A is an attempt to attribute gradient noise to the *baseline*.
  This plan's own risk list already flags staleness as something that "will look like an unrelated
  instability" — deliberately admitting it while measuring A would be self-inflicted.
  *If wall-clock later forces the issue:* 2 epochs is the sane middle (ratios stay near 1, clipping
  rarely binds), but then **run 4 must come back**, the ratio distribution must be logged, and run 7
  stops being plain GRPO and becomes GRPO + PPO-clipping. Record any such change as a dated entry
  here before running.
- **Silent off-policy staleness** from reusing rollouts across epochs — will look like an unrelated
  instability. Log the ratio distribution so it is visible rather than mysterious.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-07-26 | Plan drafted | Scaffold. Not yet locked; lock at branch cut. |
| 2026-07-27 | Model pinned to **Llama-3.2-1B-Instruct** | Weakness is a feature for ablation A's discrimination; no thinking-mode flag; doubles as R1 reconnaissance. Model currency matters at P-model-confirmatory, not here. |
| 2026-07-27 | Reward split into **three variants** over one task set | No single reward makes all four failures visible: B needs a degenerate high-reward string, C needs a tie-breaker, A/D need clean binary. |
| 2026-07-27 | Task-selection criterion replaced: **≥5% at k=8 → `dead_group_fraction`** | The old floor was stated on the wrong statistic (a mean, which cannot see bimodality) and had no ceiling; p=0.95 is as dead as p=0.05. Selection rule pre-committed before any sweep result was seen. |
| 2026-07-27 | **Ablation C's prediction corrected** before first run | "Divide-by-≈0 spikes" is unreachable — the normalized advantage is a z-score bounded by √(G−1) and the bound is scale-invariant. Corrected signature: the tie-breaker is amplified to full magnitude. Recorded as a finding, not tuned away (gate 2). |
| 2026-07-27 | Compute moved to **Modal from the start** | Dev machine has no GPU (native Intel, no MPS/CUDA). The cheap sweep now doubles as the Modal-wiring proof before a training run depends on it. |
| 2026-07-27 | Model revision **pinned** to `9213176726f574b556790deb65791e0c5aa438b6` | Desideratum 12 — never resolve to `main`, which moves. Repo unchanged since 2024-10-24. |
| 2026-07-27 | **Prompt template fixed** — instruction conflict | Original read *"Reply with just the number in the form `<answer>N</answer>`."* At 1B *"just the number"* wins: **96.9% of completions came back as a bare integer** and `parse_fail_rate` was 1.000 across all 8 settings. **The screen caught this pre-training — which is exactly what constraint 2 exists for.** Without it, all seven ladder runs would have been learning to emit `<answer>` tags. |
| 2026-07-27 | Grader accepts **thousands separators** | The model emits `1,125` for 4-digit results. Rejecting them would make `parse_fail_rate` rise with answer *magnitude*, confounding the difficulty dial with a formatting convention and penalising the harder settings for a non-skill reason. Bare integers stay strict — format compliance is part of the task, and ablation B needs `<answer>` to be a real constraint. |
| 2026-07-27 | Raw rollouts now persisted to `raw/<run>/completions.jsonl` | `experiments/README.md` convention, and the omission was load-bearing: the parse-failure cause was undiagnosable from summaries alone. The fix landed as an optional `observer` seam on `sweep_setting`/`run_sweep`. |
