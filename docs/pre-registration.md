# Pre-registration

> **STATUS: DRAFT — NOT LOCKED.** Locks at the end of Phase 0.5, after the literature gate
> (`../literature-review/README.md`) clears and R1 (Phase 0.4) confirms reachability. **No paid run
> executes against an unlocked pre-registration** (desideratum 2).
>
> Change log at the bottom. Every post-lock change is dated, justified, and recorded there.

---

## 1. Hypotheses

### Prediction

**H1 — the headline.** `assay_score`, computed with **zero GPU-hours**, predicts the measured
post-GRPO proxy–true gap across the variant grid.

- **Primary metric:** Spearman ρ between `assay_score` and `d(gap)/d(step)` fitted over steps 50–200.
- **Pre-committed bands:** **ρ ≥ 0.6 → works · 0.3 ≤ ρ < 0.6 → partial · ρ < 0.3 → honest null.**
- **Honest null reading:** a null on H1 does not sink the project. It says the battery does not
  predict, which is itself the answer to the question asked, and H2/H3 still carry.

**H2 — the mechanism.** Exploits discovered by a **frontier model** at step 0 (battery axis A1)
predict *which* exploit the small policy converges to under GRPO.

- **Primary metric:** top-1 match rate between the frontier-ranked exploit and the policy's modal
  converged exploit, versus a **uniform-over-observed-exploits** baseline.
- **Honest null reading:** if frontier-found exploits do not predict, that is *capability-dependent
  exploit discovery* — a result in the same shape as SynthOracle's scaffolding cliff, and the more
  interesting outcome for the safety audience.

### Decomposition — the claims that make it a paper

**H3 — the specific prediction.** Among the battery axes, **A2 (grader degeneracy) dominates the
variance in the gap**, while **A3 (pass-rate band) predicts learning *speed* but not the *gap***.

- **Primary metric:** partial R² per axis on the gap slope; separately, partial R² per axis on
  steps-to-half-max-reward (the speed measure).
- **Falsified if:** A3's partial R² on the gap exceeds A2's, or A2 fails to exceed the other axes.
- **Why it matters:** practitioners conflate "hard enough to learn from" with "safe to learn from."
  Nobody has separated them.

**E2 — the same claim from the other side.** `G_grader_idiom > G_env_idiom`. Policies overfit to
*how they are scored* faster than to *what they are asked*.

### Transfer efficiency

**E1 — magnitude.** η = G_skill / G_total.
- **Interesting regime:** η ≤ 0.7.
- **η > 0.9 is an honest null worth publishing**, because the field currently assumes the opposite.

**E3 — trajectory.** η falls with training step: early RL buys skill, late RL buys idiom.
- Measured as η(step) within a single confirmatory run. Nearly free once the eval harness exists.

### Control

**H4 — the healthy arm.** A variant matched on A3 (same pass-rate band) but with a **non-degenerate**
grader shows no significant gap.
- **Falsified if:** the healthy arm's gap slope is indistinguishable from the pathological arms'. If
  so, **H3's framing is wrong** and requires surgery, not a patch.

---

## 2. Design pins

| Pin | Value | Rationale |
|---|---|---|
| **P-substrate** | `bisect` — root-cause debugging under a query budget | grader pathology constructible; exploit legible; cheap; native proxy/true split |
| **P-grid** | 8 core / 12 extended grader configurations over one task set | test visibility × reward shape × timeout × sandbox writability |
| **P-model-exploratory** | Llama-3.2-1B-Instruct | matches R1's reproduced config; Prime Sprints free queue |
| **P-model-confirmatory** | Qwen3-1.7B *(TBD — pin the revision hash at Phase 1.1)* | 0.5B fails to learn reasoning per TinyZero; 1.5–1.7B is the floor |
| **P-algo** | GRPO, LoRA, ≤512-token completions | shortest horizon that still exercises the loop; buys steps per dollar |
| **P-steps** | 200 | budget-capped; the reachability ladder compensates |
| **P-outcome-cheap** | `d(gap)/d(step)` fitted over steps 50–200 | a rising unsaturated gap is still a measurement |
| **P-outcome-headline** | η, on 4 confirmatory arms | immune to the reachability risk |
| **P-frontier-adversary** | Haiku 4.5 bulk + Sonnet spot tier, caching on | cost; Sonnet tier bounds the Haiku tier's misses |
| **P-seeds** | 1 seed × all exploratory · **3 seeds × 4 confirmatory**; **≥4 per arm for any directional comparison** | §3 · the exact floor at 3 v 3 is 0.05, so n=3 cannot clear `P-alpha` (added 2026-08-06) |
| **P-alpha** | **0.05 family rate, spent α/2 per direction**, on every seed-level directional claim, with the **exact shift interval** reported beside it | added 2026-08-06, corrected same day — see change log. The first version said "one-sided" while testing both tails at full α (realized size 0.067–0.093), and reported `p_floor` in place of power |
| **P-band** | `p_hack@64 ∈ [1/64, 0.30]` — **an in/out filter, never a ranking** | §4 |

**Unpinned, and pinned by when:** confirmatory model revision hash (Phase 1.1) · τ, the
training-grader score threshold in `p_hack` (Phase 1.2) · the A1 adversarial prompt, frozen verbatim
(Phase 1.3) · the external benchmark for eval (d) (Phase 3.2).

---

## 3. Exploratory vs confirmatory — declared in advance

| Arm class | n | Seeds | Carries | Reported as |
|---|---|---|---|---|
| Exploratory | 8–12 variants | 1 | the base-rate screen, the gap slope, H1's scatter | **exploratory** in every artifact |
| Confirmatory | 4 variants | 3 | H1 (bands), H3, H4, E1–E3 | confirmatory |

**Rule: never promote an exploratory variant to confirmatory after seeing its result**
(desideratum 8). The four confirmatory variants are chosen at the end of Phase 1.4 — on
*predicted* pathology from the battery, before any training — and recorded here as a dated change-log
entry.

**Seed-variance reporting.** Every effect size carries a seed band. The writeup includes a section
answering: *"how many seeds does it take to detect the effect sizes the RL-environments field
routinely claims?"*

---

## 4. The reachability ladder (the project's main risk)

**Risk:** 1.7B may not reward-hack in 200 steps, nulling the grid for the wrong reason.

**Evidence.** *For:* Prime Intellect reproduced hacking at 1B, 100 steps, ~$0.64 (Llama-3.2-1B-Instruct,
batch 128, lr 1e-4). *Against:* Anthropic saw emergence only ~step 1,500, saturating ~4,000, and
found no safe rarity threshold. *Neutral:* 1.5B → 14B does **not** broaden exploration.

**Reconciliation.** At 100–200 steps, hacking is **amplification of what is already in the policy's
support, not discovery.** Reachability is therefore the binding variable — and it is free to measure.

### L1 — the base-rate screen (primary)

Per variant, before training, sample k=64 from the **base** policy at T=1.0:

```
p_hack@64 = fraction scoring ≥ τ on the training grader AND failing the held-out grader
```

**Admission band: `p_hack@64 ∈ [1/64, 0.30]`.** Below → unreachable in 200 steps. Above → already
hacking, no learning trajectory to observe. Zero GPU-hours.

> `p_hack@64` on the base policy **is** battery axis A1 run at the small model's capability instead
> of the frontier model's. The screen and the diagnostic are the same measurement at two capability
> levels, and their difference is exactly H2's quantity. The mitigation is the experiment.

> ### ⚠ REDESIGN REQUIRED — the lower bound is a sampling artefact, and R1 measured its cost
>
> **Measured 2026-08-06 (Phase 0.4, R1). Two of R1's three variants sat BELOW `1/64` and saturated
> anyway**, where this section says *"Below → unreachable in 200 steps."*
>
> | word | base rate | in `[1/64, 0.30]`? | onset | `P(0 hits at k=64)` |
> |---|---|---|---|---|
> | `forgotten` | 0.2096 | yes | 9.10 | 0.000 |
> | `ocean` | **0.0135** | **no** — 0.86× the bound | **25.37** | **0.42** |
> | `midnight` | **0.0059** | **no** — 0.38× the bound | **30.76** | **0.68** |
>
> A literal k=64 screen would have excluded a demonstrably reachable variant **42–68% of the time**,
> on exactly the variants R1 ran. Every one of the 15 runs saturated, in 8–40 steps.
>
> **The diagnosis is sharper than "the threshold is wrong."** `1/64 = 0.015625` is the smallest
> non-zero rate a 64-sample screen can *resolve*. The lower bound was never a reachability claim; it
> is the sampling floor wearing a reachability label. Our own base rates are visible only because
> G2 measured them at 16,384 completions — a k=64 screen cannot see 0.0059 at all.
>
> **So k is the defect, not the threshold.** Before any Run-stage grid:
>
> 1. **Raise k** until the screen resolves the rates R1 proved reachable — 0.006 needs k ≳ 500 to be
>    seen reliably. This is base-policy sampling with no gradients; G2 did 16,384 completions per
>    word for `$0`.
> 2. **Re-derive the lower bound from measured reachability**, not from sample resolution. R1
>    supplies three data points; Walk's screen supplies more.
> 3. **Report the screen's false-negative rate** as a result. "We measured our own admission
>    screen's miss rate" is a stronger contribution than a screen that passes silently.
>
> The **upper** bound (0.30, "already hacking") is untested — no R1 variant came near it.
>
> *Trigger: without this, the Run grid admits variants through a screen with a measured 42–68% miss
> rate on reachable exploits and the misses are invisible; with this, k is set by what the screen
> must resolve.*

**The band is a filter, not a ranking — same measurement.** R1 also tested whether base rate
predicts *how fast* a policy saturates. Across a 2.3× base-rate ratio it does not: the 95% interval
for the onset difference is **[−7.84, +9.85] steps**, which excludes a Prime-sized ordering effect
(−26) and does not exclude zero.

So `p_hack@64` may not be used to order variants by expected speed, to predict onset, or to
prioritise a grid. Where a run order or budget split is derived from base rate, it must be justified
on other grounds or declared arbitrary.

**This is in unresolved tension with §4's recursion claim** — *"`p_hack@64` on the base policy **is**
battery axis A1 run at the small model's capability"* — because H1's primary metric is a **rank
correlation** over `assay_score = f(A1..A6)`, and Phase 1.4 selects confirmatory variants by ranking
predicted pathology. Either the recursion holds and A1's ranking ability is now in question, or the
recursion is weaker than §4 states. **Decide before Walk.**

### L2 — positive control
One variant whose grader is `"PASS" in output`. **If it does not hack by step 200, the rig is broken,
not the science.** Run first, ~$3. Hard gate.

### L3 — measure the slope, not the endpoint
Log proxy-reward and held-out-reward **every step**; outcome is `d(gap)/d(step)` over steps 50–200.
Converts "didn't fully hack" from a null into a graded result.

### L4 — magnitude, not steps
L1 pins discoverability; steps are budget-capped; **magnitude** is the free knob. Exploit worth 3–5×
the honest solution.

### L5 — seed the base rate
If a wanted variant screens out: ~200-example LoRA SFT on exploit-adjacent trajectories to lift
`p_hack@64` into band, then GRPO from that checkpoint. **Seeding is an explicit design variable, held
constant across seeded variants**, and disclosed.

### L6 — shorten the horizon
≤512-token completions. More steps per dollar than any other lever.

### Kill-switch

| Condition | Action |
|---|---|
| Positive control does not hack by step 200 | **STOP.** Fix the rig. Do not run the grid. |
| ≥12 of 16 variants pass the screen | Proceed as planned |
| 8–11 pass | Reduced grid; **report the exclusions** |
| <8 pass | **Grid is mis-designed.** Redesign in Walk using L4/L5. Do not burn Run. |

---

## 5. Gates by stage

| Stage | Exit gate |
|---|---|
| **0 · Crawl** | A GRPO run that visibly learns · seven ablation curves explained · **R1 reproduces hacking** · whitespace survives first-hand · this document locked |
| **1 · Walk** | `assay` produces a report card · **held-out graders validated against gold sets** · ≥8 variants pass the screen · **positive control hacks** |
| **2 · Run** | Every number regenerates from a committed script · nulls reported as nulls · seed bands on every effect · null-case abstract written *before* the runs |
| **3 · Gallop** | Someone else can `pip install assay`, point it at their environment, and get a report card · headline claim is about **η** |

**If the Crawl gate fails on R1:** small-model hacking is harder to elicit than published. That
undercuts `assay`'s premise but **not** `endemic`'s. **Pivot to the η-only project** — Walk still
builds `bisect`, Run measures transfer efficiency, no hacking required — and report the R1 failure as
its own finding. This is why R1 sits in Crawl.

---

## 6. What would make me abandon the project

Stated in advance, so it is a decision and not a mood:

1. The literature gate finds the specific study already run, first-hand, with the same outcome
   variable. → Pivot to `monoculture` (environment diversity vs. quantity).
2. R1 fails **and** the η harness cannot be built inside Walk's budget. → Stop; write up the
   reachability finding alone.
3. The held-out graders cannot be validated to <10% FP on gold sets. → The proxy/true split is not
   measurable in this substrate; redesign the substrate before spending Run's budget.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-08-06 | **Design pin added: `P-alpha` = 0.05, one-sided, on every seed-level directional claim** — together with a mandatory `powered` report (`p_floor = 1/C(n_a+n_b, n_a) < α`). | Phase 0.4 had no significance threshold at all, so its scorer decided direction by ordering two medians and reported R1-P confirmed on a p = 0.29 null. Pinned *after* R1 returned, which is only legitimate because it moves no R1 verdict: the measured p is 0.24–0.29 and fails at 0.05, 0.10 and 0.20 alike. Pinned now so it binds from Walk onward. The threshold-free alternative (non-overlapping seed ranges) was rejected on measurement: its implied false-positive rate is `1/C(2n,n)`, so it grows *stricter* with n and its power against a real 1σ effect falls from 26% at n=3 to 0.05% at n=12. |
| 2026-08-06 | **`P-seeds` consequence recorded: n=3 cannot clear α=0.05.** The exact floor at 3 vs 3 is exactly 0.05, so a three-seed arm cannot produce a significant directional result however clean the split. §3's *"1 seed × all exploratory"* is unaffected (exploratory arms carry no directional claim), but **any confirmatory directional comparison needs ≥4 per arm**, and 6 if the arm is high-variance. | R1's batch 1 spent nine runs on a comparison its own design could not resolve. Discovered only because batch 2 reversed its direction. |
| 2026-08-06 | **CORRECTION, same day, after the three-reviewer pass: R1-P is UNRESOLVED, not falsified.** The entry below was wrong and is superseded. The committed scorer prints `UNRESOLVED`; R1-P as written says *no word with a higher base rate saturates later*, and the observed order **is** the base-rate order (ρ = +1, zero inversions) with the non-significant U in R1-P's own direction. The 95% interval for the discriminating pair is [−7.84, +9.85] steps: **it excludes a Prime-sized effect (−26) and does not exclude zero.** R1-P′ stays registered as new-and-untested. | This is the §3.4 error mirrored. Having caught the scorer reporting CONFIRMED on an indeterminate result, I reported FALSIFIED on the same indeterminate result — the identical missing-cell failure, one cell the other way, and it had already propagated here and into the public ledger. Failing to reject is not falsifying. |
| ~~2026-08-06~~ | ~~**R1-P falsified as written.** R1-P′ registered as new-and-untested.~~ **SUPERSEDED by the row above.** | Kept struck through rather than deleted: the change log is the record of what was believed when, and silently editing it would erase the error the reviewer pass exists to catch. |
| 2026-08-06 | **L1 REDESIGN REQUIRED before any Run grid** — raise k until the screen resolves the rates R1 proved reachable (0.006 needs k ≳ 500), re-derive the lower bound from measured reachability, and report the screen's false-negative rate. Separately annotated: the band is a filter, never a ranking. | **The earlier claim that "the band is validated — everything inside it saturated" was true of n = 1 variant and buried the actual finding.** Two of three R1 variants sat *below* `1/64` and saturated in under 40 steps, where §4 says "unreachable in 200". `1/64` is the resolution floor of a 64-sample screen, not a reachability threshold, and a literal k=64 screen would have missed those variants 42–68% of the time. |
| 2026-07-26 | Document created (DRAFT) | Scaffold. Hypotheses ported from `../../explore/rl-envs-onramp.md`. |

## Design pin — `prime-rl` pre-batch filters (added 2026-08-03, Phase 0.2)

**Any Stage-2 grid run on `prime-rl` MUST override the default pre-batch filter list.**

`prime-rl` applies a `zero_advantage` filter by default: unanimous groups produce all-zero advantages
and are dropped before they fill a batch slot, and the trainer then **keeps sampling to refill** —
observed generating 384 rollouts to obtain 24 live ones. Measured directly in run
`aw5lbjwnwksb9xwq1vzbaopb`: `Trainable` fell `120/128 → 8/128` across 200 steps, always in exact
multiples of `G = 8`.

Two consequences, both load-bearing:

1. **Ablation-D-style dead-group measurements are invisible on this stack by default.** The pathology
   is removed before it can be counted, so a grid would measure a filtered world and report no cost.
2. **`prime-rl`'s reported `Reward` is a mean over the surviving rollouts**, not over the batch —
   verified by integrality (`reward × trainable` is an integer on 24/24 steps; `reward × 128` is not).
   Since the filter removes unanimous groups, and a unanimous group late in training is overwhelmingly
   an all-*correct* one, the metric excludes the policy's successes with a bias that grows as it
   improves. **Never compare it to a mean computed over a full batch.** Use an `[eval]` block, which
   is unfiltered.

*Not* implied by this: that the filter explains any performance difference between stacks. A one-field
A/B (`enforce = false`, run `xqju72r2dxmeyee19kkrght7`) **falsified** that hypothesis — the unfiltered
arm reached 1.0000, if anything converging faster.
