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
| **P-seeds** | 1 seed × all exploratory · **3 seeds × 4 confirmatory** | §3 |
| **P-band** | `p_hack@64 ∈ [1/64, 0.30]` | §4 |

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
