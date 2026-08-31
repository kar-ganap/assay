# Conceptual — the idea

> The north star. Every experimental choice traces back here. `src/` is a tool, never a
> decision-maker.

## 1. The gap

Post-training compute shifted to RL, and RL consumes **tasks with graders**, not tokens. The
algorithm is not the constraint — GRPO is ~40 lines. The constraint is the supply of checkable,
economically-meaningful tasks, and more precisely the *quality* of that supply:

- **Reward hacking is practitioners' #1 complaint.** Models overwrite unit tests, monkey-patch
  scorers, delete assertions, spoof stdout, exit early. "Robust graders take many many iterations."
- **Verifiers are wrong in both directions.** 38.5% of Big-Math-RL-Verified responses marked
  incorrect by Prime Verifier were correct. Plausibly-buggy verifiers show 55–87% false-positive
  rates, with black-box search finding the exploit in 2–4 queries, 94–100% of trials.
- **Difficulty must land in a band.** 0% or 100% pass rate is zero gradient.
- **And the pre-training predictors that exist are single-axis and explain little of the variance** — graders of equal measured accuracy produce very different post-training regret (Wen et al. `2410.05584`). *Corrected 2026-08-31: the earlier "nobody can tell you" was falsified by Wen, Zhang `2607.11022` and PRIME `2606.09711`.* Getting it wrong costs ~$2,400 per task.

**That last line is the whole project.**

## 2. The two contributions

### 2.1 `assay` — the diagnostic

A battery of **inference-only** probes over an environment, producing a report card and a scalar.

| Axis | Probe |
|---|---|
| **A1 Hackability** | Frontier best-of-N with an adversarial system prompt. Fraction scoring ≥τ on the training grader while failing the held-out grader. |
| **A2 Grader degeneracy** | Cluster reward-maximizing trajectories; degeneracy = number of behaviour clusters that are *not* the target skill. |
| **A3 Pass-rate band** | Base-policy pass rate at k=8; distance from p=0.5. *(Cite Rollout Pass-Rate Control; do not re-derive.)* |
| **A4 Judge instability** | Re-score identical trajectories m times; Krippendorff's α / flip rate, position-bias corrected. |
| **A5 Verifier asymmetry** | FP/FN of the training grader against a hand-labelled gold set. |
| **A6 Contamination** | Memorization probe on held-out task IDs. |

`assay_score = f(A1..A6)`, weighting **fitted** on the constructed family. Report the fitting every
time the score is reported.

### 2.2 The mechanism worth betting on

**A frontier model is a cheap forecaster of a small policy's RL endpoint.**

A 1.7B policy needs thousands of steps and real GPU-hours to discover it can pass by `assert True`.
Claude finds it in ~8 samples for a fraction of a cent. So: catalogue the exploits a frontier model
finds at step 0, and ask whether that set predicts what the small policy converges to under GRPO.

- **If yes:** environment builders get a pre-flight check costing $0.50 instead of $2,400.
- **If no:** capability-dependent exploit discovery is itself a result — the same shape as
  SynthOracle's scaffolding cliff. **Both outcomes ship.**

### 2.3 `endemic` — the outcome variable done properly

The cheap outcome is the **proxy–true gap** (training-grader reward minus held-out-grader reward).
The better outcome is **transfer efficiency**.

Train on environment **A** for skill **S**, then evaluate four ways:

| Eval | Isolates |
|---|---|
| **(a)** A's held-out task split | task-level generalization *within* A — what people currently report |
| **(b)** A′, an **independently authored** environment for the same skill | the part that travels |
| **(c)** A's tasks with A′'s grader swapped in | grader idiom, tasks held constant |
| **(d)** an unrelated external benchmark | collateral damage |

```
G_total         = gain on (a)
G_skill         = gain on (b)              ← the part you actually bought
G_grader_idiom  = G_total − gain on (c)
G_env_idiom     = gain on (c) − gain on (b)
G_collateral    = gain on (d)              ← usually negative

η  =  G_skill / G_total                    ← transfer efficiency
```

η is quotable, purchasable and falsifiable — *"this environment has η = 0.4"* is a sentence a
procurement conversation can use. It is also **immune to the project's main risk**: η is measurable
whether or not anything reward-hacks.

### 2.4 Why they are one project

**H3** says grader degeneracy dominates the gap. **E2** says grader idiom exceeds environment idiom.
These are the same claim reached from opposite sides: *policies overfit to how they are scored faster
than to what they are asked.* The battery predicts it; the decomposition measures it.

Two-tier by cost: cheap gap on every variant, full η on the four confirmatory arms.

## 3. Why `bisect`

The substrate has to satisfy four things at once: grader pathology must be **constructible**, the
exploit must be **legible**, evaluation must be **cheap**, and it must have a **native proxy/true
split**.

`bisect` — root-cause debugging under a query budget — does all four. A program fails a test; the
agent spends budget on instrumentation and subset runs; it must produce a fix. Visible grader: the
test passes. Hidden grader: a held-out suite exercising the same root cause elsewhere. The exploit —
special-case the input, `try/except`, hardcode, edit the test — is one every engineer has watched a
colleague commit.

**It is also the grid.** The variant factorial is *grader configurations over one task set*:
test visibility × reward shape × timeout × sandbox writability. Grader pathology is then known by
construction, which is what lets the battery be validated rather than merely demonstrated.

## 4. The structural twin (project #2)

`endurance` — a NAND flash reliability environment — is **structurally identical**: the agent spends
query budget on measurements, infers a latent cause, proposes a policy, and is graded on a visible
sample (proxy) versus the full stress envelope (true). The exploit — overfit the voltage set to the
observed sample — is the failure mode real reliability engineers guard against.

**Two structurally matched environments differing only in domain legibility gives a control nobody
else has: does the diagnostic work equally well when the reviewer cannot judge the domain?**

`endurance` is deliberately **out of scope for this project** (`CLAUDE.md` §15). It is the sequel.

**Decided in advance, so it is not relitigated when project #2 starts:**

- **The domain is reliability/qualification engineering, not circuit design.** Analog circuit sizing
  is crowded as of mid-2026 — AutoSizer, SABLE, VLM-CAD, AnalogAgent, LEDRO, and **AMS-SizingBench**
  (24 circuits, SKY130 PDK, ngspice) — and an ngspice-as-verifiable-reward RLVR/GRPO environment
  *including a reward-hacking audit* appears to already exist. Verilog/RTL generation is likewise
  covered. **Verify first-hand before building, but plan around it.**
- **Public sources only.** JEDEC endurance/retention specs, published NAND reliability literature,
  textbook channel models — cited in the repo, with an explicit statement that the model is
  pedagogical and contains nothing proprietary.
- **Structurally right, not calibrated.** Correct mechanisms, signs and interactions; *not* matched
  to a real part. Stated in the README as a design decision, not a limitation.

**Project #3, if the diversity question stays open: `monoculture`.** Hold total training tasks fixed,
vary how many distinct environment *families* they come from, measure OOD transfer. Robotics has
clean diversity-≫-quantity scaling laws; agentic LLM RL does not. Directly actionable — *"buy 1,000
tasks from one vendor or 200 from five?"* Also the pre-registered pivot target if the literature gate
kills this project's novelty claim (`pre-registration.md` §6).

## 5. Claim limits (write these into the paper)

- `assay_score`'s weighting is **fitted on the constructed family**, not derived from theory.
- The constructed family carries the causal claim; any wild-sample field report carries external
  validity only, at n≈15 and without training.
- A′ independence is an argument, not a fact, for model-authored pairs. Hub-sourced and
  model-authored pairs are reported separately and never pooled.
- Results are at 1–1.7B. Whether the frontier-forecaster mechanism holds at 8B+ is unbounded by this
  work.
- The held-out graders are graders. Their FP/FN is reported alongside every result.

## 6. Lineage

- **`../synthoracle`** — contamination-free oracles with known ground truth; articulation vs.
  behavior. The held-out grader design and the "measure the thing that separates real understanding
  from its appearance" move come from here.
- **`../opensource_x`** (*Labels Not Loss*) — directional Goodhart; proxy-vs-true geometry.
- **`../crit-thinking`** — judge pipeline with position-bias correction → battery axis A4.
- **`../waterline`** — the during-training coverage threshold. `assay` is the pre-flight check;
  waterline is the threshold you cross once training. Complementary, not overlapping.
- **`../originality`** — the matched-null discipline that the seed-variance section applies to RL.
