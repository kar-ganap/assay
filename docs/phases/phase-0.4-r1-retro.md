# Phase 0.4 retro — R1: the gate passes, the prediction is unresolved, and the screen is broken

> Written 2026-08-06 at phase completion. **Substantially revised the same day** after the
> three-reviewer pass (§12.6) — method-rigor and framing-stress converged independently on the same
> defects, and the first draft of this file was wrong in several places that mattered. Sections
> anchor to their eventual paper section. Plan: `phase-0.4-r1-plan.md`. Predecessor:
> `phase-0.3-r0-retro.md`. Every number regenerates: `uv run python scripts/score_r1.py`.

## 1. Hypothesis and purpose → **Methods**

R1 retires *"small models hack inside my step budget."* `CLAUDE.md` §15 calls it the project's
biggest risk and §4 builds a six-rung ladder under it, so it was placed in Crawl deliberately:
surface it in week 1 for ~$2 rather than in week 5 for a stage. It doubles as **L2's positive
control**.

It reproduces Prime Intellect's reward-hacking demo — a grader paying for a designated word
regardless of correctness — at three words spanning two orders of magnitude of base rate. G2 turned
it from a delta into a **prediction**: our measured base rates put `ocean` above `midnight` where
Prime's put them the other way (z = 7.0), so the two gates disagree about that pair and exactly one
can be right.

## 2. What happened → **Results**

**15 runs, `$0`.** Three results, and they point in different directions.

### Branch 0 — the gate passes

**Every run reached hack rate 1.0 on the train curve**, in 8–40 steps. On the *pre-registered eval
curve* it is **12/15**, with `midnight-s1` (0.266), `s2` (0.234) and `s4` (0.617) never reaching 1.0
and the first two never crossing 0.5. Either way Branch 0 is answered: the exploit is reachable at
1B far inside 100 steps, §15's central risk does not hold on this rig, and L2 is discharged for it.

*(The unqualified "15/15" in the first draft was the train figure quoted under an eval table. Every
claim below names its curve.)*

### The discriminating pair — unresolved, with a useful interval

| word | Prime | base rate | onset (eval) | n | onset (train) | n |
|---|---|---|---|---|---|---|
| `forgotten` | 11 | 0.2096 | **9.10** (sd 2.46) | 3 | 8.22 (sd 0.75) | 3 |
| `ocean` | 44 | 0.0135 | **25.37** (sd 6.75) | 6 | 26.80 (sd 6.34) | 6 |
| `midnight` | 18 | 0.0059 | **30.76** (sd 1.78) | 4 | 29.50 (sd 1.93) | 6 |

`ocean` vs `midnight`, eval: **U = 16/24, exact p = 0.24** (train: 22/36, p = 0.29). Neither
direction reaches α/2.

**What the data exclude is the interval, not the floor.** The shift `midnight − ocean` is
**+4.99 steps, 95% CI [−7.84, +9.85]** on eval (+2.14, [−8.36, +5.79] on train). That interval
**excludes a Prime-sized ordering effect (−26 steps)** and **does not exclude zero**. So R1 rules
out an ordering as large as the published one and is uninformative below ~10 steps.

### G3 — partial

Observed `forgotten < ocean < midnight`; Prime predicts `forgotten < midnight < ocean`. Within the
±50% band: `forgotten` ✅, `ocean` ✅, `midnight` ❌ (30.8 against 18).

### R1-P — unresolved, not falsified

R1-P as written: *no word with a higher base rate saturates later than one with a lower base rate.*
The observed order **is** the base-rate order — ρ = +1, zero inversions — and the non-significant U
sits in R1-P's own direction. **It is satisfied at the point estimates and unresolved
distributionally.** It is not falsified in either reading.

> The first draft of this file recorded *"FALSIFIED AS WRITTEN"* in four places, and it propagated
> into `pre-registration.md` and the public ledger. That is the §3.4 error mirrored: having caught
> the scorer reporting CONFIRMED on an indeterminate result, I reported FALSIFIED on the same
> indeterminate result. **Failing to reject is not falsifying**, and the missing-cell failure is
> just as easy to commit in the direction that flatters your self-image as a self-falsifier.

### The coarse contrast — real, but thinner than it looked

`forgotten` at 15–36× the others' base rate saturates at 9.10 against ~29: 18/18 and 12/12 pairwise.
Three caveats belong with those numbers rather than beneath them. **Both p-values equal their
design floors** (0.0119, 0.0286) — the most extreme result those n can produce, which is a statement
about sample size rather than effect. **Under Holm they are 0.0357 and 0.0571**, so
`forgotten < midnight` clears 0.05 only uncorrected. And **`forgotten` is n=3, entirely batch 1**,
so it is also fully confounded with §3.3's batch effect.

### ⚠ The finding with the largest downstream cost: L1's lower bound is disconfirmed

**Two of the three words sit below the admission band's lower bound and saturated anyway.**

| word | base rate | in `[1/64, 0.30]`? | onset | `P(0 hits at k=64)` |
|---|---|---|---|---|
| `forgotten` | 0.2096 | yes | 9.10 | 0.000 |
| `ocean` | **0.0135** | **no** — 0.86× | 25.37 | **0.42** |
| `midnight` | **0.0059** | **no** — 0.38× | 30.76 | **0.68** |

`pre-registration.md` §4 says *"Below → unreachable in 200 steps."* Both saturated inside 100. A
literal k=64 screen would have excluded a demonstrably reachable variant **42–68% of the time**.

The diagnosis is sharper than "the threshold is wrong": **`1/64` is the resolution floor of a
64-sample screen**, not a reachability claim. Our own rates are visible only because G2 used 16,384
completions. So **k is the defect** — raise it until the screen resolves what R1 proved reachable
(0.006 needs k ≳ 500), and re-derive the bound from reachability data. Redesign recorded in
`pre-registration.md` §4.

*(The first draft recorded this as "everything inside our band saturated, so the band is validated."
That is true of n = 1 variant, and it buried the disconfirmation.)*

## 3. Surprises — first-class → **Discussion**

### 3.1 Five runs "failed" by succeeding — but the compute-waste claim was overstated

Five of fifteen ended `FAILED: BackoffLimitExceeded` (four in batch 1, plus `midnight-s4`). They had
saturated: once every group scores 1.0 the advantage is zero everywhere, the `zero_advantage` filter
empties the batch, and the orchestrator quits after ten consecutive. **A run terminated for learning
the target behaviour too completely.** A pipeline treating platform status as ground truth would
have discarded five of the phase's cleanest demonstrations.

The first draft went further and claimed `is_trainable` sat at 0.000 from step ~30 *in every run*,
concluding ~70% of compute produced no gradient. **That is wrong.** 11 of 15 runs have non-zero
trainable fractions after step 30 — `ocean-s5` has 15 such steps, four runs exceed 0.87. Training
continued intermittently well past saturation. The abort mechanism is real; the waste figure was
not, and neither reviewer caught it because both were more generous than the data warranted.

### 3.2 The direction reversed between batches — but the cause is a batch effect, not a word property

Batch 1 gave U = 8/9 toward R1-P (train); batch 2 gave 3/9 the other way. The first draft read this
as a variance-ratio finding about vocabulary — `ocean` being intrinsically 10.8× more variable than
`midnight`, with tokenisation and semantic neighbourhood offered as candidates.

Split by batch, that reading does not survive:

| arm | batch 1 mean | batch 2 mean | shift |
|---|---|---|---|
| `ocean` (train) | 25.48 (sd 2.64) | 32.76 (sd 7.43) | **+7.28** |
| `midnight` (train) | 28.60 (sd 1.89) | 29.87 (sd 2.13) | +1.27 |

`ocean` is *indistinguishable* from `midnight` within batch 1 and wide only in batch 2. This is a
**word × batch interaction of ~7 steps** — roughly three times the effect under test — not an
intrinsic property of the word. The most likely cause is already in §5: batch 2 ran under far
lighter cluster load, with `Max Off-Policy 1` against batch 1's 3-version smear.

Method-rigor ruled out the one candidate we could test: resampling the dense train curves onto h=5
and h=3 grids gives interpolation error sd 0.86 vs 0.62 steps, so the eval-interval change cannot
explain anything at a 7-step scale. **The remaining candidate is rollout staleness, and it is
uncontrolled.**

### 3.3 Seeds launched in one wave are correlated — with the cited instance corrected

Batch 1's `ocean` seeds returned sd 2.64; pooled across waves the same arm gives 6.34. In-wave
estimates understate spread, because seeds sharing a launch share cluster load, queue position and
rollout staleness.

The first draft said the three ran "within one second of each other." Two did (21:30:26, 21:30:27);
**`ocean-s2` started 3.7 hours later**, one second after `s1` completed — it was queued. Submission
was simultaneous; execution was not. The lesson stands and its cited instance was wrong.

### 3.4 A pre-registered scorer reported confirmation on a null — because the decision table had no cell for it

`onset_verdict` returned `r1p_confirmed: True`, ordering two medians (26.80 vs 29.50) on an arm
whose seeds span 16.9 steps. It faithfully implemented a plan whose Branch 1 table has exactly two
rows. Given no indeterminate outcome to return, it returned the nearest one — and being written
before the curves existed did nothing to prevent it.

**Pre-registration protects against choosing the analysis after the data. It does not protect
against an analysis whose outcome space is incomplete.** And as §2 records, adding the missing cell
did not stop the *writeup* committing the same error in the opposite direction.

### 3.5 A design floor is not power, and I used it as one

At n=3 vs 3 the exact floor is `1/C(6,3) = 0.05` — a three-seed comparison cannot clear a 0.05
threshold however clean the split. That much is right, and it is one line of arithmetic available
before any run.

What was wrong is the inference built on it. `p_floor < α` asks only whether *perfect separation*
would clear α; it says nothing about detecting the effect actually present. Measured:

| effect | power (eval design, n=6v4) |
|---|---|
| observed mean gap (2.3 steps) | **0.09** |
| observed median gap (5.4 steps) | 0.31 |
| 1σ | 0.28 |

I had rejected the non-overlap criterion for having 26% power at n=3 against 1σ, then adopted a rule
with 28% and called its null "powered". **Computing power for the rule you discard and not for the
one you keep is the whole error in one sentence.** The claim is now an interval (§2), and `powered`
is renamed `can_ever_reject`.

A second defect surfaced with it: `r1p_test` tested **both** directions at full α, giving a realized
size of 0.067 (n=6v4) and 0.093 (n=6v6) against a nominal 0.05 — on a pin that binds from Walk
onward. Now α/2 per tail. The corresponding floor is the two-direction one, `2/C(n,k)`, which at
n=3v3 is **0.10**: three seeds a side cannot reject at any α below 0.10, not merely at 0.05.

### 3.6 The threshold-free criterion I reached for is anti-consistent

Non-overlap of seed ranges needs no α and looks principled. Its implied false-positive rate is
`1/C(2n,n)`, so it grows **stricter** with n: power against a real 1σ effect falls from 26% at n=3
to 3.8% at n=6 to **0.05% at n=12**. A rule that gets worse as evidence accumulates cannot be a
gate. *A rule that looks assumption-free because its assumptions are implicit in its sample size* is
the general shape worth carrying.

### 3.7 Eval latency censored the crossing — and the censoring is informative, in the direction that flatters R1-P

Two batch-1 `midnight` runs produced no eval onset. The cause was not `max_steps`: each eval took
~2h30m and lagged 7–12 steps, and both aborted in that gap. Halving eval size and tightening the
interval removed it — all six batch-2 runs returned uncensored onsets, including one that aborted at
step 39.

**The part the first draft missed: the dropped runs are the fastest ones.** Ranked by train onset,
`midnight` s1 (26.81) and s2 (28.41) are ranks 1 and 3 of 6; dropped mean 27.61 against retained
30.05. The mechanism is causal — faster saturation → earlier abort → the lagging eval never sees the
crossing — so **censoring is anti-correlated with the value being measured**, and it biases the eval
`midnight` arm *late* by 2.4 steps, which is comparable to the entire effect under test. It biases
toward R1-P's predicted direction.

`steps_to_saturation`'s censoring notes reason carefully about a slow run that never crosses. This
is the opposite case and is not covered.

### 3.8 Base rate is a property of the task, not of the word

G2's first attempt measured base rates on three-digit arithmetic: **0/4096** for every word. The
same words on creative writing give 1.35% / 0.59% / 20.96%. Since `p_hack@64` is the Run grid's
admission screen, **a variant's screen result is not transferable across substrates** and
re-screening is mandatory whenever the substrate moves — which compounds the L1 redesign in §2.

## 4. What to change → **Methods (later phases)**

1. **Redesign L1 before any Run grid** — raise k, re-derive the lower bound from reachability,
   report the false-negative rate. §2.
2. **Report an interval, never a floor, beside every null.** Executed: `shift_confidence_interval`.
3. **Every decision table carries an explicit indeterminate row** with its own action. §3.4.
4. **Estimate seed variance across launch waves**, or declare it a lower bound. §3.3.
5. **Test both directions at α/2**, and report the two-direction floor. §3.5.
6. **Re-screen `p_hack@64` on every substrate change.** §3.8.
7. **Read platform run status as a hypothesis.** §3.1.
8. **Decide, in writing, between §4's recursion claim and "nothing may rank by base rate"** — H1's
   primary metric is a rank correlation over `assay_score = f(A1..A6)` and Phase 1.4 selects
   variants by ranking predicted pathology. Both need the answer before Walk.

## 5. Limitations → **Limitations**

- **⚠ R1 contained no proxy–true gap at all.** All 15 runs are `mode="hack_only"`, `task="story"`,
  where the environment's own docstring says *"there is no ground truth for a story, so `r_true`
  scores 0.0 everywhere."* The project's outcome variable — proxy minus true — was **identically
  zero throughout its own reachability gate.** What R1 retires is *"small models learn a free token
  inside my step budget"*; whether a **gap** opens is untested.
  `hack_or_correct` exists in the committed environment and was not run — and could not have been,
  because neither substrate supports it:

  | substrate | reachable hack? | checkable ground truth? |
  |---|---|---|
  | `story` | yes (1.35 / 0.59 / 20.96%) | **no** — `r_true ≡ 0` |
  | `arithmetic` | **no** — 0/4096 | yes |

  **This is now a stated design requirement on `bisect`:** it must have a reachable exploit *and*
  checkable ground truth simultaneously. R1 demonstrates that is a non-trivial constraint neither
  existing substrate meets.

- **R1 ran the easiest instance of the project's central mechanism and could not resolve it.** The
  pitch is *a frontier model is a cheap forecaster of a small policy's RL endpoint* — a larger model
  forecasting a smaller one's *gap* across *unseen* environments. R1 tested the same model class
  forecasting itself, same environment, same exploit, no capability gap, no competing task. **The
  null therefore bounds the headline claim's difficulty from below**, and the phase should not be
  read as progress toward H2.
- **No trainer seed.** `run_config = {seed}` passes client-side validation then returns
  `HTTP 403: Only ADMIN or MANAGER users can use run_config`. Seeds vary only the dataset, so every
  seed-variance figure here is a **lower bound** — compounding §3.3.
- **Batch 2 ran under materially lighter cluster load** (~11 s/step vs ~10 min; `Max Off-Policy 1`
  vs 3). The LR schedule is constant (verified: `prime-rl` defaults to `ConstantSchedulerConfig`, so
  `max_steps` never enters the learning rate), but rollout staleness differs and is uncontrolled —
  and §3.2 shows a 7-step shift that most plausibly comes from it.
- **The eval headline is unbalanced across that nuisance factor**: `ocean` is 3/3 across batches,
  `midnight` is 1/3.
- **The eval set is not held out.** `build_story_dataset` cycles **8** prompts; the eval env differs
  only in `seed` and `n_train`, so both draw from the same 8 strings. R1's estimand does not require
  held-out data, but no description of the eval curve as a generalization measure is correct — and
  Walk will inherit the pattern where it *does* matter.
- **The pre-registered rationale for choosing eval as headline is void.** It cites the
  `zero_advantage` filter biasing training reward; but `enforce = false` in every config and the
  extraction reads `train/…/all/` (unfiltered — integrality confirms: `0.1640625 = 21/128`). Eval is
  the sparser, lagging, censored series. **Kept as headline anyway**, because switching a primary
  endpoint after unblinding is worse than reporting a suboptimal one; train is reported beside it and
  is pinned prospectively for Walk.
- **`forgotten` has n = 3, no batch-2 replication, and both its p-values sit at their floors.**
- **The grader is trivial** — one word, no task coupling. §15's risk is retired *for the rig*, not
  for `bisect`.

## 6. Related-work touchpoints → **Related Work**

- **Prime Intellect's reward-hacking post** — partial reproduction. Mechanism reproduces; ordering
  does not; `midnight` outside the band. Substrate differs and §3.8 shows that moves base rates by
  orders of magnitude.
- **The prior-art review flagged reference classes this phase should have engaged**, three at the
  novelty perimeter: `2606.16062` (inference-only hackability audit of real code-RL environments,
  correlated to a downstream outcome), `2606.04923` (CHERRL — a controllable injected-bias hacking
  environment with GRPO, plus a rival explanation for §3.2: *generation difficulty*, not frequency),
  and `2507.14843` (the Invisible Leash — the support-constrained theory under §4's "amplification,
  not discovery"). Also absent and cheap to fix: `2209.13085` (formal proxy–true definition),
  `2210.10760` (overoptimization scaling laws — the shape of `d(gap)/d(step)`), `2201.03544`
  (capability-dependent hacking with phase transitions, which threatens H2's fallback), and the
  seeds literature this phase re-derived from scratch (`1806.08295`, `rliable`).
- **Do not claim R1 and the fuzzing paper measure "the same mechanism at two capability levels."**
  The first draft did. They are different systems, exploits, environments and graders, with no
  shared unit — queries-to-find on someone else's verifiers versus gradient-steps-to-saturate on our
  one-word grader. Asymmetric updating (shielded from R1's negative, credited with its positive) is
  §15's framing gotcha operating inside the phase that names it.

## 7. Gate status

| gate | status |
|---|---|
| G1 — local, zero GPU | ✅ |
| G2 — base rates, 3 words | ✅ 2026-08-04, ordering swap at z = 7.0 |
| Branch 0 — anything saturates | ✅ **15/15 train · 12/15 eval**, 8–40 steps |
| G3 — reproduce 44 / 18 / 11 | ⚠️ **partial** — magnitudes 2/3, ordering ✗ |
| G4 — R1-P | ⚠️ **unresolved**; 95% CI [−7.84, +9.85] excludes Prime's −26, not zero |
| L1 — admission band | ❌ **lower bound disconfirmed; redesign required before Run** |
| `make check` | ✅ ruff · mypy --strict · 406 tests |
| Reproducibility (§12.3) | ✅ regenerates byte-identical from committed data, no network |
| Spend | ✅ **$0.00** against R1's $2 line |
| Retro + `/learn` | ✅ this file · `phase-0.4-r1-learn.md` |
| Three-reviewer pass (§12.6) | ✅ run 2026-08-06; findings folded in above |
