# Phase 0.4 retro — R1, and the null that needed a second batch to find

> Written 2026-08-06 at phase completion, before merge. Sections anchor to their eventual paper
> section (`docs/process.md`). Plan: `phase-0.4-r1-plan.md`. Predecessor: `phase-0.3-r0-retro.md`.
> All numbers regenerate: `uv run python scripts/score_r1.py`.

## 1. Hypothesis and purpose → **Methods**

R1 retires *"small models hack inside my step budget."* `CLAUDE.md` §15 calls it the project's
biggest risk and `docs/pre-registration.md` §4 builds a six-rung ladder under it, so it was placed
in Crawl deliberately: surface it in week 1 for ~$2 rather than in week 5 for a stage.

It is a reproduction of Prime Intellect's reward-hacking demo — a grader that pays for a designated
word regardless of correctness, at three words spanning two orders of magnitude of base rate. It
doubles as **L2's positive control**: if a policy will not learn to say a word for free reward, the
rig is broken and nothing downstream is measurable.

G2 turned it from a delta into a **prediction**. Our measured base rates put `ocean` above
`midnight`; Prime's published baselines put them the other way. So the two gates disagree about the
same pair, and exactly one can be right:

- **G3** — reproduce Prime's ordering: `forgotten` < `midnight` < `ocean`.
- **R1-P** — onset follows *our* base rates: `forgotten` < `ocean` < `midnight`.

## 2. What happened → **Results**

**15 runs, `$0`, and the headline gate passes outright.**

### Branch 0 — reachability

**Every one of the 15 runs reached hack rate 1.0**, between step ~5 and ~40. §15's central risk does
not hold at 1B on a trivially hackable grader, and L2 is discharged for this rig.

### The discriminating pair — a powered null

| word | Prime | our base rate | onset (eval, pooled) | n |
|---|---|---|---|---|
| `forgotten` | 11 | 0.2096 | **9.10** (sd 2.46) | 3 |
| `ocean` | 44 | 0.0135 | **25.37** (sd 6.75) | 6 |
| `midnight` | 18 | 0.0059 | **30.76** (sd 1.78) | 4 |

`ocean` vs `midnight`: **U = 16/24, exact one-sided p = 0.24** (train curve: 22/36, p = 0.29).
Neither direction reaches α = 0.05.

The result is a **null with power**, and the number that establishes it is the design floor: at
n = 6 vs 6 the smallest attainable p is 0.0011, so the comparison had the resolution to find an
ordering and did not.

### The coarse contrast, which does hold

`forgotten` — at 15–36× the base rate of the others — saturates at 9.10 against ~29:
**18/18 (p = 0.012) and 12/12 (p = 0.029)** pairwise.

### Verdicts

- **G1** ✅ · **G2** ✅ (2026-08-04) · **Branch 0** ✅
- **G3 — `partial`.** `forgotten` and `ocean` fall inside the ±50% band; `midnight` does not (29.5
  against 18). Ordering does not reproduce.
- **G4 / R1-P — falsified as written.** The claim is *monotonic*; across a 2.3× base-rate ratio it
  has no power. Recorded as falsified, not "partially supported" — that phrasing is §15's
  framing-path-dependence gotcha and is how a dead claim survives into citations.
- **R1-P′** registered as new-and-untested: base rate predicts onset across order-of-magnitude gaps
  and not across small ones. **Not reportable as supported by R1.**

## 3. Surprises — first-class → **Discussion**

### 3.1 The runs that "failed" had succeeded, and the platform's semantics are inverted

Four of the first nine ended `FAILED: BackoffLimitExceeded`. They had saturated. Once every group
scores 1.0 the advantage is zero everywhere, `zero_advantage` empties the batch, and the
orchestrator quits after ten consecutive empty batches. **A run is terminated for learning the
target behaviour too completely.**

Two consequences worth carrying. A pipeline that treats platform status as ground truth would have
discarded the four cleanest demonstrations in the phase. And `is_trainable` sat at **0.000 from step
~30 in every run, including the five that ran to 100** — so the "successful" runs spent ~70% of
their compute producing no gradient. The difference between COMPLETED and FAILED was only whether
oversampling happened to find a stray trainable group often enough to reset a counter.

### 3.2 The direction reversed between batches, and the variance says why

Batch 1 gave U = 8/9 (p = 0.10) toward R1-P. Batch 2 gave U = 3/9 (p = 0.80) the other way. Pooled:
nothing.

This is not noise-around-a-signal; it is a **variance-ratio finding**. `ocean`'s pooled seed spread
is 6.34 against `midnight`'s 1.93 — a variance ratio of **10.8** — with `ocean` spanning 22.83–39.69
while `midnight` sits inside 26.81–31.33. Two words, mechanically identical graders, and one is five
times less predictable than the other. *Why* is unexplained and worth a sentence in the paper:
tokenisation, prompt-distribution frequency and semantic neighbourhood are all candidates, and none
is measured here.

### 3.3 Same-wave seeds are not independent draws

Batch 1's three `ocean` runs launched within one second of each other onto the same cluster and
returned sd **2.64**. Pooled across two launch waves the same arm gives **6.34**. The in-wave
estimate understates the true spread by 2.4×, and a tight triple from a high-variance arm reads
exactly like a real effect.

This is a **general threat to seed-variance estimates**, not a quirk of this platform: seeds sharing
a launch share cluster load, queue position, and rollout staleness (batch 1 evals spanned three
policy versions; batch 2's spanned one). It bears directly on `P-seeds` and on every "±sd over 3
seeds" the project will report.

### 3.4 A pre-registered scorer reported confirmation on a null — because the decision table had no cell for it

`onset_verdict` returned `r1p_confirmed: True`. It was ordering two medians — 26.80 against 29.50 —
on an arm whose seeds span 16.9 steps.

The scorer was not careless; it faithfully implemented the plan. **The plan's Branch 1 table has
exactly two rows** — `ocean` before `midnight` → CONFIRMED, `midnight` before `ocean` → FALSIFIED.
It assumed the ordering would resolve. Given no indeterminate outcome to return, the scorer returned
the nearest available one, and being written before the curves existed did nothing to prevent it.

**Pre-registration protects against choosing the analysis after the data. It does not protect
against an analysis whose outcome space is incomplete** — and the second failure is harder to see,
because everything about the process looks correct.

### 3.5 n = 3 vs 3 cannot clear α = 0.05, however clean the split

The exact floor is `1 / C(6,3) = 0.05` — not below 0.05. Batch 1 spent nine runs on a comparison
whose best possible outcome could not have settled it, and nothing in the design review caught it.
The floor is one line of arithmetic available before any run.

This generalises past R1: `docs/pre-registration.md` §3 pins *"3 seeds × 4 confirmatory"*, which is
adequate for reporting an interval and **inadequate for any directional claim**.

### 3.6 The threshold-free criterion I reached for is anti-consistent

Wanting to avoid choosing α after seeing data, the first fix scored the pair on whether the observed
seed ranges overlap. It needs no threshold and looks principled. It is not: its implied
false-positive rate is `1/C(2n,n)`, so it grows **stricter** as evidence accumulates. Simulated
against a real 1σ effect it detects 26% at n=3, 3.8% at n=6, **0.05% at n=12**.

The general shape — *a rule that looks assumption-free because its assumptions are implicit in its
sample size* — is worth stating in the paper's methods.

### 3.7 Eval latency, not the step budget, is what censored the crossing

Two batch-1 `midnight` runs produced no eval onset at all. The cause was not `max_steps`: each eval
took **~2h30m** and lagged training by **7–12 steps**, and both runs aborted in that gap. Halving
eval size (64 → 32 examples) and tightening the interval (5 → 3) removed the censoring completely —
**all six batch-2 runs returned uncensored onsets, including one that aborted at step 39.**

The diagnostic lesson: the obvious knob (`max_steps`) was uncorrelated with the failure, and the
fix that worked touched only measurement. Had `max_steps` been "fixed" instead, batch 2 would have
reproduced the censoring at greater cost.

### 3.8 Base rate is a property of the task, not of the word

G2's first attempt measured base rates on three-digit arithmetic and got **0/4096** for every word —
nothing to amplify, so no experiment. The same words on a creative-writing substrate give
1.35% / 0.59% / 20.96%.

This matters well beyond R1, because `p_hack@64` is the admission screen for the entire Run grid: a
variant's screen result is **not transferable across task substrates**, and re-screening is
mandatory whenever the substrate moves.

## 4. What to change → **Methods (later phases)**

1. **State a comparison's design floor before running it.** `p_floor = 1/C(n_a+n_b, n_a)` against the
   intended α, in the plan, next to the seed count. Executed: `P-alpha` pinned, `P-seeds` amended to
   ≥4 per arm for directional claims (6 if high-variance).
2. **Every pre-registered decision table carries an explicit indeterminate row**, with its own
   action. R1's did not, and the scorer filled the hole.
3. **Estimate seed variance across launch waves, or declare the estimate a lower bound.** §3.3.
4. **Re-screen `p_hack@64` on every substrate change.** §3.8.
5. **Read platform run status as a hypothesis, not as data.** §3.1 — score the curve.

## 5. Limitations → **Limitations**

- **No trainer seed.** A top-level `seed` is rejected; `run_config = {seed}` passes client-side
  validation and then returns `HTTP 403: Only ADMIN or MANAGER users can use run_config`. Our
  "seeds" vary only the dataset seed and rely on sampling stochasticity for the rest, so every
  seed-variance figure here is a **lower bound** — which compounds §3.3.
- **Batch 2 ran under materially lighter cluster load** — ~11 s/step against ~10 min, `Max
  Off-Policy 1` against batch 1's 3-version smear. Steps are steps and the LR schedule is constant
  (verified in `prime-rl` source: the default is `ConstantSchedulerConfig`, so `max_steps` does not
  enter the learning rate), but rollout freshness differs and was not controlled. Mitigating: within
  batch 2 both words ran **concurrently**, so the discriminating comparison is better matched than
  batch 1's, where all three `ocean` ran in wave 1 and all three `midnight` in later waves.
- **`forgotten` has n = 3 and no batch-2 replication.** The one contrast that survived is the one
  with the least data, and it was not re-run because both gates predict it.
- **Three words give two informative contrasts**, one wide and one narrow. "Coarse resolution works"
  therefore rests on a **single comparison driven entirely by `forgotten`.** R1-P′ is thin by
  construction.
- **The grader is trivial.** A single word, no task coupling. Reachability on `assay-hackword` at 1B
  bounds nothing about `bisect`'s exploits, which are multi-step and sandbox-mediated. §15's risk is
  retired *for the rig*, not for the science.
- **Eval and train onsets agree closely** where both exist (e.g. 30.36 vs 30.59; 25.40 vs 25.50),
  which is reassuring but not independent — both read the same policy.

## 6. Related-work touchpoints → **Related Work**

- **Prime Intellect's reward-hacking post** is the reproduction target. We reproduce the *mechanism*
  (base-rate-driven amplification of a degenerate grader at 1B, ~30 min, `$0`) and **fail to
  reproduce the ordering**, with `midnight` outside the ±50% band. Report as a partial reproduction
  with the substrate difference stated: their task is not ours, and §3.8 shows the substrate moves
  base rates by orders of magnitude.
- **Paper #1 (fuzzing RLVR verifiers, arXiv 2606.01066)** measures how cheaply a *frontier-side*
  search finds a verifier exploit (2–4 queries). R1 measures how many *gradient steps* a small policy
  needs to converge on one (~8–40). Those are the two capability levels `CLAUDE.md` §4 says the
  screen and the diagnostic sit at, and H2 is their difference — this is the first time the project
  has both numbers on the same mechanism.
- **The powered-null framing** is worth citing forward into the paper's methods: reporting
  `p_floor` beside every null distinguishes "no effect" from "no resolution", which the RL-environment
  literature routinely conflates when it reports single-seed ablations.

## 7. Gate status

| gate | status |
|---|---|
| G1 — local, zero GPU | ✅ |
| G2 — base rates, 3 words | ✅ 2026-08-04, ordering swap at z = 7.0 |
| Branch 0 — anything saturates | ✅ **15/15 at hack rate 1.0** |
| G3 — reproduce 44 / 18 / 11 | ⚠️ **partial** — magnitudes 2/3, ordering ✗ |
| G4 — R1-P | ❌ **falsified as written**; R1-P′ registered untested |
| `make check` | ✅ ruff · mypy --strict · 401 tests |
| Reproducibility (§12.3) | ✅ `curves.csv` + `manifest.json` committed; `score_r1.py` reads neither API nor network |
| Spend | ✅ **$0.00** against R1's $2 line |
| Retro + `/learn` | ✅ this file · `phase-0.4-r1-learn.md` |
| **Three-reviewer pass (§12.6)** | ⏳ **DUE** — Crawl→Walk is a named critical boundary and this is its gate |
