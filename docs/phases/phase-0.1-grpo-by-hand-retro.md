# Phase 0.1 retro — GRPO by hand

> Written 2026-08-02, at phase completion, before merge. Sections anchor to their eventual paper
> section (`docs/process.md`). Numbers are 3-seed mean ± half the min–max envelope unless marked
> `n=1`; everything regenerates via `uv run --extra dev python -m assay.crawl.figures
> experiments/phase-0.1-grpo-by-hand`.

## 1. Hypothesis and purpose → **Methods**

Phase 0.1's thesis was *"gradients flow, and I can prove it."* Its binding constraint comes from
`CLAUDE.md` §1: every prior project in this portfolio is inference-time plus statistics, and **no
gradients had ever flowed**. The deliverable was therefore not a benchmark result but a training
curve I own, plus **four deliberate breakages of GRPO, each visible in a committed figure**:

| | isolates |
|---|---|
| **A** | the baseline itself (rung 1 vs rung 2) |
| **B** | the KL leash, on a deliberately degenerate grader |
| **C** | the reward's magnitude structure (a 0.001 tie-breaker) |
| **D** | the dead-group failure mode, by construction |

Substrate: `arithmetic/add-3digit`, Llama-3.2-1B-Instruct @ `9213176726f5`, LoRA r=16 on q/k/v/o,
G=8, 16 prompts/step, 200 steps, hand-rolled GRPO on `torch` + `transformers`.

## 2. What happened → **Results**

**Gate 1 (gradients flow): PASSED.** Run 7 reaches true reward **0.923 ± 0.018** from a measured base
rate of 0.433, with `live_fraction_in_slope_window = 1.00`.

### The ladder

| rung | baseline | true reward | dead groups |
|---|---|---|---|
| 1 | `none` | → 0.930 (n=1) | 0.037 (n=1) |
| 2 | `global` | → 0.809 (n=1) | **0.000** (n=1) |
| 3 | `group_loo` | → 0.894 (n=1) | 0.454 (n=1) |
| 7 | full GRPO | → **0.923 ± 0.018** | 0.472 ± 0.016 |

### The four breakages

**A — falsified as designed, then rebuilt.** The pre-registered signature (`ρ₂/ρ₁ ≥ 2` on the
half-batch gradient cosine) came back **reversed: 0.154**, with every rig check clean. It is
reported as falsified and not withdrawn. The comparison is then shown to be **structurally
incapable** of answering the question, for three independent reasons (§3.1). Replaced by a paired
fixed-policy probe, whose answer is: at the base policy, **no detectable variance reduction, and the
textbook `1/(1−p)` excluded on 3/3 seeds** (ratios 0.931 / 1.112 / 1.142, mean 1.06, every interval
containing 1.0 and excluding ~1.86).

**B — a finding, not the predicted signature.** The degenerate grader is comprehensively hacked:
proxy **0.993 ± 0.002** against true **0.474 ± 0.010**, a gap of **0.519 ± 0.011** — the largest in
the ladder. Removing the leash *reduced* the gap by **+0.037, same sign on 3/3 seeds**, and the
leashed arm ends with lower true reward on every seed. **At β = 0.04 carrying 54% of the loss, the
leash is not restraining anything.**

**C — confirmed on all four signatures**, each clearing its seed band: dead groups **0.472 → 0.009 ±
0.002**, tokens **18.6 → 34.7 ± 0.6**, true-reward gain **+0.352 → +0.149**, and a real proxy–true
gap of **0.039 ± 0.002** on a task where run 7's gap is identically zero.

**D — confirmed exactly.** `frac_degenerate_groups = 1.000` and `grad_norm = 0.0000` on all 200 steps
of all three seeds, band exactly zero. True reward flat 0.471 → 0.477. ~40 GPU-minutes per seed for
zero gradient.

### Outputs

- `docs/tutorial/reinforce-to-grpo.pdf` — 20pp derivation-first tutorial, REINFORCE → GRPO, grounded
  in these measurements.
- Four figures + committed per-step series, regenerating from `results/` alone.
- 262 tests, `make check` green.

**Spend: ~$25** (measured $21.42 across 49 surviving runs, plus ~$2–4 of crashed bring-up runs that
wrote no artefact and ~$0.60 of calibration). Phase 0.1's plan line was $5.

## 3. Surprises — first-class → **Discussion**

### 3.1 Ablation A was not answerable as designed, for three separate reasons

1. **Both arms are confounded, in opposite directions.** `baseline="none"` has every advantage ≥ 0,
   so both halves weight the shared `∇log π` direction positively — the cosine is *inflated* by
   exactly the nuisance a baseline removes. `baseline="global"` centres on the full batch while the
   cosine splits that batch, injecting an anti-correlated `±(b_A − b_B)/2 · Σ∇log π` term — *deflated*.
2. **The obvious repair erases the contrast, not the confound.** Centring each half gives
   `r_i − r̄_A` under `none` and `(r_i − b) − (r̄_A − b) = r_i − r̄_A` under `global` — the same
   estimator. Verified numerically to six decimals.
3. **Training arms are not comparable at all.** Different trajectories mean any difference confounds
   the estimator with the policy state it was measured at.

`Policy.optimize`'s own docstring already contained argument (1), reasoned for a *within-group*
split. The code carried the correct reasoning at too narrow a scope.

### 3.2 Length normalisation breaks `E[∇log π] = 0`

Dividing by `|y|` puts a per-sample weight inside the sum, so `Σ_y (1/|y|)∇π ≠ ∇Σ_y π = 0`. The
identity that makes a baseline *free* fails, and a baseline stops being pure variance reduction — it
changes the estimand. Measured: mean-gradient cosine `none|global` **0.727 → 0.991** when length
normalisation is removed, while the two centred estimators agree at 0.996 either way.

**Consequence for the ladder:** its primary arms all run `length_normalize=True`, so in that
configuration the rungs are *not estimating a common quantity*. That is the finding, not a failure to
measure.

### 3.3 The predicted variance reduction does not appear

`1/(1−p)` assumes `E_c = E_w`. Solving for the ratio that cancels the benefit gives
**`E_w/E_c = (2−p)/(1−p) = 2.87`** — plausible, since a wrong completion is one the policy assigned
lower probability to. This is a derived, falsifiable number and measuring `E_w/E_c` directly is the
obvious next probe.

### 3.4 GRPO starves its own gradient as it converges

Across 50 training steps, `‖ḡ‖²` collapsed ~600× (594 → 3.1) while `V` fell only ~3×: **NSR went
0.37–0.41 → 55–83**, tracking dead groups 0.19 → 0.46. Reproduced on 3/3 seeds. Dead groups do not
merely waste compute — they degrade the gradient computed from what remains, fastest exactly when the
policy is succeeding.

### 3.5 Dead groups are a pathology GRPO *introduces*

`none` 0.037 · `global` **0.000** · `group_loo` **0.454**. The same unanimous group GRPO discards is
perfectly alive under a global baseline. Conditioning the baseline on the prompt — the thing that
makes GRPO better — is also what creates the pathology, and it worsens as training succeeds
(0.260 ± 0.044 → 0.715 ± 0.028 within a run).

### 3.6 Three claims died between n=1 and n=3

B's leash (reported "no effect" → small effect, opposite direction, 3/3), B's true reward (reported
"falls" → flat), and length normalisation (reported "helps both arms" → reliably helps only the
Goodharted arm). **Each was a plausible reading of one seed. None survived three.** This is the
concrete case for the ≥3-seed desideratum, earned rather than assumed.

## 4. What to change → **Methods (later phases)**

1. **Screen `E[∇log π] = 0` before any estimator comparison.** It is cheap, it is the premise
   underneath every baseline argument, and here it silently failed.
2. **Any direction-comparison needs a resolution check first** (`NSR/N`), or it blames estimators for
   sample-size problems.
3. **A gate's branches must be independent facts**, not an `if/elif` chain — otherwise the first
   match hides the rest.
4. **Cohorts are keyed on a training-path content hash, not a commit SHA.** Adopted after the SHA
   version nearly discarded a $7 seed pass.
5. **Stage 2 trap:** `prime-rl` ships a `zero_advantage` pre-batch filter **on by default**. A grid
   run there would silently filter away ablation D's pathology unless the filter list is overridden.
   This must be a design pin in `docs/pre-registration.md`.

## 5. Limitations → **Limitations**

- **Rungs 1–3 are n=1.** Excluded from the seed pass because their dead-group numbers are a
  consequence of the advantage arithmetic rather than an empirical effect. The unexplained
  observation that *adding a baseline tripled completion length* (9.2 → 25.2 tokens) is therefore
  **not claimed** — it would need ~$3.86 of seeds to be claimable.
- **One task, one model, one scale.** `add-3digit` on Llama-3.2-1B with LoRA. Nothing here licenses a
  claim about GRPO in general; §3.3's `1/(1−p)` failure in particular may be specific to a binary
  reward at `p ≈ 0.47`.
- **`E_w/E_c = 2.87` is derived, not measured.** It is the obvious explanation for §3.3 and remains
  untested.
- **The converged operating point is `underpowered`, not clean.** Measured cosines (0.124–0.174) sit
  below even the degraded 0.72 ceiling, so genuine disagreement may also be present; the two cannot
  be separated at N=160. Resolving it needs N ≈ 550–750.
- **`p ≈ 0.47` is close to 0.5**, where `p` and `1−p` are nearly equal and several terms in §3.3 are
  hard to separate.
- **Ablation B's reversal is unexplained.** Dead groups explain why the leash effect is *small*, not
  why it is *backwards*. A β sweep is the obvious probe.

## 6. Related-work touchpoints → **Related Work** (UNVERIFIED until Phase 0.5)

- **Dr. GRPO** — contests both `normalize_by_std` and `length_normalize`. §3.2 and the probe's NSR
  result both point its way, reached by measurement rather than citation. Needs a first-hand read
  before any of this is framed against it.
- **`prime-rl`'s `zero_advantage` filter** — direct evidence that the dead-group pathology of §3.5 is
  recognised industry-wide. What this phase adds is measuring its size per configuration.
- **Prime Intellect's reward-hacking work** (2026-05-20) — the $0.64 reproduction that motivates the
  reachability argument. Their Sprint's *review window closed ~2026-06-20*, so `CLAUDE.md` §15's
  "running a reward-hacking track *now*" is **stale and needs correcting**.

## 7. Gate status

| gate | status |
|---|---|
| Tests pass | ✅ 262, `make check` green |
| Lint / typecheck | ✅ ruff + mypy --strict clean |
| Reproducibility | ✅ figures + numbers regenerate from committed `results/` |
| Retro with paper anchors | ✅ this document |
| `/learn` with `[DELETE]` | ✅ `phase-0.1-grpo-by-hand-learn.md` |
| Three-reviewer pass | n/a — 0.1 is not a stage boundary (Crawl→Walk is after 0.5) |
