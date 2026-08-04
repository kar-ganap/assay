# Phase 0.3 retro — R0, and why it retires

> Written 2026-08-03 at phase completion, before merge. Sections anchor to their eventual paper
> section (`docs/process.md`). Plan: `phase-0.3-r0-plan.md`. Predecessor:
> `phase-0.2-ecosystem-idiom-retro.md`.

## 1. Hypothesis and purpose → **Methods**

R0 exists to retire *"my GRPO loop actually learns."* Phase 0.1 proved gradients flow on
`add-3digit`, but a small model does three-digit addition by pattern completion. Countdown demands
**search and self-verification** — genuinely different, which is why R0 sat on the never-cut list.

The phase never ran R0. Planning found it unrunnable as scoped, and the phase became **three cheap
measurements that decide whether it is worth running at all**. Each was pre-registered with a band
and both failure branches before it executed.

## 2. What happened → **Results**

**$2.21 of a $10 line. R0 retires, on three independent measurements.**

| | question | verdict | cost |
|---|---|---|---|
| **M1** | is Countdown learnable at any scale we can afford? | **`starved`** | $1.57 |
| **M2** | is a fast sampler safe for our estimator? | **`not_free`** | $0.15 |
| **M3** | is any Countdown setting learnable *and* still about search? | **nothing admitted** | $0.49 |

### M1 — the base-rate screen

Qwen2.5-1.5B and Qwen2.5-3B, four settings, n=200, k=G=8, base checkpoints, 512 tokens.

| model | cd-3 | cd-4 | cd-5 | cd-6 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.845 | 0.950 | 0.965 | 0.995 |
| Qwen2.5-3B | **0.620** | 0.925 | 0.950 | 0.955 |

`dead_group_fraction`, against a pre-registered band of ≤0.50 workable / 0.50–0.75 marginal / >0.75
starved. **Nothing clears it at either scale.** `parse_fail` 0.15–0.27 against its 0.5 rig-broken
threshold, so this is a statement about the task, not the parser.

### M2 — the sampler mismatch

| | value |
|---|---|
| median δ per token | −0.000003 |
| p01 / p99 | −0.122 / +0.120 |
| ratio @ 512 | 0.5823, ±1σ **[0.218, 1.555]** |

Band was [0.9, 1.1]. Three rig checks passed *first*: HF-vs-HF on the same token ids gave **exactly**
0.0, vLLM's pass@1 matched M1's HF sampler at z = −0.64, and `independence_ratio` = 0.968.

### M3 — the difficulty screen

| setting | pass@1 | dead | explore | tok | verdict |
|---|---|---|---|---|---|
| `cd-3` (M1) | 0.0594 | 0.620 | 6.40× | 138 | — |
| `cd-3-easy` | 0.0619 | 0.615 | 6.22× | 143 | rejected: band |
| `cd-3-mid` | 0.0587 | 0.625 | 6.38× | 137 | rejected: band |

Criteria 2–4 passed comfortably everywhere; only the band failed.

## 3. Surprises — first-class → **Discussion**

### 3.1 Phase 0.1's task-selection instrument transferred out-of-sample, first try

`dead = p⁸ + (1−p)⁸` at `G=8` was derived in Phase 0.1 to choose `add-3digit`. Applied to the ~2%
pass rate secondary sources report for 1.5B on Countdown, it predicted **0.851** before any GPU was
booked. Measured **0.845**.

A different model, a different task family, a different capability scale — and the criterion held to
0.006. That is the strongest evidence to date that the dead-group fraction is a *portable* screening
statistic rather than a description of one task, which matters because the whole `assay` thesis is
that cheap pre-training measurements predict expensive post-training outcomes.

### 3.2 The arithmetic axis is inert — the pre-registered negative was right for the wrong reason

M3's negative branch predicted a **trade-off**: that Countdown's difficulty and its search character
could not be separated. What happened instead is that the axis M3 varied **does not move difficulty
at all.**

The variants are genuinely different tasks (operand mean 13.4 → 5.7, target mean 66.8 → 21.8, ~2.5×
less arithmetic), and pass@1 moved 0.0594 → 0.0619 — **z = 0.30** against `SE_diff = 0.0084`.
Histograms, completion lengths and exploration ratios are all near-identical.

The design held the structural search space fixed by construction (3 numbers, 4 operators, same tree
shapes) and varied only arithmetic burden. It isolated the two axes cleanly, **and one of them turned
out to carry no signal.** So the bottleneck is the search over expression trees, not the sums.

This is stronger than "we could not find a setting": anyone proposing to make Countdown learnable by
easing the arithmetic now has a measured number to argue with.

### 3.3 M2's obvious reading is wrong, and the arithmetic says so

It looks like vLLM systematically over-scores its own samples. It cannot be that. Sampling
`y ~ π_vLLM` and evaluating `E[log π_HF(y) − log π_vLLM(y)]` is `−KL(π_vLLM ‖ π_HF) ≤ 0` **by
Gibbs' inequality** — negative for *any* two distinct implementations, so its sign carries no
information about which is right.

Nor is its magnitude independent information. A log-normal ratio with `E[ratio] = 1` — which
unbiasedness requires exactly — forces `μ = −σ²/2`. Observed −0.001056 against −0.000942: **12%
agreement**, confirmed from the other side by measured `E[ratio]` = 0.993 / 0.943 / 0.889 at
L = 64 / 512 / 1024.

So the correct statement is: **the importance ratio remains unbiased in expectation and becomes
useless per sequence.** Its log spreads as `σ√L` — 0.98 nats at 512 tokens — so mass slides into a
right tail while the mean stays pinned near 1. That is the textbook failure of importance sampling on
long sequences. The source is bf16 accumulation order, not semantics: the median token disagrees by
3e-6. **Length does all the work.**

### 3.4 A second stack had already solved it, with parameters that match our measurement

`verl-project/verl` ships `examples/rollout_correction/`, found the same day:

| verl config | M2's measurement |
|---|---|
| `rollout_is=sequence` | negligible per token, bites only at sequence level |
| `rollout_is_threshold=2.0` | M2's ±1σ at L=512 is [0.218, **1.555**] — the cap sits at the edge of the measured distribution |
| `rollout_is_batch_normalize=true` | fixes precisely M2's residual `E[ratio]` = 0.943 |
| `rollout_is_eff_sample_size` | ESS — effective batch < nominal batch |

Independent confirmation that `not_free` is a real effect and not a harness artifact, plus a concrete
design for rung 4 rather than one we would invent. `prime-rl`'s `Max Off-Policy` is the same
conclusion from a third stack.

### 3.5 A pre-registered threshold cannot be implemented as a naive float comparison

M3's criterion 2 is `pass@k / pass@1 ≥ 3`. Implemented as `ratio >= 3.0` it **rejects its own
boundary**: `0.30 / 0.10` is `2.9999999999999996` in IEEE754. Not hypothetical — 240 and 80 successes
out of 1600 rollouts produce exactly it. Rewriting as `pass_at_k >= 3.0 * pass_at_1` does not help
(`3.0 * 0.1` overshoots to `0.30000000000000004`).

A pre-registered line is a *claim about where the line is*. An implementation that moves it by one
ULP silently breaks the pre-registration, and no test that avoids the boundary would notice.

### 3.6 The rig-broken branch was unreachable

`mismatch_verdict`'s first branch exists to catch misaligned token ids. Writing its test found that
such an input **overflows `math.exp` and raises before the verdict can classify it** — the guard
could never fire on the case it was written for. Caught only because the test supplied a genuinely
rig-broken input rather than a merely bad one.

## 4. What to change → **Methods (later phases)**

1. **Rung 4 is un-cut, on earned evidence.** `config.py` cut it because "with a single epoch the
   importance ratio is identically 1" — a property of the *sampler/scorer pairing*, not of the
   single-epoch design, and `clipping_is_active` gates on `epochs_per_batch` alone, so the guard and
   the hazard are keyed on different things. Any sampler swap from Walk onward re-earns it, with
   §3.4's design.
2. **The dead-group band's anchor conflates two quantities.** The plan justifies ≤0.50 by "run 7
   learned at a mean dead fraction of 0.472", which is a *training mean*, while M1/M3 measure *step
   0*. Run 7's step-0 figure was 0.012. The band is still a valid screen — and it was pre-registered,
   so applying it was correct — but the justification sentence must be fixed before it is reused.
   `dead` is U-shaped in `p`, so 0.472-from-saturation and 0.620-from-starvation are opposite
   regimes the statistic cannot distinguish.
3. **Screen a reproduction target's *paper* before its task.** See §5.

## 5. Limitations → **Limitations**

- **R0 was structurally unsatisfiable from the start, and that was knowable on day one.**
  `reproductions/README.md` requires "Original number: value, with the config it was measured under"
  and a **Delta**. TinyZero publishes a cost claim and a qualitative claim, and points at a W&B log —
  **no accuracy, no config, no metrics.** No amount of compute produces a delta against nothing. The
  scale and budget problems were the visible ones; this was the fatal one, and it needed a README
  read, not a screen.
- **The screen measures step 0; TinyZero's claim is about emergence.** Search and self-verification
  are supposed to *appear during training*, so a low step-0 rate is what its thesis predicts. Using
  step 0 to rule the task out assumes reachability predicts trainability — the project's own
  `p_hack@64` assumption (§4), pre-registered, and 38% of groups do carry gradient. **The honest
  claim is bounded by budget: not learnable within R0's 200 steps at affordable scale.** Anthropic
  needed ~1,500 steps for exploits that were not reachable. This does not show Countdown is
  unlearnable.
- **Every screen is n=1 seed.** The *threshold* verdicts survive that: 0.615 against a 0.50 band is a
  0.115 margin, far outside plausible seed noise, and a threshold is what one seed can settle
  (Phase 0.2's G4 made the same argument). **The inertness claim in §3.2 is the exposed one** — it
  asserts an effect is *absent* between settings, and prompt-set seed variance is uncharacterised.
  A second seed costs ~$0.49. R0's retirement does not depend on it; the sharpness of §3.2 does.
- **M2 is one model, one task, one sequence-length regime.** bf16 on an L4. Whether the per-token σ
  of 0.043 is representative of other models, dtypes or attention backends is untested, and the
  extrapolation to L=1024 rests on `independence_ratio` measured at mean length 269.
- **The speedup vLLM would buy is unmeasured**, so "vLLM makes R0 affordable" was never established —
  only that it costs a loop change. The 7-minute app included image pull, engine init, model load and
  two scoring passes.
- **What R0 would have retired stays open.** Every task the loop has been shown to learn is
  **pattern completion, not search**. That is exactly the gap Countdown was chosen to close.

## 6. Related-work touchpoints → **Related Work**

- **`verl`** — read first-hand 2026-08-03; `docs/related-work.md` carries the verified section.
  Prior art for M2 (§3.4). Implements **DrGRPO**, independently where Phase 0.1's
  length-normalisation finding pointed. **Does not rescue R0**: no Countdown dataset in
  `examples/data_preprocess`, no tinyzero recipe in the `verl-recipe` submodule; TinyZero appears
  under "Awesome Projects Built with `verl`" as an external link. Repo moved `volcengine/verl` →
  `verl-project/verl`.
- **TinyZero** — the reproduction target. Its publishing practice is the finding: a widely-cited
  reproduction with no published metrics is not reproducible in the ledger's sense, and this is worth
  a sentence in the paper's discussion of reproduction standards.

## 7. Gate status

| gate | status |
|---|---|
| Tests pass | ✅ 350, `make check` green |
| Lint / typecheck | ✅ ruff + mypy --strict |
| Reproducibility | ✅ artifacts committed under `experiments/phase-0.3-r0/results/`; every verdict recomputes from committed data via `crawl.admission` / `crawl.mismatch`. **Read `results/PROVENANCE.md`** — M1's two JSONs carry a wrong `model_id` from a dict-spread bug, corrected there with three independent recovery paths |
| Retro with paper anchors | ✅ this document |
| `/learn` with `[DELETE]` | ✅ `phase-0.3-r0-learn.md` |
| Three-reviewer pass | n/a — 0.3 is not a stage boundary (Crawl→Walk is after 0.5) |

**Spend: $2.21** of R0's $10 line. **$7.79 released.**
