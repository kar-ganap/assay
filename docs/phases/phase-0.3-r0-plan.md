# Phase 0.3 — R0: TinyZero / Countdown

> Plan locked 2026-08-03, before any code. Branch: `phase-0.3-r0-countdown`, cut off
> `phase-0.2-ecosystem-idiom` (0.2 is unmerged at time of writing, so **0.3 depends on it landing**).
> **Stage 1 only** — two measurements. R0 itself is not planned until they return.

## Purpose

R0 retires *"my GRPO loop actually learns."* Phase 0.1 proved gradients flow on `add-3digit`, but a
1B model does three-digit addition by pattern completion. Countdown demands **search and
self-verification** — genuinely different, which is why it is the reproduction on the never-cut list.

## Why this phase opens with measurement instead of a run

`stages.md` scopes R0 as *"TinyZero / Countdown at **1.5B**"*, $10. Planning research found that
**cannot be run as written**, for three independent reasons.

### 1. There is no original number

TinyZero's README makes a cost claim (*"You can experience the Aha moment yourself for < $30"*) and a
qualitative one (*"For Qwen2.5-0.5B base, we know it fails to learn reasoning"*), and points at a W&B
log. **It publishes no accuracy, no config, no metrics.**

`reproductions/README.md` requires *"Original number: value, with the config it was measured under"*
and a **Delta**. There is nothing to compute a delta against. R0's verdict will therefore have to be
against the *qualitative* claim, and the writeup must say so — recorded here so it is a stated
limitation rather than a quiet omission.

### 2. 1.5B is probably starved, which would make R0 ill-posed there

Secondary sources put Qwen2.5-1.5B's Countdown pass rate at ~2% and describe a sparse-reward regime;
TinyZero issue #104 reports loss falling while metrics fail and no self-verification emerges.

Apply Phase 0.1's own instrument. At `p = 0.02`, with `G = 8`:

```
dead_group_fraction = p^8 + (1-p)^8 = 0.851
```

**85% of groups would produce no gradient.** For scale, `add-3digit` — the task Phase 0.1 *chose*,
on this exact criterion — sits at `p = 0.43`, `dead = 0.012`.

A failure at 1.5B would therefore be attributable to the **task**, not the loop, and would not retire
the assumption. **`stages.md`'s claim that "1.5B learns search and self-verification" is unverified
and contradicted by the available evidence** — `lessons.md` #1 firing exactly: the scaffold was
seeded from an LLM-assisted pass and its quantitative claims need first-hand reading.

### 3. It is 3–5× over budget on our loop

Measured from Phase 0.1: **~54k tok/min** generation for 1B on L4 via HF `generate` (derived from
run 7's 476k tokens in 8.8 min of generation, generation being 22% of step time). Countdown needs
reasoning-length completions — ~55× Phase 0.1's 18.6 tokens:

| scale | max_tok | tokens | gen hours | GPU | $ |
|---|---|---|---|---|---|
| 1.5B | 1024 | 26.2M | 12.3 | A100-40GB | **26** |
| 3B | 1024 | 26.2M | 24.6 | A100-40GB | **52** |

Generation alone, before scoring and backward on 1024-token sequences. TinyZero's "< $30" assumed
veRL + vLLM; our loop is HF `generate` by design (§6).

## Stage 1 — the two measurements

### M1: the Countdown base-rate screen (~$1.50)

*"Reachability is screened, never assumed"* (`tasks/todo.md`), applied to ourselves.

**Run** the existing calibration sweep at **k = G = 8** — so unanimity is a *direct, unbiased*
estimate of the step-0 dead-group rate — on **Qwen2.5-1.5B and Qwen2.5-3B**, n=200 prompts,
`max_new_tokens=512`. **Base models, not instruct**: TinyZero is R1-Zero style and this moves the
base rate substantially. Pin both revision hashes.

**Pre-registered decision band**, on `dead_group_fraction`:

| observed | verdict |
|---|---|
| ≤ 0.50 | **workable** — Phase 0.1's run 7 learned 0.571 → 0.923 at a mean dead fraction of 0.472 |
| 0.50 – 0.75 | **marginal** — proceed with reduced expectations; record the handicap explicitly |
| > 0.75 | **starved** — R0 is not well-posed at that scale |

**Rig-broken branch:** `parse_fail_rate > 0.5` means the model cannot emit a parseable expression at
all. That is a *formatting* failure, not a reasoning one, and calls for a more permissive parser —
not a verdict about the task. Phase 0.1 established this: a strict grader filters **hard problems**,
not badly-formatted ones, because parse-failure rises monotonically as pass rate falls.

Report the **per-prompt pass-rate histogram**, not the mean. A half-trivial/half-impossible task set
and a genuinely centred one share a mean while differing 55× in wasted compute — that finding is why
`add-3digit` was chosen over `add-2digit`.

#### M1 RESULT — 2026-08-03. Verdict: **starved at both scales.**

Run as pre-registered: base checkpoints, n=200, k=G=8, `max_new_tokens=512`, T=1.0, seed 0, L4.
Artifacts in `experiments/phase-0.3-r0/results/` (read `PROVENANCE.md` first — the two JSONs carry a
wrong `model_id`, corrected there). Cost **$1.57** of R0's $10 line.

| model | setting | pass@1 | **dead** | parse_fail | wrong | med tok | verdict |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | cd-3 | 0.024 | **0.845** | 0.272 | 0.704 | 189 | starved |
| Qwen2.5-1.5B | cd-4 | 0.006 | **0.950** | 0.244 | 0.750 | 261 | starved |
| Qwen2.5-1.5B | cd-5 | 0.004 | **0.965** | 0.236 | 0.760 | 304 | starved |
| Qwen2.5-1.5B | cd-6 | 0.001 | **0.995** | 0.221 | 0.779 | 346 | starved |
| Qwen2.5-3B | cd-3 | 0.059 | **0.620** | 0.151 | 0.789 | 138 | **marginal** |
| Qwen2.5-3B | cd-4 | 0.009 | **0.925** | 0.170 | 0.821 | 189 | starved |
| Qwen2.5-3B | cd-5 | 0.007 | **0.950** | 0.171 | 0.822 | 184 | starved |
| Qwen2.5-3B | cd-6 | 0.006 | **0.955** | 0.168 | 0.826 | 202 | starved |

**Against the band: nothing reaches `≤ 0.50` anywhere in the grid.** One cell lands marginal — the
largest model on the easiest setting — and it is already the most expensive model in reach.

**The rig-broken branch did not fire.** `parse_fail` is 0.15–0.27 against its 0.5 threshold, and
`wrong_answer` runs 0.70–0.83. The permissive parser is doing its job; the models produce legal
expressions that miss. So this is a statement about the *task*, which is what the branch exists to
establish.

**Per-prompt histograms confirm it is not a mixture.** 1.5B/cd-3: `[169, 24, 6, 1, 0, 0, 0, 0, 0]`
over k=0..8 — 169 of 200 prompts solved zero times out of eight, and **not one prompt** was solved
more than three times. 3B/cd-3: `[124, 59, 15, 2, 0, 0, 0, 0, 0]`. This is a uniformly hard set, not
the half-trivial/half-impossible mixture the histogram requirement was written to catch. `pass@8` is
0.155 (1.5B) and 0.380 (3B), so the ceiling is low even with eight tries.

**§2's prediction, made before any GPU was booked, held.** `p⁸ + (1−p)⁸` at the ~2% secondary-source
pass rate gave **0.851**; measured **0.845**. Phase 0.1's task-selection instrument transferred
across models, tasks and scales on its first out-of-sample use.

**Falsified: `stages.md`'s "1.5B learns search and self-verification."** Corrected in
`reproductions/README.md` and `docs/stages.md` with the measured table. `lessons.md` #1 — the
scaffold's LLM-assisted quantitative claims need first-hand reading — fires again, and this time the
cost of not checking would have been $8.43 and an uninterpretable run.

**Ancillary, and it decides one open question:** peak CUDA memory 5.93 GB (1.5B) / 9.79 GB (3B)
against an L4's 24 GB. **L4 is right for both**; there is no case for the A100 tier that consumed
$4.26 of Phase 0.1 for headroom never used.

**Consequence for R0.** Countdown at an affordable scale would train against 62–99% dead groups, so a
failure would be attributable to reward sparsity in the task rather than to the loop — R0 would not
retire *"my loop actually learns."* **What R0 becomes is a scope decision, not a technical one, and
is deferred to the user.** Options, unranked and not yet costed: (a) run it at 3B/cd-3 as an
explicitly handicapped reproduction, reporting the 0.620 dead fraction as a stated limitation;
(b) keep Countdown but add a curriculum or a shaped reward, which changes what is being reproduced;
(c) substitute a different reproduction target for R0; (d) accept `add-3digit` + Phase 0.2's
independent-trainer cross-check as sufficient evidence that the loop learns, and retire R0.

### M2: the vLLM/HF logprob mismatch (~$0.50 + image build)

Our loop samples with `policy.generate()` and differentiates `policy.logprobs()`. Today both are one
HF forward pass, so they are the same distribution **by construction**, and `config.py` leans on it:

> *"with a single epoch the importance ratio is identically 1, so clipping is a no-op no matter what
> epsilon says. **Rung 4 of the ladder is cut for exactly this reason.**"*

vLLM makes sampler and scorer different implementations. `π_HF(y)/π_vLLM(y) ≠ 1`, and per-token
discrepancies compound multiplicatively over a long sequence. Three things stop holding: the
estimator is no longer unbiased by construction; **rung 4's cut loses its justification exactly when
clipping becomes necessary** (and `clipping_is_active` would still return `False`, since it gates on
`epochs_per_batch > 1`); and R0 would be testing a *modified* loop. `prime-rl` logs `Max Off-Policy`
on every line, so the field treats this as real.

**Measure:** sample one batch (128 rollouts) with vLLM; score the *same token sequences* with our
existing `policy.logprobs()`; report the **per-token** log-prob discrepancy distribution — which is
length-independent — and the implied sequence-level ratio at L = 64 / 512 / 1024.

**Pre-registered band**, on the implied sequence ratio at the operating length:

| observed | verdict |
|---|---|
| within [0.9, 1.1] | **negligible** — adopt vLLM, record the measurement here |
| outside | **not free** — either re-enable rung 4 with genuine importance weights (un-cut on *earned* evidence), or keep HF `generate` and re-scope R0 |

Reusable regardless of R0's fate: §6 puts every phase from Walk onward on vLLM, so this has to be
characterised eventually. Now, on a task with ground truth, is the cheapest it will ever be.

#### M2 RESULT — 2026-08-03. Verdict: **`not_free`.**

128 prompts, Countdown cd-3, Qwen2.5-1.5B (same pins as M1), T=1.0/top_p=1.0, 512 max tokens, L4.
34,456 completion tokens compared. Artifact:
`experiments/phase-0.3-r0/results/mismatch-vllm-Qwen2.5-1.5B-seed0.json`. Cost **~$0.15**.

**Both rig checks pass first.**

| check | result |
|---|---|
| HF vs HF on the same ids | `max|delta| = 0.00e+00` — **exactly** zero, not "small" |
| vLLM pass@1 vs M1's HF sampler | 2/128 vs 3.12 ± 1.74 expected, **z = −0.64** |
| `independence_ratio` | **0.968** — errors are iid, so the √L extrapolation is legitimate |

The third is the one that had to be measured rather than assumed: it licenses extrapolating a
per-token spread to sequence length at all.

**The per-token discrepancy is negligible. The sequence ratio is not.**

| statistic | value |
|---|---|
| median δ per token | **−0.000003** — the typical token has no discrepancy at all |
| p01 / p99 | −0.1221 / +0.1203 |
| mean δ | −0.001056 (z = −4.5) |
| σ per token | 0.043402 |
| `max_off_policy` | 0.6432 |

| L | median ratio | ±1σ | E[ratio] |
|---|---|---|---|
| 64 | 0.9346 | [0.661, 1.323] | 0.993 |
| **512** | **0.5823** | **[0.218, 1.555]** | 0.943 |
| 1024 | 0.3390 | [0.085, 1.360] | 0.889 |

**Against the band [0.9, 1.1] at L=512: `not_free`, and not marginally** — the median alone is 0.58.

#### The obvious reading of that table is wrong, and the arithmetic says so

It looks like vLLM systematically over-scores its own samples. It does not. Two facts fix the
interpretation:

1. **The mean *must* be negative.** Sampling `y ~ π_vLLM` and evaluating
   `E[log π_HF(y) − log π_vLLM(y)]` is `−KL(π_vLLM ‖ π_HF) ≤ 0` by Gibbs. It cannot come out
   positive for *any* two distinct implementations, so its sign carries no information about which
   is "right". Its magnitude is the per-token KL: **0.00106 nats**, which is tiny.

2. **The drift is not independent of the spread — it is determined by it.** A log-normal ratio with
   `E[ratio] = 1` (which unbiasedness requires exactly) forces `μ = −σ²/2`. Observed
   `μ = −0.001056` against `−σ²/2 = −0.000942`: **agreement to 12%**. The measured `E[ratio]` of
   0.993 / 0.943 / 0.889 confirms it from the other side.

So the correct statement is: **the importance ratio remains unbiased in expectation, and becomes
useless per sequence.** Its log spreads as `σ√L` — 0.98 nats at 512 tokens — so the mass slides into
a long right tail while the mean stays pinned near 1. The *typical* sequence is off by ~1.7×, and the
estimator's variance is carried by rare large weights. That is the textbook failure of naive
importance sampling on long sequences, and it is exactly what clipping exists to contain.

**Where the discrepancy comes from:** bf16 accumulation order, not semantics. The median token
disagrees by 3e-6 and p01/p99 sit at ∓0.12. Nothing is wrong with either implementation. Length is
doing all the work.

#### Consequence: rung 4 is un-cut, on earned evidence

`config.py` cut rung 4 because "with a single epoch the importance ratio is identically 1". M2 shows
that is a property of the **sampler/scorer pairing**, not of the single-epoch design — and
`clipping_is_active` gates on `epochs_per_batch` alone, so the guard and the hazard are keyed on
different things. The docstring now records this.

**This is the pre-registered "un-cut it on earned evidence" branch firing.** Adopting vLLM anywhere
from Walk onward requires a genuine importance weight and a clip; `prime-rl` logging
`Max Off-Policy` on every line is the same conclusion reached by the field.

**For R0 specifically, this does *not* simply make the run affordable.** vLLM's speed is available
only behind a loop change that Phase 0.1 deliberately did not make, and that change is the user's to
write (§7). The cost objection is not removed — it is converted into a prerequisite.

**Not measured, and not claimed:** the actual speedup. The 7-minute app included image pull, engine
init, model load, the HF scoring pass and the control pass, so no clean generation-throughput number
comes out of it. Quoting one would be inventing it.

## Design notes

**`CountdownFamily` generates only solvable instances.** Sample an expression tree first, evaluate
it, use the result as the target. Otherwise the measured base rate confounds *"the model cannot
reason"* with *"no solution exists"*, and the screen answers the wrong question.

**The grader is permissive by design.** Normalise `× x ÷`, strip commas, `ast.parse` under an
operator allowlist. `sivit/countdown-plain@latest` on the Hub is a working reference for exactly this
parser and was read before writing ours.

**`sweep_setting` needs a grader seam.** It currently hardcodes `grade_binary(c.text, p.answer)`;
Countdown's grader needs the *given numbers*, not just the target. Adding an optional `grader`
parameter that receives the whole `Prompt`, defaulting to today's behaviour.

## Gates

**G1 — local.** `make check` green, zero GPU: generation deterministic from seed, **every generated
instance provably solvable** (asserted by search, not by construction alone), parser accepts
`×`/`x`/`÷`/commas, rejects a number used twice, `parse_fail` never conflated with `wrong_answer`.

**G2 — M1 returns a dead-group fraction** for both scales, with a per-prompt histogram, against the
band above. ✅ **MET 2026-08-03 — verdict `starved`.** See the M1 RESULT section above.

**G3 — M2 returns a per-token discrepancy distribution** and implied sequence ratios, against the
band above. ✅ **MET 2026-08-03 — verdict `not_free`.** See the M2 RESULT section above.

**Stage 1 ends there.** R0's scale, budget and well-posedness are decided by G2 and G3, and the
run itself is planned separately.

## Non-goals

- **Not running R0.** That is what these measurements decide.
- **No vLLM adoption decision.** M2 produces a number; the decision follows it.
- **Not re-scoping the reproduction ledger.** The missing-original-number problem is recorded above
  and resolved when R0's writeup is drafted.

## Change log

| date | change |
|---|---|
| 2026-08-03 | Plan locked. Stage 1 scoped to two measurements after research showed R0 unrunnable as written on three counts. Both decision bands pre-registered. |
| 2026-08-03 | **G3 met. M2 returned `not_free`** — per-token discrepancy negligible (median 3e-6) but compounding to a ±1σ sequence ratio of [0.22, 1.55] at L=512. Both rig checks passed (HF-vs-HF exactly 0; pass@1 z=−0.64; `independence_ratio` 0.968). The drift is `−σ²/2`, i.e. the ratio stays unbiased in expectation and becomes useless per sequence. **Rung 4 un-cut on earned evidence**; `config.py`'s justification annotated. |
| 2026-08-03 | **G2 met. M1 returned `starved` at both scales** — no cell in the 2×4 grid clears the pre-registered `≤ 0.50` band. Rig-broken branch did not fire. §2's `p⁸ + (1−p)⁸` prediction (0.851) matched the measurement (0.845). R0's well-posedness is now the open question; M2 proceeds regardless, since it is reusable from Walk onward. |
