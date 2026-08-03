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
band above.

**G3 — M2 returns a per-token discrepancy distribution** and implied sequence ratios, against the
band above.

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
