# Phase 0.2 — hosted run ledger

Runs execute on **Prime Intellect infrastructure**, not locally. Nothing here depends on a laptop
staying awake; recover any run at any time with `prime train logs <id>`.

| id | config | model | cost | purpose | status |
|---|---|---|---|---|---|
| `im5b1e6g4a559lnsurg7rmla` | `train-binary.toml` | `sprints/Llama-3.2-1B-Instruct` | $0 | G4, first attempt | completed 200 steps — **unscoreable**, see below |
| `aw5lbjwnwksb9xwq1vzbaopb` | `train-binary-eval.toml` | `sprints/Llama-3.2-1B-Instruct` | $0 | G4, with unfiltered eval | **PASSED** — eval 0.9980 at step 200 |
| `xqju72r2dxmeyee19kkrght7` | `train-binary-nofilter.toml` | `sprints/Llama-3.2-1B-Instruct` | $0 | follow-up A/B: `zero_advantage` filter `enforce = false` | completed — **prediction falsified** |

Dashboard: `https://app.primeintellect.ai/dashboard/training/<id>`

## Why the first run is unscoreable

`prime-rl`'s headline `Reward` is the mean over rollouts that survive the **zero-advantage pre-batch
filter**, not over the batch. Verified empirically: `reward × trainable` is an integer on 24/24
captured steps while `reward × 128` is not (e.g. step 196, `0.7917 × 24 = 19.00` vs
`× 128 = 101.34`).

The filter removes **unanimous** groups, and a unanimous group late in training is overwhelmingly an
all-*correct* one — so the reported metric excludes the policy's successes, with a bias that grows as
the policy improves. At step 200 only `8/128` rollouts survived, so `Reward 0.8750` is 7 correct out
of the one group that still contained an error.

That number is therefore **not comparable** to Phase 0.1's `0.923 ± 0.018`, which was a mean over all
128 rollouts. Scoring it against the ≥0.85 gate would compare two different estimators.

## First-run findings (stand regardless of G4's outcome)

- **G5 answered.** The `zero_advantage` filter is **on by default** in `prime-rl`. Drops are exact
  multiples of `G=8`, and `Trainable` fell `120/128 → 8/128` across the run — the dead-group
  pathology Phase 0.1 measured, here silently absorbed. **Any Stage-2 grid on this stack must
  override the filter list**, or ablation-D-style measurements are invisible.
- **The free tier has three stacked, undocumented gates**: environment PUBLIC · CI action SUCCESS ·
  README containing the phrase "reward hacking sprint" with stated hypotheses. Only the first
  returns a specific error; the other two share one opaque message.

## Comparator correction (2026-08-03)

The plan gated against "base rate 0.433". **That is the wrong comparator.** `0.433` came from the
calibration sweep (`k=8`, `max_new_tokens=256`, different prompt set). The like-for-like figure is
Phase 0.1 run 7's own first-ten-step true reward, **0.571 ± 0.019**.

The second run's step-1 eval reads **0.5879** — inside that band. An independent trainer, on an
independently published environment, reproducing our measured starting point to within the seed band
is stronger evidence for the port than matching 0.433 would have been.

Endpoint target is unchanged: run 7's **0.923 ± 0.018**, with the gate at **≥ 0.85**.

## G4 — PASSED (run `aw5lbjwnwksb9xwq1vzbaopb`)

Unfiltered eval on **held-out** prompts (seed 999 vs training's seed 0; 1 shared prompt in 512
against 1.26 expected by chance, so no leakage):

| step | eval reward | comparator |
|---|---|---|
| 1 | **0.5879** | Phase 0.1 run 7, first 10 steps: 0.571 ± 0.019 — **inside the band** |
| 25 | 0.7246 | |
| 175 | 1.0000 | 512/512 |
| 200 | **0.9980** | 511/512. Phase 0.1 run 7, last 10 steps: 0.923 ± 0.018 |

Gate was ≥ 0.85. An independent trainer, on an independently published environment, starts inside
our measured band and finishes **above** our endpoint. Validates the port *and* cross-checks the
hand-rolled loop, which is what this phase existed to do.

## The mechanism, and the follow-up it prompted

`prime-rl` does not merely *drop* dead groups — it **keeps sampling to refill the batch**. Late in
the reference run the denominator stops being `batch_size`:

```
step 196   trainable  24 of 384 generated   (94% discarded)
step 198   trainable  24 of 384 generated
step 200   trainable   8 of 128 generated
```

Phase 0.1's loop took the 128 rollouts it got and let ~70% contribute nothing. This one pays up to
3× the sampling cost to keep every gradient step at full strength. **That is the leading explanation
for 0.998 vs 0.923** — not a better optimiser, but a refusal to spend a step on a starved batch.

Run `xqju72r2dxmeyee19kkrght7` tests it: identical config with `enforce = false` on the
`zero_advantage` filter, one field changed. Prediction recorded before launch — filter off should
fall materially toward 0.923, and the generated pool should stop exceeding `batch_size`. The
discriminator only becomes visible late, once the dead fraction is high enough to force oversampling.

## Follow-up A/B — PREDICTION FALSIFIED

| step | filter ON | filter OFF |
|---|---|---|
| 1 | 0.5879 | 0.6113 |
| 25 | 0.7246 | 0.7266 |
| 75 | — | 0.9766 |
| 150 | — | 1.0000 |
| 200 | **0.9980** | **1.0000** |

Predicted: filter OFF falls materially toward Phase 0.1's 0.923. **It did not.** If anything the
unfiltered arm converged faster.

**Mechanism caveat — the lever only partially took.** Oversampling dropped sharply but did not stop
(`enforce = false` steps 193–200: `128, 128, 256, 256, 128, 128, 128, 128`, against the filtered
run's `256, 128, 128, 256, 384, 256, 384, 128, 128`). So the filter was weakened, not removed. That
cuts *against* the hypothesis rather than rescuing it: batches then held ~104 dead rollouts of 128 —
our loop's exact situation — and performance did not degrade.

**Conclusion: the filter is not why an independent trainer scored higher.** Dead groups still cost
*compute*, exactly as Phase 0.1 measured. What is now unsupported is that reclaiming them explains
the performance difference.

The tidy arc this kills is worth naming: "dead groups are a pathology, and here is the production
mitigation that fixes it" was the write-up that had already half-formed. The data does not support it.

## What must NOT be claimed from this

**Both prime runs are n=1.** `CLAUDE.md` §10.3, added at the end of Phase 0.1, says *no directional
claim from n=1* — earned when three claims reversed, flattened or narrowed between n=1 and n=3.

So: **"prime-rl trains better than our hand-rolled loop" is not an assertion this phase is entitled
to make.** G4 is a *threshold* (0.998 ≥ 0.85), which one seed can settle; the ~0.07 comparison
against 0.923 ± 0.018 is a *direction*, and it cannot.

Remaining untested differences: no KL term in the prime configs (our run 7 carried β = 0.04, and
ablation B found the leash made things worse), optimizer, schedule, warmup, LoRA rank, on-policy
discipline. Candidates, not conclusions.
