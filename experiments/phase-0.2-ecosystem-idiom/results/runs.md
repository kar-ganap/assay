# Phase 0.2 — hosted run ledger

Runs execute on **Prime Intellect infrastructure**, not locally. Nothing here depends on a laptop
staying awake; recover any run at any time with `prime train logs <id>`.

| id | config | model | cost | purpose | status |
|---|---|---|---|---|---|
| `im5b1e6g4a559lnsurg7rmla` | `train-binary.toml` | `sprints/Llama-3.2-1B-Instruct` | $0 | G4, first attempt | completed 200 steps — **unscoreable**, see below |
| `aw5lbjwnwksb9xwq1vzbaopb` | `train-binary-eval.toml` | `sprints/Llama-3.2-1B-Instruct` | $0 | G4, with unfiltered eval | running |

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
