# Compute spend tracking

Log every compute/API cost **at the time incurred**, not retrospectively (`docs/desiderata.md` §17).

**Spend is gated by stage.** A stage's budget is not committed until the prior stage's exit gate
passes. **Crawl is authorised to the Modal credit pool, ~$17** (corrected 2026-08-04 from an erroneous ~$50) — see the budget-revision block below; the ~$17 figure elsewhere in this file is the superseded planning estimate and is kept only where it records what a decision was made against.

**Target: ~$98 · Hard cap: $150.** At the 2-month cadence expect ~$120 — longer projects accrete
re-runs.

---

## Ledger

| Date | Stage | Phase | Run | Backend | Duration | Cost | Running total | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-07-27 | 0 · Crawl | 0.1 | Calibration sweep — 5 runs (n=4/16/16/64/200, k=4–8) | Modal L4 | ~45 min GPU | **~$0.60 (est.)** | ~$0.60 | ~19,000 completions ≤256 tok, Llama-3.2-1B. Pinned Phase 0.1's task. Still an estimate — the sweep predates the artifact volume, so there is no per-step wall clock to reconcile from. |
| 2026-07-28 | 0 · Crawl | 0.1 | Bringing up `HFPolicy`: ~12 invocations, mostly failures | Modal A10G → A100-40GB | ~1–1.5 h GPU + image builds | **~$2–4 (est.)** | ~$3–5 | 6 OOM/crash failures. **Cannot be reconciled** — a crashed run writes no artifact, so these leave no measurable trace. |
| 2026-07-28→08-01 | 0 · Crawl | 0.1 | **All 22 surviving runs** — LR probes ×7, overfit, run7 ×4, run1 ×2, run2 ×3, run3, ablations B/B-control/C/D | Modal A100-40GB + L4 | **8.0 h GPU, measured** from per-step wall clock + 2 min/run container overhead | **$8.41 (measured)** | ~$12–13 | A100-40GB $4.26 · L4 $4.15. Per-run breakdown regenerates from the volume. |
| 2026-08-03 | 0 · Crawl | 0.2 | Ecosystem-idiom port — 3 hosted GRPO runs (G4 binary, G4 unfiltered A/B, filter probe) | Prime Sprints free queue | ~3 × 200 steps | **$0.00** | ~$12–13 | Free tier, Llama-3.2-1B-Instruct. All five gates met at zero spend. |
| 2026-08-03 | 0 · Crawl | 0.3 | **M1 — Countdown base-rate screen.** Qwen2.5-1.5B + Qwen2.5-3B, 4 settings each, n=200, k=8, 512 tok — 12,800 completions | Modal L4 | **1 h 58 m app wall clock, measured** (13:48→15:46 PDT, `ap-ZFGqkvE6gZKI5QYK2ArRsN`) | **$1.57 (measured, upper bound)** | ~$14–15 | Wall clock includes image build and model downloads, so charged GPU time is at most this. Verdict **starved at both scales** — see below. |
| 2026-08-03 | 0 · Crawl | 0.3 | **M2 — vLLM/HF log-prob mismatch.** 128 prompts, 34,456 tokens compared, + one failed launch (missing `nvcc`) | Modal L4 | ~11 min GPU across two apps (7 min + 4 min), measured | **~$0.15 (measured)** | ~$15 | Estimated at $0.50 in the plan. Verdict **`not_free`** — see below. |
| 2026-08-03 | 0 · Crawl | 0.3 | **M3 — difficulty screen.** Qwen2.5-3B, `cd-3-easy` + `cd-3-mid`, n=200, k=8 | Modal L4 | **37 min, measured** (17:48→18:25 PDT, `ap-7NCM78vjTJDs2AWumfdi6G`) | **$0.49 (measured)** | ~$15.50 | Estimated $0.50. Nothing admitted; the arithmetic axis is inert. **R0 retires** — total Phase 0.3 spend **$2.21** of R0's $10 line, and the remaining $7.79 is released. |

### RECONCILED 2026-08-01 — measured, not reconstructed

**$8.41** across 22 surviving runs, computed from each run's own per-step wall clock plus a 2 min/run
container-and-model-load overhead, at **verified** rates. Replaces the reconstructed "$8–13".

**Rates verified 2026-08-01** from `modal.com/pricing` (the table below had L4 as *unverified* since
the first run): **L4 $0.799/h · A100-40GB $2.099/h · A100-80GB $2.498/h · H100 $3.949/h.**
A100-40GB is **2.6×** L4, not the 3–4× previously assumed.

**$4.26 of the $8.41 went to A100-40GB, and essentially all of it was avoidable** — those runs peaked
at 13.5–14.5 GB against an L4's 24 GB. The tier was raised to chase an OOM whose real cause was
gradient checkpointing never engaging, and left on afterwards.

Not included, and not recoverable: the ~6 crashed runs (no artifact ⇒ no wall clock) and image
builds. **Realistic total ≈ $10–12** including the two estimated rows above.

> 🛑 **CRAWL IS OVER BUDGET.** $12–13 spent of an authorised **$17**, with **R0 ($10) and R1 ($2)
> both unstarted and both on the never-cut list**. Finishing Crawl as scoped needs ~$12 more against
> ~$4–5 remaining. This is a scope decision, not an accounting one, and it is the user's:
> Phase 0.1's own line was **$5**.

> 🛑 **REPLAN TRIGGER FIRED.** `spend.md`'s own rule: *"a stage's actual spend exceeds its estimate
> by >50% → stop, re-estimate, and record the revision before continuing."* Phase 0.1 is 60–160%
> over its $5 line, and R0 ($10) and R1 ($2) are both on the never-cut list. **Runs are stopped.**
>
> Two causes, both mine. Spend was not logged at the point of incurring across two days despite the
> rule saying so and my flagging it twice. And the GPU tier was raised to diagnose an OOM and never
> lowered once the real cause was found — a ~3–4× multiplier left switched on for every subsequent
> run.
>
> **Measured cost of the remaining work:** 8 ladder entries × ~10 min ≈ 1.3 h. On L4 that is roughly
> **$1**, against ~$3.30 on A100-40GB. The open `Prime Sprints free queue` thread in `tasks/todo.md`
> is now directly relevant — it was scoped as moving $28.

> ⚠️ **Both rows are estimates and neither was logged at the time.** Desideratum 17 says log at the
> point of incurring; on 2026-07-28 roughly a dozen Modal invocations went by unlogged while
> debugging. That is exactly the drift the rule exists to prevent — the numbers here were
> reconstructed after the fact and could be off by a factor of two in either direction.
> **Reconcile against the Modal dashboard before Phase 0.1 closes**, and log the ladder runs as they
> happen rather than afterwards.

> ⚠️ **This figure is an estimate, not a reading.** Modal's L4 rate is not in the verified-pricing
> table below, and the cost was not captured at the time of the runs. Confirm from the Modal
> dashboard and replace this row — logging an estimate as if it were a measurement is the failure
> mode desideratum 17 exists to prevent.

---

## Budget by stage (from `docs/stages.md`)

| Stage | Authorised? | Line | Est |
|---|---|---|---|
| **0 · Crawl** | ✅ **yes** | Phase 0.1 plumbing smoke (0.6B, easy task) | $5 |
| | | R0 — TinyZero/Countdown at 1.5B | $10 |
| | | R1 — Prime Intellect 1B hacking (free queue if available) | $2 |
| | | **subtotal** | **$17** |
| **1 · Walk** | ⛔ gated on Crawl exit | R2 fuzzing repro (no GPU) · screen + positive control · A1/A2 exploit-finder | **$20** |
| **2 · Run** | ⛔ gated on Walk exit | exploratory grid ($0 on free queue) · 4×3 confirmatory at 1.7B · judge probes | **$32** |
| **3 · Gallop** | ⛔ gated on Run exit | R3 · η evals · field report | **$29** |
| | | **Total** | **~$98** |

## Verified pricing (re-verify at first run of each backend)

| Backend | Rate | Verified |
|---|---|---|
| Modal H100 | **$3.949/h** | 2026-08-01 — modal.com/pricing |
| Modal A100-80GB | **$2.498/h** | 2026-08-01 — modal.com/pricing |
| Modal A100-40GB | **$2.099/h** | 2026-08-01 — modal.com/pricing (**2.6x** L4, not the 3-4x assumed) |
| Modal L4 | **$0.799/h** | 2026-08-01 — modal.com/pricing |
| Tinker — Qwen3-8B train | $0.40/M tok | 2026-07-11 (via waterline) — **re-verify**; a price step-up was flagged for 2026-07-17 |
| Tinker — Qwen3-8B sample | $0.40→$0.60/M tok | same |
| Prime Sprints free queue | **$0** | ⬤ **PROVEN** 2026-08-03 — 3 hosted GRPO runs, one to eval 0.9980 |
| RunPod Community **RTX A6000, 48 GB** | **$0.33/h** | ⬤ 2026-08-04, runpod.io/pricing — **2.4x cheaper than Modal L4 with 2x the memory** |
| RunPod Community RTX A5000, 24 GB | $0.16/h | ⬤ 2026-08-04 — 5x cheaper than Modal L4 |
| RunPod Community A100-80GB PCIe | $1.19/h | ⬤ 2026-08-04 |
| Vast.ai RTX 4090 (interruptible) | ~$0.29-0.50/h | ☐ secondary sources only — **and we cannot use spot: no mid-run checkpointing** |
| Verda (ex-DataCrunch) | from ~$0.17/h | ☐ not read first-hand |
| Haiku 4.5 / Sonnet | per Anthropic pricing, caching on | unverified |

## Pre-commit estimates

Any single run above **$15** requires a written estimate here **before** running.

| Date | Run description | Expected cost | Result enabled |
|---|---|---|---|
| 2026-08-01 | **Clean ladder rerun** — all 10 arms × seed 0, one SHA (`3f7035b`), one GPU (L4) | **$5.88** (7.4 h × $0.799, from measured per-arm wall clock) | An internally comparable ladder. The current one spans **three git SHAs and two GPU tiers**, and 4 arms ran `git_dirty` |
| 2026-08-01 | Ablation A paired fixed-policy probe — 3 seeds × {base, 50-step warmup} | **~$0.30** | A's replacement measurement; retires the falsified training-arm comparison |

**Why the rerun is not optional.** `run1` (A100, `a9322218`), `run7` (A100, `fe047ab5`) and the
ablations (L4, `54403b7f`, **dirty**) are not comparable to each other. Ablation A compared run1
against run2 across *different hardware and different code* — a second reason that comparison was
void, independent of the metric confound. `experiments/README.md` already says a run whose manifest
does not identify its code cannot enter analysis.

## Budget revision — recorded 2026-08-03, and still incomplete

The two 🛑 blocks above fired against the **$17 plan line** and have since been overtaken, but the
ledger was never updated to say so. Recording the state rather than leaving it contradicting itself:

- **The line was verbally raised** during the Phase 0.1 seed pass ("we can increase crawl's budget"),
  which is why the seed pass, the probe reruns and M1 all ran after the stop. **No revised figure was
  ever written down** — so `spend.md` still says $17 while ~$23 has been spent. That gap is the
  finding here; the fix needs a number from the user, not from me.
- **The real constraint is the ~$50 Modal credit pool**, not the plan line. The original alarm was
  against a planning estimate, not a wall.
- **Phase 0.2 cost $0** — three hosted runs on Prime's free tier — which recovered roughly the
  overrun the alarm was raised about.

> ✅ **RESOLVED 2026-08-04, then CORRECTED the same day. The Modal pool is ~$17, not ~$50.**
>
> I first recorded ~$50 from an earlier remark ("about $50 worth of modal credits"). **That is
> wrong** — the user corrected it: *"the modal credits are only ~$17 - the rest were eaten up by a
> different project."* The extension was authorised ("Crawl spending can be extended"); the pool is
> simply ~3x smaller than I assumed. Recording the error rather than overwriting it, because a
> budget figure I inferred and got wrong by 3x is exactly the kind of thing that should leave a
> trace.
>
> **CONFIRMED by the user 2026-08-04: ~$17 is the *current balance*** — available going forward,
> not a share already partly spent. The ambiguity flagged here earlier is closed.
>
> **What it changes, under either reading.** Modal spend to date is ~$14.50-15.50 (Phase 0.1
> ~$12-13, Phase 0.3 $2.21; Phase 0.2 was $0 on Prime's free tier and cost no credits).
>
> | | against ~$17 in hand |
> |---|---|
> | **R1** (0.4, $2 line) | comfortable |
> | **Walk** ($20 plan line) | **exceeds it** |
> | **Remaining science, costed bottom-up** | **~$145-410 — see `scripts/cost_model.py`** |
>
> **The mitigation already exists and is proven.** R1 is explicitly *"Prime Intellect 1B hacking
> (free queue if available)"*, and **Phase 0.2 ran three hosted GRPO runs on that queue for $0**,
> including a 200-step training run. `stages.md` already budgets Walk's exploratory grid at $0 there
> too. So the binding constraint on the next phase is **free-tier availability, not Modal credits** —
> which makes the free tier a single point of failure worth naming.
>
> The original $17 *plan line* and this ~$17 *pool* being near-equal is a coincidence, not a
> reconciliation. The two 🛑 blocks above fired against the plan line; they are historical.

## M1 — what $1.57 bought (2026-08-03)

R0's line is **$10**. The screen spent **$1.57** of it to find out that the other **$8.43** would have
bought an uninterpretable result.

| model | setting | pass@1 | **dead groups** | parse_fail | wrong_answer | med tok | verdict |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | cd-3 | 0.024 | **0.845** | 0.272 | 0.704 | 189 | starved |
| Qwen2.5-1.5B | cd-4 | 0.006 | **0.950** | 0.244 | 0.750 | 261 | starved |
| Qwen2.5-1.5B | cd-5 | 0.004 | **0.965** | 0.236 | 0.760 | 304 | starved |
| Qwen2.5-1.5B | cd-6 | 0.001 | **0.995** | 0.221 | 0.779 | 346 | starved |
| Qwen2.5-3B | cd-3 | 0.059 | **0.620** | 0.151 | 0.789 | 138 | marginal |
| Qwen2.5-3B | cd-4 | 0.009 | **0.925** | 0.170 | 0.821 | 189 | starved |
| Qwen2.5-3B | cd-5 | 0.007 | **0.950** | 0.171 | 0.822 | 184 | starved |
| Qwen2.5-3B | cd-6 | 0.006 | **0.955** | 0.168 | 0.826 | 202 | starved |

Pre-registered band: **≤ 0.50 workable · 0.50–0.75 marginal · > 0.75 starved**. **Nothing clears it at
either scale.** The cheapest cell in the grid — 3B on the easiest setting — is 0.620, and it is the
*most expensive* model we can afford.

**This is a result about the task, not a broken rig.** The pre-registered rig-broken branch fires at
`parse_fail > 0.5`; observed is 0.15–0.27, and `wrong_answer` carries 0.70–0.83. The models emit legal
arithmetic that misses the target — reasoning and failing, not failing to format.

**The prediction was free and it was right.** `dead = p⁸ + (1−p)⁸` at `G = 8` is Phase 0.1's own
task-selection criterion. Applied to the ~2% pass rate secondary sources report for 1.5B, it
predicted 0.851 before any GPU was booked; measured 0.845. `add-3digit` — the task that criterion
*chose* — sits at 0.012.

**Peak memory validated the L4 choice by measurement**: 5.93 GB (1.5B) and 9.79 GB (3B) against 24 GB.
No repeat of Phase 0.1's A100 tier-creep, which cost $4.26 for headroom never used.

**Spend impact:** the screen is *inside* R0's line, not additional to it. What it changes is that the
remaining $8.43 is not yet committed to anything, because R0's scale is now an open question rather
than a settled one.

## M2 — what ~$0.15 bought (2026-08-03)

**Verdict `not_free`**, and it un-cuts a rung of the ladder.

Per-token discrepancy between vLLM sampling and our HF scorer is *negligible* — median **−3e-6**,
p01/p99 at ∓0.12, mean −0.00106 nats. Over 512 tokens it compounds to a ±1σ sequence ratio of
**[0.218, 1.555]** against a pre-registered negligible band of **[0.9, 1.1]**.

The tempting reading — "vLLM over-scores its own samples" — is wrong, and the arithmetic says so.
The mean *must* be negative (it is `−KL(π_vLLM ‖ π_HF)`, Gibbs), and its size is *determined* by the
spread: a log-normal ratio with `E[ratio] = 1` forces `μ = −σ²/2`, and −0.001056 vs −0.000942 agrees
to 12%. So **the ratio stays unbiased in expectation and becomes useless per sequence** — the
textbook failure of importance sampling on long sequences, which is what clipping exists to contain.

Three rig checks passed before any of that was read: HF-vs-HF on the same token ids gave **exactly**
0.0, vLLM's pass@1 matched M1's HF sampler at **z = −0.64**, and `independence_ratio` came in at
**0.968**, which is what licenses extrapolating per-token spread to sequence length at all.

**Spend consequence.** This does **not** make R0 cheap. vLLM's speed is available only behind a loop
change (rung 4: a genuine importance weight and a clip) that Phase 0.1 deliberately did not make. The
cost objection is converted into a prerequisite, not removed. The speedup itself is **unmeasured** —
the 7-minute app included image pull, engine init, model load and two scoring passes, so no clean
throughput number exists and none is quoted.

**$0.15 against a $0.50 estimate.** The failed launch (vLLM V1 needs `nvcc`, which `debian_slim` does
not carry) cost ~4 minutes of L4.

## Compute options and the funding gap — 2026-08-04

`scripts/cost_model.py` puts the remaining science at **$147 / $549 / $1,654** (low/mid/high) against
**~$17** in hand. Full analysis in `docs/compute-options.md`. Three things worth carrying here:

1. **Price shopping cannot close a 32x gap.** The cheapest credible provider is 2-5x under Modal.
   The levers that can are **credit programmes** (up to the whole gap) and **scope** (2-4x).
2. **RunPod Community RTX A6000 is the best on-demand fit found: $0.33/h for 48 GB**, against Modal
   L4's $0.799 for 24 GB. That removes the tier-escalation risk the cost model had to assume.
3. **We cannot use spot/interruptible pricing**, where the headline savings are. Verified by
   inspection: the loop has **no mid-run checkpointing**, so a preemption loses the whole run — and
   the exposure grows with run length, which is the cost model's dominant term. Cheap spot carries an
   engineering prerequisite on `loop.py` (user's file, §7).

**Do not switch providers yet.** The dominant uncertainty is `bisect`'s episode length ($875 span),
unmeasurable until Phase 1.1. Optimising $/h against a 3x band is premature.

## Replan triggers

- A stage's actual spend exceeds its estimate by **>50%** → stop, re-estimate, and record the
  revision before continuing.
- Prime Sprints free queue unavailable → exploratory grid at 0.6B on Modal (~$14, weaker screen) **or**
  cut the grid 12 → 8. Decide before running, not during.
- Tinker credits land → Run and Gallop get materially cheaper; consider 8B confirmatory arms. Record
  the decision as a `docs/pre-registration.md` change-log entry, since it moves a design pin.
