# Compute spend tracking

Log every compute/API cost **at the time incurred**, not retrospectively (`docs/desiderata.md` §17).

**Spend is gated by stage.** A stage's budget is not committed until the prior stage's exit gate
passes. Only **Crawl (~$17)** is authorised right now.

**Target: ~$98 · Hard cap: $150.** At the 2-month cadence expect ~$120 — longer projects accrete
re-runs.

---

## Ledger

| Date | Stage | Phase | Run | Backend | Duration | Cost | Running total | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-07-27 | 0 · Crawl | 0.1 | Calibration sweep — 5 runs (n=4/16/16/64/200, k=4–8) | Modal L4 | ~45 min GPU | **~$0.60 (est.)** | ~$0.60 | ~19,000 completions ≤256 tok, Llama-3.2-1B. Pinned Phase 0.1's task. Still an estimate — the sweep predates the artifact volume, so there is no per-step wall clock to reconcile from. |
| 2026-07-28 | 0 · Crawl | 0.1 | Bringing up `HFPolicy`: ~12 invocations, mostly failures | Modal A10G → A100-40GB | ~1–1.5 h GPU + image builds | **~$2–4 (est.)** | ~$3–5 | 6 OOM/crash failures. **Cannot be reconciled** — a crashed run writes no artifact, so these leave no measurable trace. |
| 2026-07-28→08-01 | 0 · Crawl | 0.1 | **All 22 surviving runs** — LR probes ×7, overfit, run7 ×4, run1 ×2, run2 ×3, run3, ablations B/B-control/C/D | Modal A100-40GB + L4 | **8.0 h GPU, measured** from per-step wall clock + 2 min/run container overhead | **$8.41 (measured)** | ~$12–13 | A100-40GB $4.26 · L4 $4.15. Per-run breakdown regenerates from the volume. |

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
| Prime Sprints free queue | $0 (Llama-3.2-1B, validated submissions) | **unverified — check terms; moves $28** |
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

## Replan triggers

- A stage's actual spend exceeds its estimate by **>50%** → stop, re-estimate, and record the
  revision before continuing.
- Prime Sprints free queue unavailable → exploratory grid at 0.6B on Modal (~$14, weaker screen) **or**
  cut the grid 12 → 8. Decide before running, not during.
- Tinker credits land → Run and Gallop get materially cheaper; consider 8B confirmatory arms. Record
  the decision as a `docs/pre-registration.md` change-log entry, since it moves a design pin.
