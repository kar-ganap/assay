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
| 2026-07-27 | 0 · Crawl | 0.1 | Calibration sweep — 5 runs (n=4/16/16/64/200, k=4–8) | Modal L4 | ~45 min GPU | **~$0.60 (est.)** | ~$0.60 | ~19,000 completions ≤256 tok, Llama-3.2-1B. **Estimate — confirm against the Modal dashboard and replace.** Pinned Phase 0.1's task. |

**Running total: ~$0.60 (estimated, unconfirmed)** of Crawl's authorised $17.

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
| Modal H100 | $3.95/h | 2026-07-12 (via `../waterline/tasks/spend.md`) — **re-verify** |
| Modal A100-80GB | ~$3.20/h | unverified |
| Modal L4 | **unverified** | used for the Phase 0.1 calibration sweep — **look up and record** |
| Tinker — Qwen3-8B train | $0.40/M tok | 2026-07-11 (via waterline) — **re-verify**; a price step-up was flagged for 2026-07-17 |
| Tinker — Qwen3-8B sample | $0.40→$0.60/M tok | same |
| Prime Sprints free queue | $0 (Llama-3.2-1B, validated submissions) | **unverified — check terms; moves $28** |
| Haiku 4.5 / Sonnet | per Anthropic pricing, caching on | unverified |

## Pre-commit estimates

Any single run above **$15** requires a written estimate here **before** running.

| Date | Run description | Expected cost | Result enabled |
|---|---|---|---|
| — | *(none yet)* | — | — |

## Replan triggers

- A stage's actual spend exceeds its estimate by **>50%** → stop, re-estimate, and record the
  revision before continuing.
- Prime Sprints free queue unavailable → exploratory grid at 0.6B on Modal (~$14, weaker screen) **or**
  cut the grid 12 → 8. Decide before running, not during.
- Tinker credits land → Run and Gallop get materially cheaper; consider 8B confirmatory arms. Record
  the decision as a `docs/pre-registration.md` change-log entry, since it moves a design pin.
