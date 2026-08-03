# TODO — session handoff (read me first)

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. Update the STATUS + NEXT blocks at the end of each
> session.

## STATUS (as of 2026-08-02 — seed pass running)

- **Phase 0.1 nearly complete** on `phase-0.1-grpo-by-hand`. `make check` green (262 tests).
- **All four breakages have clean results at n=1**, from the clean ladder of 2026-08-02: ten arms,
  one commit (`f1cc4048`), one GPU tier (L4), 200 steps each, 410 GPU-min, **$5.45**.
- **⏳ Seed pass running** (`4d4d64e`): seeds 1–2 on the 7 arms that carry a claimed *difference*.
  ~8.8 h, ~$7.05. Rungs 1–3 stay at n=1 by design — their dead-group numbers match the derivation
  exactly, so a seed band would only measure noise around a derived value.
- **Spend reconciled from measured wall clock: $8.41** at the time of reconciliation, rising to
  **~$21** with the clean ladder and probes, ~$28 once the seed pass lands. Modal rates verified:
  L4 $0.799/h, A100-40GB $2.099/h. User holds ~$50 of Modal credits, so the earlier
  "Crawl is over budget" alarm was against the *plan line*, not a wall.

### Results, one line each

| | verdict |
|---|---|
| **A** no baseline | **Falsified, then redesigned.** The training-arm comparison is structurally void (two confounds pointing opposite ways + non-comparable trajectories). Replaced by a paired fixed-policy probe: at the base policy, **no detectable variance reduction and `1/(1-p)` excluded on 3/3 seeds**. At a converged policy the probe is `underpowered` — GRPO starves its own gradient (NSR 0.37 → 55–83). |
| **B** no KL leash | **A finding, not the signature.** The degenerate grader is fully hacked (proxy 0.993 vs true 0.486, **gap +0.507**, the largest in the ladder), and removing the leash changes the gap by −0.018 — nothing — despite KL carrying 49% of the loss. B's mechanism runs through D's. |
| **C** tie-breaker | **Confirmed on all four signatures.** dead 0.468→0.008, tokens 17.2→35.5, true-reward gain +0.315→+0.130, gap +0.042. |
| **D** unanimous groups | **Confirmed exactly.** `frac_degenerate = 1.000` and `grad_norm = 0.0000` on all 200 steps; 313k tokens for zero gradient. |

**Length normalisation** breaks `E[grad log pi] = 0`, which is why A was unreadable — and removing it
improved NSR on every arm and final true reward on both arms tested. Two independent lines toward
Dr. GRPO, reached by measurement.

## WHAT REMAINS

1. **Wait for the seed pass**, then refresh every headline number to a 3-seed mean + band.
2. **`docs/tutorial/reinforce-to-grpo.tex`** (18 pp, builds clean) needs one consolidated revision:
   probe table is stale at N=40 (N=160 exists), the converged/`underpowered` result is missing
   entirely, and **no table states its `n`** — every number currently reads as settled when it is
   n=1. User is waiting on this to print it.
3. **Retro** + `/learn`, then merge (§13 needs both).
4. Optional: +$3.86 for seeds on rungs 1–3, which is what the unexplained *"a baseline tripled
   completion length"* observation (9.2 → 25.2 tokens) would need to be claimable rather than dropped.

## DECISIONS ALREADY MADE (do not relitigate — the *why* is in `docs/conceptual.md`)

- **The project is `assay` + `endemic` merged, not either alone.** H3 (grader degeneracy dominates
  the gap) and E2 (grader idiom > environment idiom) are the same claim from two sides. Two-tier
  outcome: cheap gap on all variants, full η on 4 confirmatory arms.
- **The outcome is a slope, not an endpoint.** `d(gap)/d(step)` over steps 50–200. A rising
  unsaturated gap is still a measurement.
- **`bisect` is the substrate**, and the variant grid is *grader configurations over one task set* —
  so grader pathology is known by construction.
- **Reachability is screened, never assumed.** `p_hack@64 ∈ [1/64, 0.30]`, plus a positive control
  that must hack. The screen *is* battery axis A1 at a lower capability — the mitigation is the
  experiment.
- **R1 sits in Crawl on purpose.** It is both a reproduction and the reachability gate, so the
  project's biggest risk surfaces in week 1 for ~$2 rather than in week 5 for a stage.
- **`endurance` (the NAND environment) is project #2** and does not move forward.
- **Learning-first:** user writes the GRPO loop, the grader factorial, the battery scoring, and every
  hypothesis test. Claude scaffolds plumbing. (`CLAUDE.md` §7.)
- **Model floor is 1.5B for anything requiring reasoning** — TinyZero found 0.5B fails on Countdown.
  0.6B is for plumbing smoke tests only.

## READ ORDER for a cold start

1. `CLAUDE.md` — the design + conventions.
2. `docs/conceptual.md` — the idea and why the two legs are one project.
3. `docs/stages.md` — Crawl/Walk/Run/Gallop, gates, cut order, budget.
4. `docs/pre-registration.md` — hypotheses + the reachability ladder.
5. `docs/phases/phase-0.1-grpo-by-hand-plan.md` — the active phase.

## OPEN THREADS

- [ ] **Literature gate (Phase 0.5) — 0/5 Block A.** Tier-1 threats first: **#1** (2606.01066, closest
      on the diagnostic leg), **#2** (Breaking Barriers ICLR 2026, closest on the η leg), **#3**
      (Prime Sprints, closest on the prediction leg). **If #2 turns out to occupy the skill-fixed /
      authorship-varied axis, `endemic` is dead and the project reverts to gap-only `assay`** —
      record that decision in the gate.
- [x] ~~**Prime Sprints free queue: live? terms?**~~ — **RESOLVED 2026-08-01.** The free compute
      is live (`sprints/Llama-3.2-1B-Instruct`, $0/$0/$0, operational) but the *sprint* closed
      ~2026-06-20 and no new track has been announced — **`CLAUDE.md` §15's "running now" is
      stale.** Requirement established empirically: the free tier **requires a PUBLIC
      environment** (ownership is fine, a fork was accepted; private was rejected). So the cost
      is publishing `bisect` before the paper. Decide at Phase 0.2. **Trap:** `prime-rl` ships a
      `zero_advantage` pre-batch filter ON BY DEFAULT — a Stage-2 grid there would silently
      filter away ablation D's pathology unless the filter list is overridden.
- [ ] **Tinker waitlist** — $150 credits would make Run and Gallop materially cheaper and could put
      the confirmatory arms at 8B.
- [ ] **Pin the confirmatory model revision hash** (Phase 1.1). Candidate was Qwen3-1.7B;
      **Prime hosted training now offers Qwen3.5-2B at $0.15/1M train**, which is a live
      alternative and ~4-8x cheaper than Modal at our measured token volumes.
- [x] ~~Phase 0.1 task choice~~ — **DONE 2026-07-27/28.** Chosen by a calibration sweep, not
      intuition, on `dead_group_fraction` rather than a mean. `arithmetic/add-3digit` is primary
      after a documented deviation; `add-2digit` is the robustness arm. The ">=5% at k=8" criterion
      in the original plan was superseded: it is stated on the wrong statistic and has no ceiling.
- [ ] **`#11` (Anthropic natural emergent misalignment) vs the `p_hack@64` screen.** Their "no safe
      rarity threshold" is at ~1,500+ steps; ours is a 200-step budget. Expected to reconcile — but
      **if it doesn't, L1's admission band needs redesign.**
- [x] ~~No remote yet~~ — **DONE.** `github.com/kar-ganap/assay`, private. `main` sits at the
      scaffold; work is on `phase-0.1-grpo-by-hand`, unmerged (§13: merge needs retro + `/learn`).
- [x] ~~**Reconcile spend**~~ — **DONE 2026-08-01** from measured per-step wall clock, not the
      dashboard: $8.41 across 22 runs, rates verified from modal.com/pricing.
- [ ] **Ablation A wants >=3 seeds per arm.** `run1` has 2. Its metric sits at rho~0.04 with a seed
      band of 0.006 — real but tight, because two half-batch gradients in a million-parameter space
      are nearly orthogonal. If the ratio lands inside the band, the honest report is "not
      measurable at this batch size", **not** "the baseline does not reduce variance".

## SPEND GATE

Stage budgets are **not** committed until the prior stage's gate passes (`docs/desiderata.md` §17).
Crawl's ~$17 is the only authorised spend right now.
