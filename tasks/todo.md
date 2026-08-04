# TODO — session handoff (read me first)

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. Update the STATUS + NEXT blocks at the end of each
> session.

## STATUS (as of 2026-08-03 — Phase 0.3 stage 1, M2 in flight)

- **Phase 0.1 complete and merged.** Hand-rolled GRPO, seven ablations, four breakage signatures.
- **Phase 0.2 complete, pushed, NOT merged** (`phase-0.2-ecosystem-idiom`). All five gates met at
  **$0** on Prime's free tier: env published as `gkartik/assay-add3digit`, an independent trainer
  reached eval **0.9980** against a gate of 0.85, and its step-1 eval (0.5879) landed inside Phase
  0.1 run 7's own band (0.571 +/- 0.019). `zero_advantage` confirmed on by default — a Stage-2 pin.
- **Phase 0.3 is stage 1 only: two measurements before committing to R0.**
  `phase-0.3-r0-countdown`, cut off 0.2, so **0.2 must land first**.

### M1 — DONE 2026-08-03. Verdict `starved`. $1.57 of R0's $10 line.

The Countdown base-rate screen, k=G=8, n=200, base checkpoints, 512 tok, L4:

| model | cd-3 | cd-4 | cd-5 | cd-6 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.845 | 0.950 | 0.965 | 0.995 |
| Qwen2.5-3B | **0.620** | 0.925 | 0.950 | 0.955 |

(`dead_group_fraction`; pre-registered band <=0.50 workable / 0.50-0.75 marginal / >0.75 starved.)

**Nothing clears the band at either scale.** Not a rig artifact — `parse_fail` is 0.15-0.27 against
its 0.5 threshold and `wrong_answer` carries 0.70-0.83. **Phase 0.1's own `p^8 + (1-p)^8` criterion
predicted 0.851 before a GPU was booked; measured 0.845.** L4 validated by measurement (5.9 / 9.8 GB
peak). Artifacts in `experiments/phase-0.3-r0/results/` — **read `PROVENANCE.md` first**, the two
JSONs carry a wrong `model_id` from a dict-spread bug, corrected there.

### M2 — running

vLLM vs HF per-token log-probs on the same token ids. Band `[0.9, 1.1]` on the implied sequence
ratio at L=512. Pure core in `crawl/mismatch.py` (zero GPU, 20 tests); Modal side is
`measure_mismatch`.

## WHAT REMAINS

1. **M2's result** against its band, then log spend.
2. **Decide what R0 becomes** — *the user's call, not a technical one.* Four options are written up
   in `docs/phases/phase-0.3-r0-plan.md`: (a) run at 3B/cd-3 as an explicitly handicapped
   reproduction; (b) add a curriculum or shaped reward, which changes what is reproduced;
   (c) substitute a different R0 target; (d) accept `add-3digit` plus Phase 0.2's
   independent-trainer cross-check as sufficient and retire R0.
3. **Merge 0.2**, then 0.3's retro + `/learn`.
4. **`mismatch_verdict` wants a review pass** (§7): I drafted the verdict function; the band itself
   is the user's from the plan. It recomputes from the committed artifact, so the run does not
   foreclose changing it.

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
- [ ] **`verl` is missing from `CLAUDE.md` §6's backend list, and was never evaluated.** Read
      first-hand 2026-08-03 (`docs/related-work.md` has the verified section). §6 says `prime-rl` or
      `trl`; verl appeared exactly once in this repo, describing TinyZero's stack. It implements
      **DrGRPO** — independently where Phase 0.1's length-normalisation finding pointed — plus GSPO,
      DAPO, and the `rollout_is_*` surface that M2's result now calls for. Against it: Phase 0.2
      validated `prime-rl` at **$0** on the free tier; verl's examples assume 64xH800.
      **Proposed §6 amendment, to be decided at Phase 0.3's `/learn`** — add verl as a candidate,
      not a switch. Note also the repo moved: `volcengine/verl` -> **`verl-project/verl`**.
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
