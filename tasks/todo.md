# TODO — session handoff (read me first)

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. Update the STATUS + NEXT blocks at the end of each
> session.

## STATUS (as of 2026-07-29 — PAUSED on compute budget)

- **Phase 0.1 in progress** on `phase-0.1-grpo-by-hand`, 29 commits, `make check` green (229 tests).
- **⛔ PAUSED: Modal budget exhausted (~$0.18 left).** Waiting for the monthly quota reset.
  `tasks/spend.md`'s **>50% replan trigger has fired** — Phase 0.1's $5 line is at ~$8–13 of Crawl's
  $17, with R0 ($10) and R1 ($2) still owed and both never-cut. **Reconcile against the Modal
  dashboard before resuming.**
- **GATE 1 PASSED.** Run 7 on `add-3digit`, 2 seeds, 200 steps: reward 0.438→0.873 and 0.406→0.905
  against the screen's base rate 0.433. Gain **+0.456 vs a seed band of 0.032 — 14× the band**, and
  `live_fraction_in_slope_window = 1.00` on both seeds.
- **Primary arm swapped** (documented deviation): `add-3digit`, not the rule's `add-2digit`, which
  saturates to ~100% dead groups within ten steps. `learning_rate = 1e-5`, `kl_coef = 0.04`.
- **14 runs recovered locally** into `experiments/phase-0.1-grpo-by-hand/`; all also live on the
  Modal volume `assay-phase01`. Nothing is at risk.
- **`TRAIN_GPU` is back to L4** — measured peak is 13.5–14.5 GB, so A100-40GB was never needed.

## WHAT REMAINS (≈1.3 h GPU ≈ $1 on L4)

8 ladder entries with no `add-3digit` run: `run2`, `run3`, `ablation_b`, `ablation_b_control`,
`ablation_c`, `ablation_c_nolennorm`, `ablation_d`, `run7_nolennorm`.

```
modal run --detach src/assay/modal_app.py::ladder --runs run2 --seeds 0,1,2   # completes ablation A
modal run --detach src/assay/modal_app.py::ladder --runs run3,ablation_b,ablation_b_control,ablation_c,ablation_c_nolennorm,ablation_d,run7_nolennorm
modal run src/assay/modal_app.py::fetch                                        # land the artifacts
```

`run1 × 2 seeds` is already done, so **ablation A needs only `run2`.** It wants ≥3 seeds per arm: its
metric (half-batch gradient cosine) sits at ρ≈0.04 with a seed band of 0.006 — measurable but tight,
because two half-batch gradients in a million-parameter space are nearly orthogonal.

**Doable now, with zero compute:** figures from the 6 completed 200-step runs · the
`add-2digit` vs `add-3digit` saturation comparison (a real A3 finding, with curves) · seed-variance
section · retro · `/learn`.

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
- [ ] **Prime Sprints free queue: live? terms?** Moves $28 and decides the Stage-2 grid model.
- [ ] **Tinker waitlist** — $150 credits would make Run and Gallop materially cheaper and could put
      the confirmatory arms at 8B.
- [ ] **Pin the confirmatory model revision hash** (Phase 1.1). Candidate: Qwen3-1.7B.
- [ ] **Phase 0.1 task choice** — needs base pass rate ≥5% at k=8. Candidates: 3-digit arithmetic
      with regex extraction; strict output-format; an easy Reasoning Gym family.
- [ ] **`#11` (Anthropic natural emergent misalignment) vs the `p_hack@64` screen.** Their "no safe
      rarity threshold" is at ~1,500+ steps; ours is a 200-step budget. Expected to reconcile — but
      **if it doesn't, L1's admission band needs redesign.**
- [ ] No remote yet. Create `github.com/kar-ganap/assay` (private) when Phase 0.1 lands.

## SPEND GATE

Stage budgets are **not** committed until the prior stage's gate passes (`docs/desiderata.md` §17).
Crawl's ~$17 is the only authorised spend right now.
