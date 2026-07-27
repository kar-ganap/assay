# TODO — session handoff (read me first)

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. Update the STATUS + NEXT blocks at the end of each
> session.

## STATUS (as of 2026-07-26)

- **Scaffold complete.** Repo initialised at `random_projects/assay`, `main`, no remote yet.
- `make check` **green** (ruff clean · mypy --strict clean on 10 files · 20 tests pass). Branch renamed
  `master` → `main` to match `CLAUDE.md` §13.
- `src/assay/` is **typed stubs** (`NotImplementedError`). No compute path implemented.
- `docs/pre-registration.md` = **DRAFT, not locked.**
- `docs/related-work.md` = **UNVERIFIED** — one LLM-assisted research pass, 0/12 papers read
  first-hand.
- `literature-review/README.md` gate = **0/5 Block-A papers read.**
- `tasks/spend.md` = **$0.**
- Origin plan (the reasoning behind all of this): `../explore/rl-envs-onramp.md`.

## NEXT ACTION (the one thing to do)

**Phase 0.1** (`docs/phases/phase-0.1-grpo-by-hand-plan.md`): pick the task, measure the base
policy's pass rate (require ≥5% at k=8 before committing), cut `phase-0.1-grpo-by-hand`, and write the
loop.

*(Scaffold is green — `make check` passes as of 2026-07-26. No longer a blocker.)*

**In parallel, cheap and unblocking:** apply for **Tinker credits** ($150 on waitlist clearance) and
**check whether the Prime Sprints free queue is live and on what terms** — that second one moves $28
of budget and determines whether Stage 2's exploratory grid is free.

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
