# todo — session entry point

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. **Update the STATUS + NEXT blocks at the end of each
> session** — this file went a full phase out of date between 08-04 and 08-24, which is exactly the
> failure `lessons.md` 2026-08-18 names, landing on the one file whose job is to be current.

## STATUS (as of 2026-08-24)

**Stage 0 · Crawl — 4 of 5 phases complete. `main` is at `ea4c4ea`, 406 tests green.**

| phase | state | headline |
|---|---|---|
| **0.1** GRPO by hand | ✅ merged | 43% → **92%**. Four deliberate breakages. A degenerate grader gives a **52-point** proxy–true gap on demand — and the **KL leash made that gap wider** by 0.037 on 3/3 seeds while carrying 54% of the loss. |
| **0.2** ecosystem port | ✅ merged | Published to the Hub. An independent trainer's **first** measurement (58.8%) landed inside our band (57.1% ± 1.9%) — that, not the 99.8% endpoint, is the evidence the port is faithful. **$0**. |
| **0.3** R0 | ✅ merged | **Disqualified on the ledger's own rule** — no published number to reproduce. Three screens for **$2.21** of a $10 line. |
| **0.4** R1 | ✅ merged | **Reachability confirmed** (15/15 train, 12/15 eval; 8–40 steps; **$0**). **L1's lower bound disconfirmed.** R1-P **unresolved**, not falsified. |
| **0.5** literature gate | 🔄 **active**, unmerged | `phase-0.5-literature-gate`, 2 commits, rebased on `main`. **6/16 read** — Block A 2/5, Block D 4/4. |

**Spend:** ~$23–25 of GPU credit (nearly all in 0.1, which overran its $5 line), **~$17 remaining**.
0.2 and 0.4 cost `$0`. The ledger's running-total column is stale — see its top banner.

## NEXT — the pre-registration cannot lock, and Crawl cannot exit until it does

`docs/pre-registration.md` locks at the end of 0.5, once **(a)** the literature gate clears and
**(b)** R1 confirms reachability. **(b) is done.** Two things block **(a)**:

### 1. Finish Block A — three papers left, and one can kill a leg

**Block A is the must-read for Phase 0.5.** Nothing else is gating.

| # | paper | why it blocks | status |
|---|---|---|---|
| **2** | **Breaking Barriers** (ICLR 2026, uiuc-kang-lab) | **Read this first.** Does anyone hold *skill fixed, authorship varied*? **If that axis is occupied, `endemic` is dead and the project reverts to gap-only `assay`.** Sharper now: Block D's #14 found that **cutting η collapses the novelty margin**, so #2 and #14 squeeze from opposite sides. | ☐ |
| **11** | **Natural Emergent Misalignment** (2511.18397) | "No safe rarity threshold" vs our 200-step budget. **R1 has already part-answered it** — hacking at 8–40 steps at 1B, against their ~1,500. And its contingency (*"if it doesn't reconcile, L1's admission band needs redesign"*) **has already fired for an unrelated reason**. | ☐ |
| **4** | **Rollout Pass-Rate Control** (2605.05112) | **Partly pre-empted** — Block D's #13 (PROPEL) already owns the A3 pass-rate band, citing Wei et al. 2025. Reading #4 now confirms A3's demotion to cited-not-contributed rather than establishing it. Lowest stakes of the three. | ☐ |

Read: #1 ✅ · #3 ✅. Blocks B and C are Walk/Run-time, not gating. Block D (4 papers) is read but
**at Process step 1 only** — findings, not decisions.

**Two proposed additions, unadjudicated** (from the Phase 0.4 prior-art reviewer): `2606.16062`
(inference-only hackability audit of *real* code-RL environments, correlated to a downstream
outcome) and `2507.14843` (the support-constrained RLVR theory underneath `CLAUDE.md` §4's
"amplification, not discovery"). Both are novelty-perimeter candidates.

### 2. Three decisions that are the owner's under §7, and all are lock prerequisites

- [ ] **`P-outcome-cheap`'s estimand.** Under amendment — a slope alone can't separate "never hacked"
      from "hacked before step 50", and **H4 inverts under it**. Three candidates laid out in
      `pre-registration.md` §4 L3. H1's ρ bands were set against the slope and must be re-affirmed
      against the replacement **before** Run data exists.
- [ ] **E3's measurement window.** Unpinned entirely. Same underlying question — settle with the above.
- [ ] **The one-sentence novelty claim.** **Falsified on three independent readings** (#1, #13 §C5,
      #14 §C5). Must be replaced before the lock.

## ALSO OWED BEFORE THE NEXT STAGE

- [ ] **L1 redesign.** Two of three R1 variants sat *below* `1/64` and saturated anyway; `1/64` is the
      resolution floor of a 64-sample screen, not a reachability threshold. Measured false-negative
      rate **42–68%**. Fix: raise k (0.006 needs k ≳ 500), re-derive the bound from reachability,
      report the miss rate as a result. Cheap — base-policy sampling, no gradients.
- [ ] **The §4 contradiction.** §4 says the screen and the diagnostic are the same measurement; the
      new annotation says nothing may **rank** by base rate. But **H1's primary metric is a rank
      correlation** over `assay_score = f(A1..A6)`, and Phase 1.4 selects confirmatory variants by
      ranking predicted pathology. Both can't stand.
- [ ] **Block D's scope calls** — demote A3 to cited-not-contributed and promote H3 to load-bearing;
      promote #9 Block C → Block A; amend `stages.md`'s cut order (cutting η now costs novelty);
      and whether `canary`'s diversity battery takes the Vmax cross-stack study (§17-B handoff).

## DECISIONS ALREADY MADE (do not relitigate — the *why* is in `docs/conceptual.md`)

- **The project is `assay` + `endemic` merged, not either alone.** H3 and E2 are the same claim from
  two sides. Two-tier outcome: cheap gap on all variants, full η on 4 confirmatory arms.
- ⚠️ ~~**The outcome is a slope, not an endpoint.**~~ **SUPERSEDED 2026-08-15.** A slope-only estimand
  is blind to a variant saturated before the window opens. Becomes **level + slope**; estimand choice
  open (above). The *original* reasoning still holds for the hazard it was written against — a run
  truncated too slow — it was simply one-sided.
- **`bisect` is the substrate**, and the grid is *grader configurations over one task set*, so grader
  pathology is known by construction. **R1 added a hard requirement:** a substrate needs a reachable
  exploit **and** checkable ground truth *simultaneously*. Neither Crawl substrate managed both.
- ⚠️ **Reachability is screened, never assumed** — the principle stands, **the band does not.**
  `p_hack@64 ∈ [1/64, 0.30]`'s lower bound is disconfirmed (above). The screen *is* battery axis A1
  at a lower capability; that identity is now in tension with §4 (above).
- **R1 sat in Crawl on purpose** and it paid: the project's biggest risk was retired in week 1 for
  `$0`, and the screen's defect was found before any Run money was committed.
- **`endurance` (the NAND environment) is project #2** and does not move forward. Re-examined
  2026-08-04 for the grant proposal and declined again: `endemic` already carries the generalisation
  leg, `conceptual.md` calls `endurance` *"structurally identical"* to `bisect`, and two unbuilt
  environments in one proposal doubles the cost model's dominant uncertainty. **One sentence naming
  it as the sequel is the most it gets.**
- **Learning-first:** the user writes the GRPO loop, the grader factorial, the battery scoring, and
  **every hypothesis test**. Claude scaffolds plumbing. (`CLAUDE.md` §7.)
- **Model floor is 1.5B for anything requiring reasoning.** 0.6B is for plumbing smoke tests only.
- **Statistics pins earned in 0.4:** `P-alpha` = 0.05 family rate, α/2 per direction, with the exact
  **interval** reported — never a design floor. **≥4 seeds per arm for any directional claim** (the
  exact floor at 3v3 is 0.05, so n=3 can never clear it), and **seeds launched in one wave count as
  one draw**.

## READ ORDER for a cold start

1. **this file** — status + next action.
2. `docs/plain-english-summary.md` — what exists and what it found, no jargon.
3. `CLAUDE.md` — design + conventions (§7 learning-first, §17 sister projects, §15 gotchas).
4. `docs/pre-registration.md` — hypotheses, the ladder, **and the two live amendments**.
5. `docs/phases/phase-0.4-r1-retro.md` — the most recent phase, including where our own quality
   controls failed twice.

## OPEN THREADS

- [ ] **Tinker waitlist** — $150 credits would make Run and Gallop materially cheaper, possibly 8B
      confirmatory arms.
- [ ] **Pin the confirmatory model revision hash** (Phase 1.1). Candidate Qwen3-1.7B; **Prime hosted
      training offers Qwen3.5-2B at $0.15/1M train**, ~4–8× cheaper than Modal at our volumes.
- [ ] **Reconstruct Phase 0.1's missing ledger rows**, or mark the phase total an irrecoverable
      estimate. The running-total column reads ~$15.50 against ~$23–25 actually spent.
- [ ] **Ablation A wants ≥3 seeds per arm.** `run1` has 2, sitting at ρ≈0.04 with a 0.006 band. If the
      ratio lands inside the band the honest report is *"not measurable at this batch size"*, **not**
      "the baseline does not reduce variance".
- [ ] **Why `ocean` showed a +7-step batch shift** is attributed to rollout staleness by elimination,
      not measurement. Block D's #14 offers a testable rival: *generation difficulty*, not frequency.
- [x] ~~Free queue live? terms?~~ **RESOLVED** — live, `$0`, but **requires a PUBLIC environment**.
      Repo and both environments are public; 0.2 and 0.4 ran free on it.
- [x] ~~`verl` missing from §6~~ **RESOLVED at 0.3's `/learn`** — added as a candidate, not a switch.
- [x] ~~Literature gate 0/5~~ **superseded** — see NEXT.

## SPEND GATE

Stage budgets are **not** committed until the prior stage's gate passes (`docs/desiderata.md` §17).
Crawl's remaining **~$17** is the only authorised spend. Walk's line is **~$20** and is **not**
released until Crawl's exit gate — which is the pre-registration lock above.
