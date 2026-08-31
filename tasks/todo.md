# todo — session entry point

> **Next session, start here.** Where the project is, the single next action, and the decisions
> already made so you don't relitigate them. **Update the STATUS + NEXT blocks at the end of each
> session** — this file went a full phase out of date between 08-04 and 08-24, which is exactly the
> failure `lessons.md` 2026-08-18 names, landing on the one file whose job is to be current.

## STATUS (as of 2026-08-31)

**Stage 0 · Crawl — 4 of 5 phases merged. `main` at `6b5a2a9`, 424 tests green.**
Phases 0.1–0.4 complete and merged (see the table further down for what each found).
**0.5 is active and unmerged** on `phase-0.5-literature-gate` — 9 commits, rebased on current `main`,
tree clean.

**Spend:** ~$25–27 of GPU credit, ~$15 remaining. ⚠️ **The Modal credits expired at end of August —
verify the balance before planning any paid run.**

### What happened since 2026-08-24

**A three-clause adversarial prior-art sweep** (3 launched reviewers + 2 orphaned sub-agents, all
delivered) **falsified the novelty claim for the fourth time** and forced four corrections, all
merged to `main`:

| | outcome |
|---|---|
| **Novelty claim** | **replaced.** Old version struck through in `docs/related-work.md`. Now two narrow conjunctions, each qualifier documented with the paper that forces it |
| **Framing sentence** | **was false.** *"Nobody can tell you an environment is good before spending the compute"* — Zhang `2607.11022` and Wen `2410.05584` do exactly that. Corrected in `README.md`, `CLAUDE.md` §0, `conceptual.md` |
| **L1's floor** | **derived, not asserted.** `−log ζ/k` (Wu et al. `2507.14843` Appx C.4). At k=64 the 95% floor is **0.047 — 3× above** the old `1/64`. `p_hack@k` re-pinned at **k = 512** |
| **`P-outcome-cheap`** | **settled** — mean gap over steps 50–200 as H1's scalar, slope beside it. ρ bands re-affirmed |
| **§4 recursion** | **narrowed.** R1 falsified base-rate→onset at small capability; it does not touch A1→gap at frontier capability, which is what H1 ranks |

**S1 — the substrate screen — ran twice on Modal and was REJECTED both times** (`$1.60` total; on
the 0.5 branch, `experiments/phase-0.5-substrate/results/S1-RESULT.md`). It falsified two of our own
hypotheses in a row and ended somewhere better:

- **S1a** killed "prose vs digits" — prose in the *question* still gave 9-token completions.
- **S1b** killed "length" — reasoning gave 160-token fluent prose and **still 0/1024 hack words**.
- **What binds is vocabulary breadth.** At a matched 3,802-token budget: story **1000** distinct word
  types, word-problem reasoning **188**. Verifiability *constrains* the output space; a lexical
  exploit *needs* it open. **The two are in tension by construction** — R1 lacked a truth signal for
  the same reason its exploit was reachable.
- **Consequence:** the hack-word model does not transfer to a verifiable task. `bisect`'s exploits
  must be **structural** (special-case, try/except, hardcode, edit the test) — which is what the
  README already names, so the design is sound. Only *lexical* exploits were screened; structural
  reachability remains untested and is `bisect`'s premise.

## NEXT — three things, in this order

### 1. Finish Clause 2 of the novelty claim ⚠️ needs a raised search budget

**`docs/related-work.md` records Clause 2 at *moderate* confidence and says explicitly it must not be
published as an unqualified "nobody has" until these close.** Clause 1 is high confidence and safe —
it rests on positive findings, which incomplete search can only strengthen.

What's done: keyword sweeps (TIER 1 empty), forward citations of Wen `2410.05584` (**33 papers, none
qualify**), forward citations of Mahmoud `2605.12474` (one flagged, **verified false positive**).

What's missing:
- **Gao `2210.10760`'s citation graph — unwalked.** The canonical overoptimization paper; anyone
  doing a proxy-vs-gold decomposition would plausibly cite it.
- **Non-arXiv venues** — OpenReview bot-blocks, ACL Anthology untouched. `djinn` is blog-only and was
  found by luck.
- **Keyword search structurally cannot find this claim** — the decomposition tends to be a secondary
  analysis inside a section. Autorubric `2603.00077` does its cross-judge probe in **Section 6**.
- Semantic Scholar concept search returned **HTTP 429**.

**Method that worked, and the one that didn't:** forward-citation walks on the S2 graph API are
unmetered and productive. Keyword search is the wrong instrument for a negative claim. **Cap
sub-agent fan-out** — uncapped fan-out burned all 200 WebSearch calls and stalled two agents.

### 2. Finish Block A — one paper left

**#2 Breaking Barriers is DONE** (`2506.19733`, read first-hand). It occupies **E1's headline** —
intra-domain transfer across independently-authored benchmarks — but **not E2**, because it swaps
whole benchmarks (tasks and grader together) and never holds tasks fixed. **E2 is now the
load-bearing leg of `endemic`, not E1.**

| # | paper | status |
|---|---|---|
| 1 | Fuzzing RLVR Verifiers `2606.01066` | ✅ read |
| 2 | Breaking Barriers `2506.19733` | ✅ read |
| 3 | Prime Sprints | ✅ read — **review file still owed**, material is in `phase-0.4-r1-plan.md` |
| 4 | Rollout Pass-Rate Control `2605.05112` | ☐ **partly pre-empted** — #13 owns A3 |
| 11 | Natural Emergent Misalignment `2511.18397` | ☐ **partly answered by R1** — 8–40 steps at 1B vs their ~1,500 |

**Newly required reads, surfaced by the sweep:** `2606.09711` (PRIME — read the full paper, not just
the abstract), `2606.16062` (Rajan/EQS), `2607.11022` (Zhang), `2605.02909` (COLM 2026 — **error
*pattern* not *rate* determines collapse**, which bears directly on the re-pinned band).

### 3. E3's window — still your call

**Not free**: each η sample point is **four** evaluation targets, so N points cost 4N passes per arm.
8 points = 8× the endpoint η; **5 points (0, 10, 40, 100, 200) = 5×**. The pre-registration's
*"Nearly free once the eval harness exists"* is wrong and needs correcting either way.

## ALSO OWED BEFORE THE NEXT STAGE

- [x] ~~**L1 redesign**~~ **SETTLED 2026-08-31** — floor is `−log ζ/k`, k re-pinned at 512.
- [x] ~~**The §4 contradiction**~~ **SETTLED 2026-08-31** — annotation narrowed; the recursion holds.
- [ ] ⚠️ **THE COST MODEL HAS NO TERM FOR GRADER EXECUTION, and on `bisect` that may dominate.**
      *(First flagged as a screen problem, which was the wrong number — corrected 2026-08-31.)*
      Every **training** rollout also needs a graded execution, because on `bisect` the reward *is*
      "does the test pass":

      | | executions, 12-variant grid |
      |---|---|
      | the screen at k=512 | 12,288 |
      | **training** | **2,457,600** — 200× more |

      The screen is **0.5% of one training run** and is not the problem. `scripts/cost_model.py`
      prices runs purely from GPU tokens (`run_cost(params_b, tokens, vllm_speedup, hourly)`) and has
      no execution term. `bisect` is the first substrate where a grade is a container plus a test
      suite rather than an integer compare — same architecture, different cost structure.

      **It all turns on one unmeasured number**, seconds per graded execution:

      | per execution | grid core-hours | at 32× parallel |
      |---|---|---|
      | 2s | 1,365 | 43 h |
      | 10s | 6,827 | 213 h |
      | 30s | 20,480 | 640 h |

      A **15× span**, against a Run line of **$32**. At the top of it, CPU beats GPU as the dominant
      cost. **Measure it first thing at Phase 1.1** — time one graded episode end to end (container
      start, suite, teardown); one number collapses the span. Then add
      `seconds_per_graded_execution` to `ASSUMPTIONS` with the same low/mid/high treatment as
      `bisect_tokens_per_episode`, so the sensitivity table shows which term actually dominates.
      **And treat grader throughput as a design constraint** — if it is ~30s, the grid's timeout axis
      is a budget lever, not just a pathology dial, and that must be known before the factorial is
      fixed.

- [ ] **Two screen-level savings, worth having but not urgent at 0.5%.** The grid varies the
      **grader**, not the tasks — so sample completions **once** and score them under all 12 graders
      (12× off sampling). And **short-circuit the conjunction**: `p_hack` needs proxy-passes AND
      true-fails, so run the cheap visible test first and only run the expensive held-out suite on
      the survivors. ⚠️ **Adaptive stopping is valid for "clearly above the band" and NOT for
      "below"** — concluding `p < 0.0059` needs the full k by `−log ζ/k`. That asymmetry is the L1
      error we just fixed and would be easy to reintroduce as an optimisation.
- [ ] **H4 is published** (Helff `2604.15149` runs our healthy-vs-degenerate contrast). Demote to a
      manipulation check.
- [ ] **H1 has three published comparators** — Wen τ = 0.66/0.47, Zhang ρ = 0.80, PRIME ρ = 0.87.
      Pre-register against them or H1 reads as replication.
- [ ] **`bisect`'s visible/hidden design is published as an eval** — SpecBench `2605.21384`. Cite it
      as the eval-side companion.
- [ ] **A lever for `bisect`'s reachability:** Countdown-Code `2603.07084` got a proxy–true gap in
      **under 100 steps** with **1% SFT contamination**. Stage-2 feasibility is mixed, not negative.
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
