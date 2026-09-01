# Literature review — reading plan + decision gate

First-hand engagement following the program's paper-engagement house style (`../../epibench` is the
most evolved instance; also `../../waterline`, `../../synthoracle`). The reading list is seeded from
`../docs/related-work.md` — but **nothing here is "read" until a first-hand ⬤ pass clears it.**
LLM-surfaced IDs and claims can be wrong.

> **Status — 2026-08-06. 6 / 16 read first-hand** — Block A **2 / 5** (#1, #3) and Block D **4 / 4**
> (#13–#16, the Vmax stack). The other 10 remain unread.
>
> **Block A** — read in-project, at full Process depth:
> - **#1 fuzzing-RLVR-verifiers** ⬤ read 2026-08-04 — `01-fuzzing-rlvr-verifiers-2606-review.md`.
>   Every *quantitative* claim in `related-work.md` verified exact. Two *qualitative* claims wrong,
>   one of which **falsifies the current novelty claim** (they do stake a pre-training-diagnostic
>   claim). A revised claim is proposed in that review and **awaits the user's decision**.
> - **#3 Prime Sprints reward-hacking** ⬤ read 2026-08-04, twice and independently, during Phase 0.4
>   — see `../docs/phases/phase-0.4-r1-plan.md`. Publishes a reproducible steps-to-saturation table;
>   free queue verified live. Its own review file is still to be written up from that material.
>
> **Block D** — drafted by an **out-of-project session** (interview prep) at **Process step 1 only**.
> Findings and proposals, not decisions; nothing below has been adjudicated in-project.
>
> **The one-sentence novelty claim in `../docs/related-work.md` is falsified as written** — now on
> three independent readings (#1, #13 §C5, #14 §C5) — and must be replaced before
> `pre-registration.md` locks.
>
> **Block D added 2026-08-06 (recursive novelty-perimeter rule).** Researching Vmax surfaced three
> papers by one six-author team that were absent from the original list and sit closer to `assay`
> than anything on it. Headline consequences:
> - **#14 states `assay`'s thesis verbatim** — *"Both check solvability without checking
>   non-triviality"* — about named systems, with measurements. Best motivating citation available,
>   and the most serious novelty threat. **Survives on generality + validation target only.**
> - **#13 owns the A3 axis** (pass-rate band, [1/8, 3/8], citing Wei et al. 2025) and independently
>   validates `assay`'s *genre*: a cheap pre-training statistic (RVP) predicts RL gain better than
>   held-out accuracy. Demote A3 to cited-not-contributed; promote **H3 to the load-bearing claim.**
> - **#9 Endless Terminals is now a measured failure exhibit** (17.4% portability; 99% hardcoded
>   assertions) → **promote Block C → Block A.**
> - **Cutting η now collapses the novelty margin** (#14 C5). Amend the §8 cut-order in `docs/stages.md`.
> - **#16 names `assay`'s question as its own open problem.** The frozen-probe defence that carries
>   #13's entire reward is validated once, in a narrow action space, and its authors state: *"it is
>   plausible that with further optimization this changes, but further work is needed to ascertain
>   the (in)efficacy of this mitigation."* Read with #13's policy-drift limitation, the two jointly
>   assert a threshold exists and leave it unlocated. **ID lesson:** #13 cites it only by framework
>   name (RLFR); the actual title is *"Features as Rewards…"* — the ⬤ rule earned its keep.
> - **Vmax has four disjoint collapse-measurement stacks across three papers, none cross-validated**,
>   while their own hiring call asks for *"normative baselines for measuring the quality of RL
>   environments."* `canary`'s battery addresses this at zero GPU cost — likely the fastest external
>   artefact available (#15 C2, Discussion).

## Process (one paper at a time)

1. Claude writes a structured review → `NN-slug-YYYY-review.md` (template below); PDF beside it.
2. You read the paper + the review.
3. Interactive Q&A → captured in the review's *Discussion Notes*.
4. Claude quizzes + challenges (comprehension + Challenge Corner).
5. Update `synthesis-engagement.md` (not any Claude-authored synthesis).
6. Mark complete in Status.

**Recursive novelty-perimeter rule:** if a first-hand read surfaces closer prior art — especially
anything threatening novelty — add it with the **next number** (don't renumber), a dated note on why,
and require a first-hand read. This field moved fast in H1 2026; assume more will surface.

### Per-paper review template

`## Background` · `## Key Ideas` (numbered mechanisms) · `## Results` at three levels (smart
high-schooler / undergrad / early grad) · `## Key Quotes` (verbatim) · `## Study Questions` ·
`## Challenge Corner` (adversarial) · `## Connection to Project` (differentiation table; what to
adopt; what to differentiate for reviewers) · `## Discussion Notes` (filled during step 3).

## Clusters (the roles papers play for assay)

- **K1 · The gap / motivation:** 6, 9
- **K2 · Closest prior art / scoop (novelty lives here):** 1, 2, 3
- **K3 · The battery's axes (adopt + attribute):** 4, 5
- **K4 · Reachability (drives the L1–L6 ladder):** 11
- **K5 · Borrowed substrates + reproduction targets:** 7, 8
- **K6 · Scope boundaries (cited to bound, not to build on):** 10, 12

## Reading list

### Block A — Pivotal. **Read before the pre-registration locks (Phase 0.5).**

| # | K | Paper | id | Key question for us | Status |
|---|---|---|---|---|---|
| 1 | K2 | **Before the Model Learns the Bug: Fuzzing RLVR Verifiers** | 2606.01066 | Confirm it really uses **injected** bugs and **never trains**. If it trains, our differentiation collapses. | ⬤ **READ 2026-08-04** — injected ✅; "never trains" ✗ (tabular policy gradients, no neural fine-tune). Differentiation survives but **moves from the question to the validation**. |
| 2 | K2 | **Breaking Barriers: Do RL Post-Training Gains Transfer To Unseen Domains?** | ICLR 2026, uiuc-kang-lab | Does anyone hold **skill fixed, authorship varied**? This is the closest threat to `endemic`. | ☐ unread |
| 3 | K2 | **Prime Sprints — reward-hacking track** | primeintellect.ai/blog/reward-hacking | Is "predict onset from 20 steps" already answered? Is the free queue live, and on what terms? | ⬤ **READ 2026-08-04** (x2, independently) — publishes a steps-to-saturation table; **free queue verified LIVE**. Review file pending. |
| 4 | K3 | **Rollout Pass-Rate Control** | 2605.05112 | Exact form of the p≈0.5 result, so A3 cites rather than re-derives. Does it *also* claim the gap? | ☐ unread |
| 11 | K4 | **Natural Emergent Misalignment from Reward Hacking** | 2511.18397 | The step-1,500 emergence and "no safe rarity threshold." Does it invalidate the `p_hack@64` screen? | ☐ unread |

**Gate deliverable (Block A):** *state assay's one-sentence novelty claim so it survives a first-hand
read of #1 and #2, confirm #3 has not answered the zero-step question, and confirm #11 does not
invalidate the reachability screen.* Checklist below.

### Block B — Important (during Walk / Run)

| # | K | Paper | id | Role | Status |
|---|---|---|---|---|---|
| 5 | K3 | HUD — verifier & reward design | hud.ai/resources | practitioner failure-mode taxonomy → axis definitions | ☐ unread |
| 6 | K1 | Epoch AI — RL environments FAQ | epoch.ai/gradient-updates/state-of-rl-envs | the economics; source of the $2,400/task figure the project is justified against | ☐ unread |
| 7 | K5 | Reasoning Gym | 2505.24760 | borrowed η substrate + reproduction R3 | ☐ unread |
| 8 | K5 | GEM: A Gym for Agentic LLMs | 2510.01051 | interface norms | ☐ unread |

### Block D — additions 2026-08-06 (Vmax program; recursive novelty-perimeter rule)

*Surfaced while researching Vmax (`vmax.ai`), whose "MTS, Automated Environment Design" call names
"establish normative baselines for measuring the quality of RL environments" as a responsibility.
All three are by the same six-author team. IDs ⬤-verified 2026-08-06 by reading each PDF directly;
all three **read first-hand** the same day — the only papers on this list that have been.*

| # | K | Paper | id | Role | Status |
|---|---|-------|----|------|--------|
| 13 | K2/K3 | **PROPEL — Breaking the Solver Bottleneck** | 2606.18284 | Closest *methodological* neighbour: probe predicts solver pass-rate, replaces rollouts as generator-RL reward. Owns A3. Validates the genre (RVP). Its SWE label has **no held-out grader** — the opening. | ⬤ **read** `13-…` |
| 14 | K2/K5 | **unix-ctf — Procedural Environments** | 2605.29115 | States `assay`'s thesis verbatim; bidirectional contract = hand-built A1+A2 for one domain; mechanical 99%/0% hardcoded-assertion count = a free A2 sub-probe; reproduces + measures #9. | ⬤ **read** `14-…` |
| 15 | K2/K6 | **PopuLoRA — Co-Evolving LLM Populations** | 2605.16727 | Structural (vs authored) grader degeneracy; the reward–objective dissociation quote; the fourth disjoint collapse-metric stack. No novelty threat. | ⬤ **read** `15-…` |
| 16 | K2/K4 | **Features as Rewards** (Prasad, Watts et al., **Goodfire AI**) | 2602.10067 | The upstream dependency #13 inherits wholesale: invents the frozen-probe-as-reward recipe. Frames its own risk as **reward-model non-identifiability under limited coverage** (explicitly inverse-RL). **Names the pressure-threshold question as open in limitations.** | ⬤ **read** `16-…` |

**Open implications raised by Block D — RAISED, NOT DECIDED.** These were surfaced by an
out-of-project session (interview prep, 2026-08-06) and are recorded as findings only. Scope,
roadmap, gate, and pre-registration consequences are the project owner's call with full context;
nothing below has been applied, and no decision in `docs/` or `tasks/` was changed.

- The novelty sentence in `../docs/related-work.md` — "nobody can tell you an environment is good
  before spending the compute" — appears **falsified as a genre claim** by #13, which does forecast
  one property of a task cheaply and validates it. A narrower claim keyed to the *quantity forecast*
  looks defensible. Adjudication owed.
- **Pass-rate band discrepancy:** A3 is currently "distance from p=0.5"; #13 targets **[1/8, 3/8]**
  citing Wei et al. 2025. Analytically resolvable — see #13 Study Q4.
- **A2 measurement:** #14's mechanical hardcoded-assertion count separated 99%/0% where their judged
  rubric managed +0.90 on the same construct. Bears on whether A2 should lead mechanical or judged.
- **Structural vs authored degeneracy** (#15): A2 currently covers authored pathology only. Whether
  structural belongs in scope, as an axis or a named exclusion, is open.
- **η's role in the novelty margin** (#14 C5): the differentiation from unix-ctf appears to rest
  partly on the η leg, which §8's cut-order currently permits dropping. Worth the owner's attention.
- **`assay_score` trivial baselines** (#15 C1): #15's operator controls are a model for pre-registering
  uniform-weight and best-single-axis comparisons.

### Block C — Supporting (intro / related work)

| # | K | Paper | id | Role | Status |
|---|---|---|---|---|---|
| 9 | K1 | Endless Terminals | 2601.16443 | env supply at scale; a consumer of `assay`, not a competitor. **New evidence 2026-08-06 (from #14, not from reading #9):** a controlled n=120 reproduction measured 17.4% portability, and 99% of its tasks carry ≥1 hardcoded exact-string assertion. *Whether this changes #9's block/role is for the owner to triage.* | ☐ unread |
| 10 | K6 | Environment Scaling survey | 2511.09586 | GEF-loop taxonomy; check it doesn't already name our question as open | ☐ unread |
| 12 | K6 | Credit-assignment survey | 2604.09459 | cited to bound scope — what we deliberately are not doing | ☐ unread |

## Decision gate (Block A) — clear before the pre-registration locks

*Marker: `[~]` = review drafted (step 1), pending read + Q&A (steps 2–5) before it can be `[x]`.*

- [ ] **All Block-A IDs verified first-hand (⬤ id).**
- [ ] **Novelty survives #1 (Fuzzing RLVR Verifiers)** — confirm injected-bugs + no-training, so
      "natural defects + closed loop to GRPO" is genuinely open.
- [ ] **Novelty survives #2 (Breaking Barriers)** — confirm the skill-fixed / authorship-varied axis
      is unoccupied. **If it is occupied, `endemic` is dead and the project reverts to gap-only
      `assay`** — record that decision here.
- [ ] **#3 has not answered the zero-step prediction question**, and the free-queue terms are known
      (this moves $28 of budget).
- [ ] **#11 does not invalidate the `p_hack@64` screen** — reconcile "no safe rarity threshold" with
      a 200-step budget. Expected reconciliation: their result is at ~1,500+ steps, so reachability
      still binds at ours. **If it does not reconcile, L1's admission band needs redesign.**
- [ ] **Gate deliverable met:** the one-sentence novelty claim survives, or the design is adjusted
      and the change logged in `../docs/pre-registration.md`.

## Resolved concerns (do not relitigate)

*(empty — populated as the gate closes)*
