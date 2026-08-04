# Literature review — reading plan + decision gate

First-hand engagement following the program's paper-engagement house style (`../../epibench` is the
most evolved instance; also `../../waterline`, `../../synthoracle`). The reading list is seeded from
`../docs/related-work.md` — but **nothing here is "read" until a first-hand ⬤ pass clears it.**
LLM-surfaced IDs and claims can be wrong.

> **Status — 2026-08-04. Block A in progress: 2 / 5 read first-hand (2 / 12 overall).**
>
> - **#1 fuzzing-RLVR-verifiers** ⬤ read — `01-fuzzing-rlvr-verifiers-2606-review.md`. Every
>   *quantitative* claim in `related-work.md` verified exact. Two *qualitative* claims wrong, one of
>   which **falsifies the current novelty claim** (they do stake a pre-training-diagnostic claim).
>   A revised claim is proposed in that review and **awaits the user's decision**.
> - **#3 Prime Sprints reward-hacking** ⬤ read 2026-08-04, twice and independently, during Phase 0.4
>   — see `../docs/phases/phase-0.4-r1-plan.md`. Publishes a reproducible steps-to-saturation table;
>   free queue verified live. Its own review file is still to be written up from that material.
>
> The one-sentence novelty claim in `../docs/related-work.md` is **currently falsified as written**
> and must be replaced before `pre-registration.md` locks.

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

### Block C — Supporting (intro / related work)

| # | K | Paper | id | Role | Status |
|---|---|---|---|---|---|
| 9 | K1 | Endless Terminals | 2601.16443 | env supply at scale; a consumer of `assay`, not a competitor | ☐ unread |
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
