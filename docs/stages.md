# Stages — Crawl · Walk · Run · Gallop

**Design principle: every stage is independently shippable.** Stop after any one and you have a
complete artifact and a claim you can defend. Nothing is a half-built bridge.

**Span:** ~124 h → **~31 days at 4 h/day, ~62 days at 2 h/day.** **Gates, not dates**, decide when to
move. **Spend is gated too** — do not commit a stage's budget until the prior gate passes.

| Stage | Phases | Thesis | Ship-if-you-stop | Hours | Spend |
|---|---|---|---|---|---|
| **0 · Crawl** | 0.1–0.5 | *Gradients flow, and I can prove it.* | Repo + writeup: GRPO from scratch, seven ablations, two reproductions | 20–28 h | ~$17 |
| **1 · Walk** | 1.1–1.6 | *I can build an environment and diagnose it.* | `bisect` on the Hub + `assay` v0 CLI | 25–35 h | ~$20 |
| **2 · Run** | 2.1–2.5 | *The diagnostic predicts the training outcome.* | The money figure + a paper | 30–40 h | ~$32 |
| **3 · Gallop** | 3.1–3.6 | *…and it predicts what you actually bought.* | η table + Hub field report | 30–40 h | ~$29 |

**What each stage proves, to whom.** Crawl closes the portfolio's gradient gap. Walk *is* the RL-env
engineer job description, near-verbatim. Run is the frontier-lab research-engineer case. Gallop is
the product layer — understanding what buyers of environments actually need.

---

## Stage 0 · CRAWL — gradients flow

**Thesis:** *I can train a model with RL and explain every term in the objective.*

| Phase | Work |
|---|---|
| **0.1 · GRPO by hand** | Implement REINFORCE → +baseline → +group baseline (GRPO) → +clip → +KL → +advantage-norm on a 0.6B model and a task where the base policy already has nonzero pass rate. **Then break each one deliberately:** remove the baseline (variance explodes), remove KL (entropy collapse to a single string), normalize advantages inside a degenerate group (divide-by-~0 spikes), make the group all-correct (zero gradient, wasted step). One notebook, seven curves, a paragraph each on *why*. **User writes this** (`CLAUDE.md` §7). |
| **0.2 · The ecosystem idiom** | Rebuild the same task as a `verifiers` environment — `load_environment`, dataset, `Rubric` with reward functions, `SingleTurnEnv`. Run under `prime-rl`, push to the Hub. **DONE 2026-08-03** (`gkartik/assay-add3digit`). API verified against source: verifiers 0.2.1 ships **both** v0 and v1; v1 splits the Environment into TaskSet/Task/**Harness**, and its ten shipped harnesses show the purpose — one task set under many agent scaffolds. Degenerate for a single-turn task, so **v0 was chosen**, on evidence: `vf-init` ships only a v0 template, every published env is v0, and `v1/legacy.py` bridges v0 over the same v1 protocol so nothing is foreclosed. **The v0/v1 decision is revisited at 1.1**, when `bisect` brings tools, a sandbox and timeouts — and note two of the grid's four axes (timeout, sandbox writability) are harness concerns. `ToolEnv` and the `OpenEnv` RFC deferred there with it. |
| **0.3 · R0** | TinyZero / Countdown at **1.5B**. Retires: *"my loop actually learns."* |
| **0.4 · R1** | Prime Intellect 1B reward-hacking repro (Llama-3.2-1B-Instruct, 100 steps, batch 128, lr 1e-4). Retires: *"small models hack inside my step budget."* **This is the reachability gate, moved to week 1 on purpose.** |
| **0.5 · Gate + lock** | Verify `related-work.md` first-hand (literature gate). Lock `pre-registration.md`. Apply for Tinker credits. Freeze `desiderata.md`. |

**Exit gate.** A GRPO run that visibly learns · seven ablation curves you can explain · **R1
reproduces hacking** · whitespace survives first-hand · pre-registration locked.

**If the gate fails.** R1 not reproducing is the *informative* failure: small-model hacking is harder
to elicit than published. That undercuts `assay`'s premise but **not** `endemic`'s. **Pivot to the
η-only project** (Walk still builds `bisect`; Run measures transfer efficiency; no hacking required)
and report the R1 failure as its own finding.

---

## Stage 1 · WALK — build and diagnose

**Thesis:** *I can build an environment, and I can tell you what is wrong with it before you train.*

| Phase | Work |
|---|---|
| **1.1 · `bisect`** | ~40 seeded bugs in small pure-Python programs, subprocess + timeout, query-budget accounting, visible grader, hidden held-out suite. **No Docker.** Pin the confirmatory model revision hash. |
| **1.2 · The variant grid** | 8 (core) / 12 (extended) grader configurations over the one task set: test visibility × reward shape × timeout × sandbox writability. Pin τ. **User designs the pathologies.** |
| **1.3 · The battery** | A1 hackability · A2 grader degeneracy · A3 pass-rate band. Freeze the A1 adversarial prompt verbatim. **User writes the scoring logic.** |
| **1.4 · Screen** | `p_hack@64` on every variant; apply the admission band; run the **positive control**; apply the kill-switch. Choose and record the 4 confirmatory variants — **on predicted pathology, before any training.** |
| **1.5 · R2** | Fuzzing-verifiers repro — **no GPU**. Retires: *"adversarial probing finds grader bugs"* = A1 + A5. |
| **1.6 · Gold sets** | Hand-label ~40 trajectories per variant to validate the **held-out graders themselves**. |

**Exit gate.** `assay` produces a report card · **held-out graders validated against gold sets** (the
graders must not be the bug) · ≥8 variants pass the screen · **positive control hacks by step 200.**

**If the gate fails.** <8 variants pass → the grid is mis-designed; redesign **here** using L4 (raise
magnitude) and L5 (seed the base rate) rather than burning Run. Positive control doesn't hack → stop
and fix the rig.

---

## Stage 2 · RUN — does the diagnostic predict?

**Thesis:** *A zero-GPU-hour score predicts what RL will do to the policy.*

| Phase | Work |
|---|---|
| **2.1 · Exploratory** | 8–12 variants × 1 seed, **per-step gap logging**. Prime Sprints free queue where available. Reported as exploratory. |
| **2.2 · Confirmatory** | 4 variants × 3 seeds at 1.7B. Carries H1 / H3 / H4. |
| **2.3 · Analysis** | H1 (ρ vs pre-committed bands) · H2 (frontier exploit → converged exploit vs uniform baseline) · H3 (partial R² per axis, gap *and* speed) · H4 (healthy control). |
| **2.4 · R4** | Pass-rate-band claim, from runs already in hand. **$0.** |
| **2.5 · Seed-variance section** | *"How many seeds does it take to detect the effect sizes the RL-environments field routinely claims?"* Costs nothing; the seeds are already run. |

**Exit gate.** Every number regenerates from a committed script · nulls reported as nulls ·
seed-variance bands on every effect size · **the null-case abstract was written before the runs.**

**If the result is null.** H1 failing is a shipped result — "frontier models find exploits small
policies never reach" is capability-dependent discovery, the same shape as SynthOracle's scaffolding
cliff. H2 and H3 still carry.

---

## Stage 3 · GALLOP — what did you actually buy?

**Thesis:** *The diagnostic predicts transfer efficiency, not just reward hacking — and the field's
environments are worse than it thinks.*

| Phase | Work |
|---|---|
| **3.1 · A′ sourcing** | Two existing Hub environments for the same skill by different authors (headline anchor) + frontier-model-authored pairs from a shared spec, no cross-visibility (for n). **Report separately; never pool.** Measure and disclose idiom correlation for the authored pairs. |
| **3.2 · R3** | Reasoning Gym transfer repro. Retires: *"my eval harness measures transfer."* Published config is 3B — run at 1.7B and **report the scale delta rather than claiming a match.** Pin the external benchmark for eval (d). |
| **3.3 · η decomposition** | Evals (a)–(d) on the 4 confirmatory arms. E1 · E2 · E3. Inference-only. |
| **3.4 · A4 / A5 / A6** | Judge instability (port the crit-thinking position-bias machinery) · verifier asymmetry · contamination probe. Refit `assay_score`; **report the refit as a refit.** |
| **3.5 · Field report** | Run `assay` over ~15 Hub environments. **Frame as a contribution to the commons** — credit authors, offer PRs, not a takedown. |
| **3.6 · Ship** | CLI polish · HTML report card · `pip install`-able. arXiv + RLEval-style venue + Alignment Forum. Contribute `bisect` to the Hub. |

**Exit gate.** Someone else can `pip install assay`, point it at their environment, and get a report
card. The paper's headline claim is about **η**, not just the hacking gap.

---

## Sequencing rules

1. **Gates, not dates.** A stage ends when its gate passes. Three weeks instead of two means the
   project is 2 months, not broken.
2. **Never cut, at any stage:** R0 · R1 · the base-rate screen · the positive control · per-step gap
   logging · the confirmatory seeds.
3. **Cut order within a stage** (decided in advance, followed without renegotiation):
   **A6 → field report → A4/A5 → grid 12→8 → η on 4 arms → η entirely.** If η goes, the project
   reverts to gap-only `assay` and stays publishable — that is the whole reason the outcome variable
   is two-tier.
4. **Never promote an exploratory variant to confirmatory after seeing its result.**
5. **`endurance` does not move forward.** It is project #2. Pulling it into Gallop is the single most
   likely scope failure.
6. **Log spend at time of incurring.** Grid creep is the second most likely failure.
7. **Write the null-case abstract before Stage 2 starts.**

---

## Budget by stage (~$98; hard cap $150)

| Stage | Line | Cost |
|---|---|---|
| **Crawl** | Phase 0.1 plumbing smoke (0.6B, easy task) | $5 |
| | R0 — TinyZero/Countdown at 1.5B | $10 |
| | R1 — Prime Intellect 1B hacking (free queue if available) | $2 |
| | **subtotal** | **$17** |
| **Walk** | R2 — fuzzing repro (no GPU) | $10 |
| | Base-rate screen + positive control | $4 |
| | A1/A2 frontier exploit-finder (Haiku bulk + Sonnet spot, caching on) | $6 |
| | **subtotal** | **$20** |
| **Run** | Exploratory grid, 8–12 × 1 seed — **Prime Sprints free queue** | $0 |
| | Confirmatory: 4 × 3 seeds at 1.7B (~0.6 h × $3.95/h) | $28 |
| | Judge probes | $4 |
| | **subtotal** | **$32** |
| **Gallop** | R3 — Reasoning Gym transfer | $12 |
| | η evaluations, (a)–(d) × 4 arms (inference) | $12 |
| | Field report over ~15 Hub environments (Haiku only) | $5 |
| | **subtotal** | **$29** |
| | **Total** | **~$98** |

**The trade.** Reproductions eat ~a third of the budget, paid for by moving the exploratory grid onto
the free queue. Good trade — the free queue is a 1B Llama, the *right* model for a screen and the
wrong one for a headline, so the split falls along the scientifically correct line anyway.

**Contingencies.** Free queue unavailable → exploratory grid at 0.6B on Modal (~$14, weaker screen)
or cut 12 → 8. Tinker credits land ($150, Qwen3-8B train $0.40/M tokens) → Run and Gallop get
materially cheaper; consider 8B for confirmatory arms. Modal reference: H100 $3.95/h, A100-80GB
~$3.20/h.

**Headroom.** At the 2-month cadence, expect **$120** rather than $98 — longer projects accrete
re-runs. Hard cap $150 (desideratum 17).
