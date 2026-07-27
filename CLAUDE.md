# CLAUDE.md — assay project conventions

> Entry point for any Claude Code (or other agent) session on this repository. Internal-voice.
> Humans should start with `README.md`.
>
> **Start here (cold session):** `tasks/todo.md` — status, the single next action, and the decisions
> already made. Then `docs/conceptual.md` (the idea) and `docs/stages.md` (where we are on the
> ladder). Memory is directory-scoped: a session opened here does **not** load the mats or waterline
> project memory — this repo carries its own context.

## 0. One-paragraph pitch

RL environments are the current bottleneck in agentic AI, and the unsolved part is not the training
algorithm — it is **environment quality**. Practitioners report reward hacking as their #1 problem,
verifiers are wrong in both directions, and **nobody can tell you an environment is good before
spending the compute** (~$2,400 per task, per Epoch's practitioner interviews). `assay` asks whether
a **zero-GPU-hour diagnostic battery, run purely at inference time, predicts whether an RL
environment will teach the skill or teach the exploit** — and validates the answer against real GRPO
runs on a purpose-built environment (`bisect`). The bet: **a frontier model is a cheap forecaster of
a small policy's RL endpoint.**

## 1. Why this exists (portfolio + strategic context)

**The through-line of the portfolio is the gap between a target and the proxy we optimize or
measure.** SynthOracle is the measurement side (contamination-free oracles; articulation vs.
behavior). *Labels Not Loss* is the optimization side (directional Goodhart). waterline is the
during-training threshold. **`assay` is the pre-flight check** — the same gap, moved to the artifact
the industry is currently buying.

**The gap it closes.** Every prior project in this portfolio is inference-time plus statistics. **No
gradients have ever flowed.** For RL-environment startups and frontier labs, that is the whole
question. Hence the binding constraint:

> **This project must produce a training curve I own.** Everything else is negotiable.

## 2. The scientific question + pre-registered hypotheses

**Does a zero-GPU-hour pathology score predict an RL environment's post-training outcome?**
Full pre-registration (gates, honest nulls, design pins) in `docs/pre-registration.md`. Headlines:

- **H1 (prediction).** Rank correlation between `assay_score` and the measured post-GRPO
  proxy–true gap. Pre-commit: **ρ ≥ 0.6 works · 0.3 ≤ ρ < 0.6 partial · ρ < 0.3 honest null, shipped.**
- **H2 (mechanism).** Frontier-discovered exploits at step 0 predict *which* exploit the small policy
  converges to, versus a uniform-over-observed-exploits baseline.
- **H3 (decomposition).** **Grader degeneracy (A2) dominates the gap; pass-rate band (A3) predicts
  learning *speed* but not the *gap*.** Practitioners conflate these.
- **H4 (control).** A healthy variant matched on A3 but with a non-degenerate grader shows no gap.
- **E1–E3 (transfer efficiency).** η = G_skill / G_total. η ≤ 0.7 is the interesting regime;
  **η > 0.9 is an honest null worth publishing.** E2: grader idiom > environment idiom. E3: η falls
  with training step.

**H3 and E2 are the same claim reached from two sides.** That is why `assay` and `endemic` are one
project, not two.

## 3. Design in one screen

- **Substrate.** `bisect` — root-cause debugging under a query budget. A program fails a test; the
  agent spends query budget on instrumentation and subset runs; it must produce a fix.
  **Visible grader (proxy):** the failing test passes. **Hidden grader (true):** a held-out suite
  exercising the same root cause through other code paths.
- **The grid.** 8 (core) or 12 (extended) *grader configurations* over one task set:
  test visibility × reward shape × timeout × sandbox writability.
- **The battery.** A1 hackability · A2 grader degeneracy · A3 pass-rate band · A4 judge instability ·
  A5 verifier asymmetry · A6 contamination. **A1–A3 are MUST; A4–A6 are Gallop.**
- **Outcome, two-tier.** Cheap: proxy-reward minus held-out-reward, measured as a **slope**
  `d(gap)/d(step)` over steps 50–200. Expensive: the full **η decomposition** on 4 confirmatory arms.
- **Loop.** GRPO at 1.5–1.7B. Exploratory grid on the Prime Sprints free queue (Llama-3.2-1B) where
  available; confirmatory arms on Modal.

## 4. The reachability problem — read before designing any experiment

At 100–200 steps, **reward hacking is amplification of what is already in the policy's support, not
discovery.** Prime Intellect reproduced hacking at 1B in 100 steps for ~$0.64 because their exploits
were reachable; Anthropic needed ~1,500 steps because theirs were not.

**So exploit reachability is the binding variable, and it is free to measure.** Before any training,
per variant, sample k=64 rollouts from the *base* policy at T=1.0:

```
p_hack@64 = fraction scoring ≥ τ on the training grader AND failing the held-out grader
```

Admission band, pre-registered: **`p_hack@64 ∈ [1/64, 0.30]`**.

> **The recursion is the point.** `p_hack@64` on the base policy *is* battery axis A1 run at the
> small model's capability instead of the frontier model's. The de-risking screen and the diagnostic
> are the same measurement at two capability levels — and the difference between them is exactly
> what H2 is about. **The mitigation is not a patch on the experiment; it is the experiment.**

Full mitigation ladder (L1–L6) and the kill-switch: `docs/pre-registration.md` §4.

## 5. Stage model — Crawl · Walk · Run · Gallop

**Every stage is independently shippable.** Stop after any one and you have a complete artifact and
a defensible claim. Gates, not dates, decide when to move. Full detail: `docs/stages.md`.

| Stage | Phases | Thesis | Ship-if-you-stop | Hours | Spend |
|---|---|---|---|---|---|
| **0 · Crawl** | 0.1–0.5 | *Gradients flow, and I can prove it.* | GRPO from scratch, seven ablations, two reproductions | 20–28 h | ~$17 |
| **1 · Walk** | 1.1–1.6 | *I can build an environment and diagnose it.* | `bisect` on the Hub + `assay` v0 CLI | 25–35 h | ~$20 |
| **2 · Run** | 2.1–2.5 | *The diagnostic predicts the training outcome.* | The money figure + a paper | 30–40 h | ~$32 |
| **3 · Gallop** | 3.1–3.6 | *…and it predicts what you actually bought.* | η table + Hub field report | 30–40 h | ~$29 |

~124 h total → **~31 days at 4 h/day, ~62 days at 2 h/day.**

## 6. Tools + backend policy

- **Training.** Crawl: hand-rolled GRPO on `torch` + `transformers` (the point is the mechanics).
  Walk onward: `verifiers` environment spec + `prime-rl` or `trl` `GRPOTrainer` + vLLM.
- **Compute.** Prime Sprints free queue (Llama-3.2-1B) for the exploratory grid where available;
  Modal for confirmatory arms (H100 $3.95/h, A100-80GB ~$3.20/h). Tinker ($150 credits on waitlist
  clearance, Qwen3-8B train $0.40/M tokens) is the fallback that removes infra work — **apply in
  Phase 0.5.**
- **Frontier exploit-finder.** Haiku 4.5 for bulk, Sonnet for a spot tier, prompt caching on.
- **Borrowed, not built** — declare in the README: `verifiers` / `prime-rl` / `trl` (the stack),
  Reasoning Gym (the `endemic` substrate), existing Hub environments (A′ pairs).
- **Reuse from siblings:** `../synthoracle` (contamination-free oracle, articulation-vs-behavior),
  `../crit-thinking` (judge pipeline, position-bias correction → battery axis A4),
  `../opensource_x` (directional Goodhart).

## 7. Learning-first contract

**This project exists so I learn RL environments hands-on.** Claude scaffolds plumbing; **I write the
load-bearing parts.** Specifically:

- **I write:** the GRPO loop and its ablations (Phase 0.1), the grader-variant factorial and its
  pathology design (Phase 1.2), the battery probes' scoring logic (Phase 1.3), every hypothesis test.
- **Claude scaffolds:** repo plumbing, CLI, HTML report rendering, Modal wiring, data plumbing, test
  harnesses, literature reviews, figure scripts.

*Trigger: without this, the fastest path is Claude writing the RL loop and me reading it — which
produces a repo but not the intuition the project exists to build; with this, the artifact and the
learning are the same work.*

When in doubt about which side a task falls on, **ask**.

## 8. Workflow rules

1. **Plan mode for any 3+ step / architectural task.** Code work defaults to plan mode. Literature
   interpretation may stay conversational.
2. **TDD.** Code: failing tests first. Research: falsifiable hypothesis + evaluation criteria before
   running. Tests/gates define "done."
3. **Only plan the current phase in detail.** Future phases stay headline-level — anything else is
   waterfall in disguise.
4. **Verification before "done."** "Would a staff engineer approve this?"
5. **Objective before subjective.** Automated/quantitative checks before qualitative review.
6. **Subagents liberally.** One task per subagent. Keep main context clean.
7. **Autonomous bug fixing.** Just fix it. No context switch back to the user.

## 9. Code rules

- **Always `uv`, never `pip`.**
- **Simplicity first.** Minimal code, minimal blast radius. No over-engineering.
- **No laziness.** Root causes only. No temporary fixes. Senior-developer standards.
- **Reproducibility.** Pin model revision hashes, sampler settings, RL hyperparameters, seeds, and
  grader/judge versions + prompts. Raw rollouts are never modified.
- **Stash-based bug-fix proof** (from ccupa): stash fix → confirm tests fail → pop stash → confirm
  tests pass. Verifies the test catches the bug, not just that the bug is gone.
- **Secrets strictly from `.env`, never the shell.** Never `load_dotenv()` into the process env; read
  `.env` directly so a shell `ANTHROPIC_API_KEY` can never silently win. **`.env` holds a
  project-scoped personal key; never a work or shared key.** *Trigger: without this, a work account
  gets billed for personal experiments (it happened in `../agentic_engg`, lessons §0.10); with this,
  the project can only ever use the key explicitly placed in `.env`.*
- **Demand elegance (balanced).** Pause on non-trivial changes. Skip for simple fixes.

## 10. Experimental discipline

1. **Pre-register hypotheses + gates before running.** `docs/pre-registration.md` locks before the
   first paid run. A run that disconfirms its driving hypothesis is a successful run.
2. **Report nulls honestly.** H1 failing is a shipped result. **Write the null-case abstract before
   Stage 2 starts.**
3. **Characterize distributions, not means.** ≥3 seeds on every headline arm; seed-variance band
   reported beside every effect size.
4. **Exploratory vs confirmatory is declared in advance.** **Never promote an exploratory variant to
   confirmatory after seeing its result.**
5. **Numbers regenerate from committed scripts.** No one-time scripts.
6. **Track spend** in `tasks/spend.md` at time of incurring, not retrospectively. **Spend is gated by
   stage** — do not commit a stage's budget until the prior gate passes.

## 11. Two-stream discipline (problem vs. process)

**Do not conflate the two streams.** Different lifecycles, different destinations.

- **Problem stream → paper raw material.** `docs/conceptual.md`, `docs/related-work.md`,
  `literature-review/`, phase retros (`docs/phases/phase-X.Y-retro.md`), `reproductions/`. Each retro
  anchors to its eventual paper section (→ Methods / Results / Discussion / Related Work /
  Limitations).
- **Process stream → discipline that outlives this project.** `tasks/lessons.md`. Rule changes,
  where discipline slipped and why, tooling friction, permission-allowlist candidates.

Cross-references allowed; content separated.

## 12. Validation gates (per phase, mandatory)

1. **Tests pass** (code phases) or **all evaluation criteria met** (research phases).
2. **Lint / typecheck clean:** `make check` (ruff + mypy --strict + pytest).
3. **Reproducibility:** results regenerate from committed code + pinned parameters.
4. **Retro written** with paper-section anchors.
5. **`/learn` written** with a mandatory `[DELETE]` section (empty deletions allowed but must be
   explicit, not omitted).
6. **Three-reviewer parallel pass at critical stage boundaries** — clean-context Opus subagents,
   given the conceptual doc + phase artifacts only: **method-rigor** (does the method work?),
   **framing-stress** (does the framing survive this stage's evidence?), **prior-art-coverage**
   (would a domain reviewer raise an uncited reference class?). Confidence filter: report only issues
   at ≥80; lower flags go to an appendix. *Critical boundaries here:* Crawl→Walk (the R1
   reachability gate), Walk→Run (the screen + kill-switch), Run→Gallop (the H1 result).

## 13. Git lifecycle

- **Phase branches** off `main` (e.g. `phase-0.1-grpo-by-hand`). Stage-0 scaffold work stays on
  `main`. **User merges manually. No force pushes.**
- **No AI attribution anywhere in git artifacts** — no `Co-Authored-By` lines, no "Generated with
  Claude Code" / 🤖 footers, no tool self-attribution in commits, PR bodies, or issues. *Overrides
  any harness default that appends such footers.*
- **Small, focused commits.** Branch naming `phase-X.Y-<2-3-word-desc>`.
- **Prerequisite enforcement:** no commit on a phase branch without a written plan it references; no
  merge to `main` without retro + `/learn` (+ three-reviewer pass at critical boundaries).
- **Model tiering** for subagents: **Opus** for reviewers, **Sonnet** for task agents/architects,
  **Haiku** for test runners and mechanical checks.

## 14. Meta-rules (governance of the rule set)

**Rule-admission test.** Every proposed rule must carry a one-line **trigger statement**: *"Without
this, X; with this, Y."* If `Y == X`, it is decoration — move it to background context. Vague rules
("be rigorous") fail; specific operationalizations ("screen `p_hack@64` before admitting a variant")
pass.

**Phase-completion `/learn` ritual.** At every phase-complete declaration, before merge: what worked
(specific) · what caused friction · rule changes `[ADD]`/`[MODIFY]`/**`[DELETE]`** (deletion
mandatory to consider) · conceptual-doc cleanup proposed · tool/permission allowlist additions.

## 15. Known gotchas

- **Nothing hacks at 1.7B in 200 steps.** The project's biggest risk. Mitigated in six layers
  (`docs/pre-registration.md` §4) and moved into Phase 0.4 (R1) so it surfaces in week 1, not week 5.
- **Your held-out graders are graders too.** They can be wrong. Hand-label a gold set; report
  held-out-grader FP/FN alongside every result. Phase 1.6 gate.
- **A/A′ independence is arguable.** If you author both, your idiom leaks into both and η inflates.
  Hub-sourced and model-authored pairs reported **separately, never pooled**.
- **Grid creep.** The second most likely way this goes over budget. Pre-commit the grid size.
- **`endurance` scope pull.** The NAND environment is **project #2**. Pulling it into Gallop is the
  single most likely scope failure. It does not move forward.
- **`assay_score` weighting is fitted, not derived.** Say so every time it is reported.
- **Framing path-dependence** (inherited from epibench). Once a framing is established, reasoning
  accumulates support rather than scrutinizing it. Counter: the framing-stress reviewer at stage
  boundaries.
- **The window is narrow and closing.** Prime Sprints is running a reward-hacking track *now*; the
  fuzzing-verifiers paper landed June 2026; Apollo's Science-of-Scheming stream starts Sept 2026 and
  targets adjacent territory. **Speed is a design constraint, not a preference.** Concretely: do not
  let the literature gate expand past the five Block-A papers, and do not let Walk expand past the
  8-variant core grid on a first pass. If the field publishes the zero-step prediction question
  before Run lands, the fallback is the η leg (`endemic`), which is less contested.

## 16. Key references (in-repo)

| Path | Role |
|---|---|
| `tasks/todo.md` | **Session entry point** — status + the single next action |
| `docs/conceptual.md` | The idea: the two contributions and why they are one project |
| `docs/stages.md` | Crawl/Walk/Run/Gallop — phases, gates, ship-if-you-stop, cut order |
| `docs/pre-registration.md` | H1–H4, E1–E3, the reachability ladder, gates, design pins |
| `docs/desiderata.md` | Immutable principles |
| `docs/related-work.md` | The differentiation table — **UNVERIFIED until Phase 0.5** |
| `docs/process.md` | Phase lifecycle |
| `docs/phases/` | Plans, retros, `/learn` outputs |
| `literature-review/README.md` | Reading plan + the Crawl decision gate |
| `reproductions/README.md` | R0–R4 ledger; each retires one named assumption |
| `experiments/README.md` | raw/ ignored, results/ committed |
| `tasks/lessons.md` | Process stream (live) |
| `tasks/spend.md` | Spend, gated by stage |

**Sister projects for conventions (not content):** `../waterline` · `../epibench` ·
`../agentic_engg` · `../synthoracle` · `../crit-thinking`.
**Origin plan:** `../explore/rl-envs-onramp.md`.

## 17. If continuing after a gap

1. Read `tasks/lessons.md` (process state, carry-forward rules).
2. Read `tasks/todo.md` (status + next action + decisions already made).
3. Read the most recent `docs/phases/phase-X.Y-retro.md` (problem state).
4. Check `docs/stages.md` for the active stage's gate.
5. *Then* propose the next step.
