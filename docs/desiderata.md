# Desiderata

Immutable principles for **assay**. Frozen at the end of Stage 0 (Crawl). Changes require an
explicit, dated entry in the relevant phase retro.

## 1. Python + uv
Python 3.11+. `uv` as package manager. No bare `pip`. `make check` (ruff + mypy-strict + pytest)
green before anything merges.

## 2. Pre-register, then run
Hypotheses, the primary metric per gate, and honest-null readings are written into
`pre-registration.md` **before** the corresponding experiment runs. A run that disconfirms its
driving hypothesis is a successful run. **The null-case abstract is written before Stage 2 starts.**

## 3. Research design is law
`conceptual.md` + `pre-registration.md` are the north star. Every experimental choice — the grader
factorial, the battery axes, the outcome variable, the model scale — lives there. `src/` is a tool,
never a decision-maker.

## 4. Gradients must flow
The project's binding constraint. Every stage from Crawl onward produces or consumes a real training
run. A stage that ships only analysis has not shipped.

## 5. Exploit reachability is screened, never assumed
No variant enters the grid without a measured `p_hack@64` inside the pre-registered admission band.
The positive control must hack before the grid runs. *Trigger: without this, a grid that produces no
hacking is a null for the wrong reason and costs a stage; with this, the failure surfaces in Phase
0.4 for ~$2.*

## 6. The held-out grader is a grader
It can be wrong. Every held-out grader is validated against a hand-labelled gold set, and its FP/FN
is reported alongside every result that depends on it. Never treated as ground truth by assertion.

## 7. Contamination-free ground truth
The "true" leg of any proxy–true gap must be a grader the policy was not optimized against
(SynthOracle-style), so a result is genuine and not in-distribution overfit.

## 8. Exploratory and confirmatory are declared in advance
Which variants carry which hypothesis is pre-registered. **Never promote an exploratory variant to
confirmatory after seeing its result.** Exploratory findings are labelled exploratory in every
artifact that reports them.

## 9. Report distributions, not point estimates
≥3 seeds on every headline arm. Every reported number carries uncertainty (seed bands, permutation
tests, confidence intervals). One PRIMARY metric per gate; the battery corroborates, never
substitutes.

## 10. Fitted is disclosed as fitted
`assay_score`'s axis weighting is fitted on the constructed family. Any artifact reporting the score
says so in the same breath. No implied out-of-sample validity that was not measured.

## 11. Independence is argued, not assumed
For the η decomposition, A and A′ must be independently authored. Hub-sourced pairs and
model-authored pairs are reported **separately, never pooled**; idiom correlation is measured and
disclosed for the model-authored ones.

## 12. Reproducibility
Pin model **revision hashes**, sampler settings (temperature, top-p), RL hyperparameters, seeds,
grader/judge model versions **and prompts**. Snapshot every dataset. Document what cannot be pinned.
Raw rollouts are never modified.

## 13. Reproducible numbers only
Every figure and paper number regenerates from a committed script reading committed results
(`experiments/<phase>/results/*.json`). No one-time scripts. Raw ignored, derived committed.

## 14. Reproductions retire named assumptions
Each entry in `reproductions/` states the assumption it retires. A reproduction that does not
load-bear on this project does not get run. **Verdicts include "failed to reproduce."**

## 15. Objective before subjective
Automated/quantitative scoring (held-out test pass, permutation-tested slopes) runs before any
LLM-as-judge or qualitative reading.

## 16. Borrowed is declared
Every borrowed component — `verifiers`, `prime-rl`, `trl`, Reasoning Gym, Hub environments — is named
in `README.md` with what it does. Contribution claims are stated against that list, not around it.

## 17. Track spend, gated by stage
Log all compute/API cost in `tasks/spend.md` at the time incurred. **A stage's budget is not
committed until the prior stage's gate passes.** Target ~$100; hard cap $150.

## 18. Git workflow
Phase branches off `main` (e.g. `phase-0.1-grpo-by-hand`). User merges manually. No force pushes.
Small, focused commits. **No AI attribution in any git artifact** — no `Co-Authored-By`, no
"Generated with" footers, no 🤖, in commits, PR bodies, or issues.

## 19. Documentation as code
`CLAUDE.md` updated at the end of each phase. Phase plans before implementation, retros after.
Everything in the repo.

## 20. Scope is defended, not negotiated
`endurance` is project #2 and does not move forward. The cut order in `stages.md` is decided in
advance and followed. Cutting is a decision made against a written list, never in the moment.
