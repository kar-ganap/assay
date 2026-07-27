# Phase 0.1 — GRPO by hand

**Stage:** 0 (Crawl) · **Branch:** `phase-0.1-grpo-by-hand` · **Status:** PLANNED, not started
**Est:** 8–10 h · **Est spend:** ~$5

---

## Purpose

Build the mechanical intuition the rest of the project rests on: **implement policy-gradient RL on a
language model from scratch, then break each component deliberately and watch it fail.**

This is the phase that closes the portfolio's gradient gap (`../conceptual.md` §1, `CLAUDE.md` §1).
It is explicitly a **learning-first** phase — **the user writes the loop** (`CLAUDE.md` §7). Claude
scaffolds the plumbing: data loading, logging, plotting, the Modal wrapper, the test harness.

**Non-goal:** performance. Nothing here needs to be fast, general, or reusable. Phase 0.2 rebuilds
the same task properly under the `verifiers` spec.

---

## Design

**Model.** Qwen3-0.6B (or Llama-3.2-1B-Instruct). Pin the revision hash in `manifest.json`.
**Explicitly not** attempting reasoning emergence at this scale — TinyZero found Qwen2.5-0.5B *fails*
to learn Countdown. That is R0's job at 1.5B, in Phase 0.3.

**Task.** Something where the base policy already has **nonzero pass rate**, so a gradient exists on
step 1. Candidates: 3-digit arithmetic with a regex-extracted answer; a strict output-format task
(emit `<answer>N</answer>`); a Reasoning Gym family filtered to easy. Pick one in TEST, record why.

**Reward.** Binary, programmatic, sub-millisecond. No LLM judge anywhere in this phase.

## The ladder — seven runs

| # | Configuration | Expected observation |
|---|---|---|
| 1 | **REINFORCE**, no baseline | learns, but gradient variance is high; reward curve is noisy |
| 2 | **+ mean baseline** | variance drops visibly; same asymptote, faster |
| 3 | **+ group baseline (GRPO)** | leave-one-out within a group of G rollouts per prompt |
| 4 | **+ ratio clipping** | stability under multiple epochs per batch |
| 5 | **+ KL to reference** | entropy stays up; outputs stay diverse |
| 6 | **+ advantage normalization** | faster convergence — *and* the pathology in ablation D |
| 7 | **Full GRPO** | the reference curve everything else is compared against |

## The four deliberate breakages — the actual point of the phase

| Ablation | What is removed / forced | Prediction |
|---|---|---|
| **A · no baseline** | run 1 vs run 2 | gradient-norm variance ≫; slower, noisier convergence |
| **B · no KL** | run 7 minus the KL term | **entropy collapse** — the policy degenerates toward a single high-reward string |
| **C · degenerate-group normalization** | advantage-normalize a group whose rewards are near-identical | **divide-by-≈0 advantage spikes**; loss instability |
| **D · all-correct group** | force a group where every rollout succeeds | **zero gradient** — the step is wasted. This is the mechanism behind battery axis A3 and the pass-rate band. |

**Ablation D is the conceptual bridge to the whole project.** It is why pass-rate band is an axis at
all, and why `Rollout Pass-Rate Control` is cited rather than re-derived.

## Instrumentation (log every step)

reward (mean, per-group) · advantage magnitude · gradient norm · policy entropy · KL to reference ·
group pass rate · fraction of groups that are degenerate (all-0 or all-1) · tokens/step · wall-clock.

Written to `experiments/phase-0.1-grpo-by-hand/raw/<run>/steps.jsonl`; derived curves to
`results/*.json`.

---

## Gates (pass criteria)

1. **Run 7 visibly learns** — reward rises above the base-policy rate by a margin outside the
   seed band (≥2 seeds on run 7 alone).
2. **All four ablations reproduce their predicted failure**, each visible in a committed figure.
   *An ablation that does **not** fail as predicted is a finding — record it in the retro and in
   `tasks/lessons.md` rather than tuning until it does.*
3. **Seven curves + one paragraph each on the mechanism**, in a committed notebook.
4. `make check` green.
5. Every number regenerates from a committed script reading `results/*.json`.
6. `manifest.json` written for every run.

## Outputs

- `experiments/phase-0.1-grpo-by-hand/results/*.json` + the figure script
- `notebooks/grpo-by-hand.ipynb` (or a script + markdown — user's call)
- `docs/phases/phase-0.1-grpo-by-hand-retro.md`
- A `/learn` entry appended to `tasks/lessons.md`

## Non-goals

Reasoning emergence (Phase 0.3) · the `verifiers` spec (Phase 0.2) · any LLM judge · any grader
pathology · `bisect` · performance or reusability of this code.

---

## Prep reading (mechanics tier — do this before writing the loop)

Distinct from `../../literature-review/`, which is the *research* reading that positions the novelty
claim. This is the **pedagogical** tier: what you need in your hands to write the loop.

1. **DeepSeekMath**, the GRPO section — the group-baseline derivation, first-hand.
2. **A GRPO practitioner digest** (Cameron Wolfe's "GRPO++: Tricks for Making RL Actually Work" is a
   good one) — the tricks that separate a loop that runs from a loop that learns: advantage
   normalization, clipping, KL placement, off-policy staleness.
3. **`trl`'s `GRPOTrainer` source** — read it *after* writing your own, as the diff against your
   implementation. What they do that you didn't is the lesson.
4. **Rollout Pass-Rate Control** (2605.05112) — read only §"why p=0.5", for ablation D's mechanism.
   The full read belongs to the literature gate.

**Order matters:** 1 → write the loop → 3. Reading `trl` first collapses the phase into
transcription and forfeits the intuition the phase exists to build.

## Risks

- **Base policy has zero pass rate on the chosen task** → no gradient on step 1 and the phase stalls.
  *Mitigation:* measure base pass rate **before** picking the task; require ≥5% at k=8. This is the
  same reachability logic as `p_hack@64`, applied to task difficulty instead of exploit difficulty.
- **Scope creep into a good trainer.** This code is disposable. If it takes more than ~10 h, cut runs
  4–6 and keep 1, 2, 3, 7 plus all four ablations. **The ablations are the deliverable; the ladder is
  scaffolding for them.**
- **Silent off-policy staleness** from reusing rollouts across epochs — will look like an unrelated
  instability. Log the ratio distribution so it is visible rather than mysterious.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-07-26 | Plan drafted | Scaffold. Not yet locked; lock at branch cut. |
