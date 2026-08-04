# Related work — the differentiation table

> **⚠ UNVERIFIED. Every row here came from a single LLM-assisted research pass on 2026-07-26.**
> arXiv IDs, dates and quantitative claims from an LLM pipeline **can be wrong**. Nothing in this
> table is load-bearing until a first-hand ⬤ read clears it at the Phase 0.5 literature gate
> (`../literature-review/README.md`).
>
> *Inherited lesson (waterline, canary): every ID / date / result that positions a claim gets a
> first-hand read before it enters a locked doc.*

## The one-sentence novelty claim (to be defended or amended at the gate)

> *Nobody has asked whether an environment's post-training outcome is predictable from inference-only
> probes run before training — and nobody has held the skill fixed while varying the environment's
> authorship.*

## Differentiation

| Prior work | id | Overlap | Differentiation | Verified? |
|---|---|---|---|---|
| **Before the Model Learns the Bug: Fuzzing RLVR Verifiers** | 2606.01066 | Fuzzing verifiers; FP rates (math 83.2% / JSON 86.9% / code 55.7%); exploit found in 2–4 queries, 94–100% of trials | Uses **intentionally injected synthetic bugs** and **never trains a model** — both are the paper's own stated limitations. We use **natural defects in a constructed-pathology environment** and **close the loop to actual GRPO outcomes**. | ☐ |
| **Prime Sprints — reward-hacking track** | primeintellect.ai/blog/reward-hacking | Asks "can we predict hacking onset from the first 20 steps"; 1B, 100 steps, ~$0.64 | Ours is **zero steps** — pre-training, no GPU. And the frontier-as-forecaster mechanism (H2) is distinct from onset detection. Also a **compute source and a venue**, not only prior art. | ☐ |
| **Breaking Barriers: Do RL Post-Training Gains Transfer To Unseen Domains?** | ICLR 2026 (uiuc-kang-lab) | RLVR cross-domain transfer; 1.5B; 16 benchmarks; HumanEval+ → −13.8% on MATH | They vary the **domain** (math → code → knowledge). We hold the **skill fixed and vary the environment's authorship**. "Does math RL transfer to code" is answered; "do gains from *this* debugging env transfer to *another author's* debugging env" is not. **This is the closest threat to `endemic` — read first.** | ☐ |
| **Rollout Pass-Rate Control** | 2605.05112 | Derives p≈0.5 as the max-signal point four ways; prefix-replay method | Owns the pass-rate axis. **Cite, use as battery axis A3, attribute — do not re-derive.** Our H3 claims A3 is *not* the dominant axis, which is a claim *about* their axis, not a competing derivation of it. | ☐ |
| **HUD — verifier & reward design guide** | hud.ai/resources | Names the failure modes: proxy exploitation, sparse-reward blindness, rigid graders, grader inconsistency | Qualitative practitioner guidance. We **quantify each as a battery axis and validate against outcomes**. | ☐ |
| **Epoch AI — RL environments FAQ** | epoch.ai/gradient-updates/state-of-rl-envs | The economics: $200–2,000/task, ~$2,400 compute/task, reward hacking as #1 complaint | Journalism, not measurement. Our motivation, and the source of the cost figure the project is justified against. | ☐ |
| **Reasoning Gym** | 2505.24760 | 100+ procedural task families, algorithmic verifiers; +9.7% MATH / +7.7% BBH at 3B | **Borrowed substrate** for the η decomposition, and reproduction target R3. Not a competitor. | ☐ |
| **GEM: A Gym for Agentic LLMs** | 2510.01051 | Gym-for-agentic-LLMs framing; single/multi-turn envs; 5 trainer integrations | Infrastructure. Establishes the interface norms we adopt; makes no claim about environment *quality*. | ☐ |
| **Endless Terminals** | 2601.16443 | Procedurally generated terminal envs at scale; "simple RL succeeds when environments scale" | Scales environment *supply*; is silent on environment *quality screening*. A natural consumer of `assay`, not a competitor. | ☐ |
| **Environment Scaling for Interactive Agentic Experience Collection** | 2511.09586 | GEF-loop taxonomy; complexity/realism/interactivity axes | Survey. Useful for framing; check whether it already names the prediction question as open. | ☐ |
| **Natural Emergent Misalignment from Reward Hacking** (Anthropic) | 2511.18397 | Hacking emerges ~step 1,500, saturates ~4,000; **no safe rarity threshold** | The counter-evidence to the reachability assumption — read it *against* our own design. Drives the L1–L6 ladder. | ☐ |
| **Credit assignment survey (reasoning → agentic)** | 2604.09459 | 47 methods, 2024–early 2026 | **What we are deliberately not doing.** Cited to bound scope. | ☐ |

## `verl` — read first-hand 2026-08-03, and it is prior art for M2

> **This section is ⬤ first-hand**, unlike the ☐ table above: the repo, its README, its
> `examples/` tree and the `rollout_correction` script were read directly on 2026-08-03. It is kept
> separate so the verified content is not diluted by the unverified table.

**Repo moved:** `volcengine/verl` → **`verl-project/verl`** (22.8k stars, pushed 2026-08-04). Any
link or citation using the old path should be updated.

### Prior art for M2, with parameters that corroborate our measurement

`examples/rollout_correction/` ships a production treatment of the exact effect M2 measured, on the
same afternoon and independently:

| verl config | what M2 measured |
|---|---|
| `rollout_is=sequence` — sequence-level importance sampling | M2 reported the **sequence** ratio, because that is the level at which the discrepancy degrades; per-token it is negligible (median 3e-6) |
| `rollout_is_threshold=2.0` — cap on IS weights | M2's ±1σ interval at L=512 is **[0.218, 1.555]**. The cap sits at the edge of the distribution we measured — it is not an arbitrary constant |
| `rollout_is_batch_normalize=true` — self-normalised, mean→1 | Fixes exactly the residual M2 saw: `E[ratio]` = **0.943** at L=512 rather than exactly 1 |
| `rollout_is_eff_sample_size` — ESS, monitored every step | The degeneracy diagnostic: effective batch < nominal batch, which is what a wide ratio distribution costs |

**Why this matters beyond citation.** It is independent confirmation that M2's `not_free` verdict is
a real effect and not a harness artifact, from a codebase that had to solve it at scale — and it is a
**concrete design for rung 4**, better than one we would invent. `prime-rl`'s `Max Off-Policy` log
line is the same conclusion reached by a second stack.

### It does **not** rescue R0

Checked directly, because the reproducible-examples pitch suggests otherwise:

- `examples/data_preprocess/` has gsm8k, math, geo3k, aime2024, hellaswag, full_hh_rlhf — **no
  Countdown**.
- The `recipe/` submodule (`verl-project/verl-recipe`) has `r1`, `dapo`, `prime`, `spin`, `sppo`,
  `entropy`, … — **no tinyzero**.
- TinyZero appears once in the README, under the heading **"Awesome Projects Built with `verl`"**,
  as an external link to `Jiayi-Pan/TinyZero`.

So TinyZero is **community work built on verl, not a verl recipe**, and R0's blocking problem —
*the original publishes no number to compute a delta against* — is untouched. Nothing here changes
the R0 decision.

### Where it belongs in the project

`CLAUDE.md` §6 lists `prime-rl` or `trl` for Walk onward. **verl is absent, and was never
evaluated** — one mention in the entire repo, describing TinyZero's stack. Two reasons it should be
a candidate rather than missing:

- It implements **DrGRPO**, which is independently where Phase 0.1's length-normalisation finding
  pointed — reached here by measurement, there by design.
- It implements **GSPO** and **DAPO**, adjacent to the grader-factorial work, and the
  `rollout_is_*` surface above.

Against that: Phase 0.2 validated `prime-rl` at **$0** on the free tier, a hard advantage verl does
not have (its examples assume 64×H800). **This is a recorded candidate, not a proposed switch.**

## Verification protocol

1. Verify every arXiv ID / URL resolves to the claimed paper (⬤ id).
2. First-hand read the **Tier-1 threats** before locking the pre-registration:
   **2606.01066** (closest on the diagnostic leg), **ICLR 2026 Breaking Barriers** (closest on the
   η leg), **Prime Sprints** (closest on the prediction leg).
3. For each, fill the per-paper review template in `../literature-review/README.md` and record
   whether the novelty claim survives.
4. **Recursive novelty-perimeter rule:** if a first-hand read surfaces closer prior art, add it with
   the **next number** (don't renumber), a dated note on why, and require a first-hand read. Assume
   more will surface — this field moved fast in H1 2026.

## Resolved concerns (do not relitigate once the gate closes)

*(empty — populated at the Phase 0.5 gate)*
