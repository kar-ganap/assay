# 01 · Before the Model Learns the Bug: Fuzzing RLVR Verifiers

> **⬤ FIRST-HAND**, 2026-08-04. Abstract page + full HTML (`arxiv.org/html/2606.01066v1`) read
> directly. arXiv **2606.01066**, Jaideep Ray (single author).
> Cluster **K2 — closest prior art / scoop.** `README.md`'s key question: *"Confirm it really uses
> **injected** bugs and **never trains**. If it trains, our differentiation collapses."*
>
> **Headline: the differentiation does not collapse, but two of `related-work.md`'s claims about
> this paper are wrong, and the fix sharpens our novelty claim rather than weakening it.**

## Background

RLVR replaces preference labels with executable reward functions — math answer checkers, JSON
tool-call validators, code unit-test harnesses. The paper's framing sentence is the same observation
`assay` is built on:

> *"That makes the reward partly a software artifact: if the verifier is wrong, optimization can
> learn the bug."*

Their response is a **fuzzing framework**: generate adversarial completions, run them against a buggy
verifier and a stricter reference, log paired decisions, and report false-positive, false-negative,
disagreement, exploit and uncertainty metrics.

## Key Ideas

1. **Paired verifiers as ground truth.** Rather than judging a verifier in isolation, they pair each
   *intentionally buggy* implementation with a *stricter reference variant* and score the
   disagreement. This is the same move as our proxy/true split, one level down — at the grader rather
   than at the environment.
2. **Adversarial completion generation** — the fuzzer produces completions designed to slip past a
   verifier, which is the mechanism our battery axis **A1 (hackability)** performs with a frontier
   model instead of a fuzzer.
3. **Black-box query budget as the exploit metric.** How many verifier queries to find a false
   positive. This is a *cheapness-of-exploit* measure and maps onto A1's cost axis.
4. **Optimization experiments** — template search plus **tabular policy gradients** — asking whether
   repeated reward feedback *amplifies* verifier false positives.

## Results, at three levels

**Smart high-schooler.** If the grader that scores an AI's homework has a bug, the AI will find the
bug and exploit it instead of learning. This paper builds a bug-finder for graders, and shows that
buggy graders wave through wrong answers most of the time — and that the bug is easy to find, usually
in two tries.

**Undergrad.** They construct buggy/strict verifier pairs in three domains and measure false-positive
rate under adversarial completions: math **0.832**, JSON tool-calls **0.869**, code unit tests
**0.557**, with all strict variants at **0.000**. A black-box search finds an exploit within two
queries in 94/100 math and 98/100 JSON trials, and within four queries in 100/100 of both.

**Early grad.** The contribution is a measurement instrument and a taxonomy of "short-verifier
mistakes", validated on synthetic defects, plus a demonstration that tabular policy-gradient
optimisation amplifies those false positives within fixed template pools. The paper is explicit that
this is **not** an end-to-end RLVR result.

## Key Quotes (verbatim)

> *"We pair intentionally buggy implementations with stricter reference variants."*

> *"The taxonomy is intentionally practical: these are short-verifier mistakes rather than exotic
> attacks."*

> *"The evaluation is scoped to verifier behavior, not end-to-end RLVR training."*

> *"The optimization experiments use template search and tabular policy gradients. They measure
> whether repeated reward feedback amplifies verifier false positives in the fixed template pools;
> they do not estimate exploit rates for neural fine-tuning runs."*

> *"Within two verifier queries in 94/100 math trials and 98/100 JSON-tool trials. Within four
> queries, both reach 100/100."*

> **⚠ *"verifier reliability is measurable before expensive training begins, so RLVR builders should
> stress-test reward verifiers before using them as training objectives."*** ← the sentence that
> forces a novelty-claim revision. See below.

## Connection to Project

### What `related-work.md` got right

| claim | verdict |
|---|---|
| FP rates math 83.2% / JSON 86.9% / code 55.7% | ✅ **exact** — Table 2 gives 0.832 / 0.869 / 0.557 |
| exploit in 2–4 queries, 94–100% of trials | ✅ **exact** |
| uses **intentionally injected** bugs | ✅ confirmed verbatim |
| arXiv ID and title | ✅ correct |

The LLM-assisted pass was accurate on every *quantitative* claim. That is worth recording, because
`lessons.md` #1 predicts the opposite and this is evidence about when it does and does not bite.

### What it got wrong

**1. "never trains a model" is too strong.** They *do* run optimisation — template search and
**tabular policy gradients** — to test amplification. What they do not do is train a *neural* policy
end-to-end. The correct phrasing is **"never trains a neural policy end-to-end; optimisation is
tabular over fixed template pools."** The distinction matters: they have an amplification result, it
is just not on a real policy.

**2. ⚠ The bigger one — they DO stake a pre-training-diagnostic claim.** Our one-sentence novelty
claim currently reads:

> *"Nobody has asked whether an environment's post-training outcome is predictable from inference-only
> probes run before training..."*

Against *"verifier reliability is measurable before expensive training begins, so RLVR builders should
stress-test reward verifiers before using them as training objectives"*, **"nobody has asked" is
false.** They asked. A reviewer would find this in the abstract-adjacent text and the claim would not
survive.

### The differentiation that does survive — and it is stronger

The gap is not the *question*; it is the **validation**. They recommend pre-training stress-testing
and never check whether it predicts anything, and they say so themselves: *"they do not estimate
exploit rates for neural fine-tuning runs."*

| | fuzzing-RLVR-verifiers | `assay` |
|---|---|---|
| defects | **injected** synthetic bugs | **constructed pathology** in a grader factorial over one task set |
| optimisation | tabular policy gradients, fixed template pools | **GRPO on a real 1–2B policy** |
| outcome measured | verifier FP/FN, disagreement | **post-training proxy–true gap**, as a slope |
| the loop | **open** — asserts pre-training measurement matters | **closed** — the diagnostic is scored against the training outcome it claims to predict |

**Proposed novelty claim (for the user's review, not yet adopted):**

> *Pre-training verifier diagnostics have been proposed and measured; **none has been validated
> against the post-training outcome it claims to predict**. `assay` closes that loop — and separately
> holds the skill fixed while varying the environment's authorship.*

This is narrower, defensible against a first-hand read of #1, and — usefully — it makes #1 a
**supporting citation** rather than a threat: their instrument establishes that graders are broken
and cheaply exploitable; ours asks whether that predicts what training does.

### What to adopt, with attribution

- **The query-budget-to-exploit metric** for A1. Cheaper and more interpretable than a binary
  "hackable?", and it gives a published comparator (2–4 queries).
- **Paired buggy/strict verifiers** as the calibration for A2 (grader degeneracy) — the same shape as
  our proxy/true split.
- **Their taxonomy of "short-verifier mistakes"** as the seed for the grader-factorial axes, credited.

## Study Questions

1. Their strict verifiers score **0.000** FPR. Is that because the strict variant is *correct*, or
   because the fuzzer is only searching for the buggy one's failure mode? What would it take to
   fuzz the strict verifier adversarially?
2. Code FPR (0.557) is far below math (0.832) and JSON (0.869). Is code genuinely harder to fool, or
   is the injected bug simply less severe? Does that undercut cross-domain comparisons?
3. If tabular policy gradients already amplify FPs in fixed template pools, what does a neural policy
   add — more exploits, or the same exploits found faster?

## Challenge Corner (adversarial)

- **Against them:** injected bugs are chosen by the author, so FP rates measure *how bad a bug they
  chose*, not how bad real verifiers are. The 0.832 is not a property of math verification.
- **Against us:** a reviewer could argue our "constructed pathology" is the same critique in different
  clothes — we also author our defects. **Our answer must be the grader factorial**: we vary the
  grader systematically over one fixed task set and measure the *response curve*, rather than
  reporting a level for one chosen defect. Worth stating explicitly in the paper.
- **Against the field:** both papers measure environments the authors built. Neither establishes what
  the FP rate of a *randomly chosen Hub environment* is — which is what Gallop's field report (3.5)
  is for, and is a stronger contribution than either paper makes.

## Discussion Notes

*(to be filled during the interactive pass — step 3 of the process in `README.md`)*
