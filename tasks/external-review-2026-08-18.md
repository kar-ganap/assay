# External review — claim propagation, 2026-08-18

> **Scope banner.** Written by an out-of-project session (application prep) that read the repo from
> the outside. It reports **findings only**. No scope, hypothesis, axis definition, gate status,
> pre-registration, or roadmap item has been changed, and §3 is **raised, not decided** — it is the
> project owner's call. Nothing in this file has been applied.

**Trigger.** `README.md` is now linked from an outward-facing document, so it is being read by people
who will not open `docs/phases/`. Reading it cold surfaced one pattern.

---

## 1. The finding: a propagation gap, not an accuracy gap

**Every correction below was already made correctly inside the project.** The retros caught all three,
`tasks/lessons.md` turned one into a standing rule, and `reproductions/README.md` and
`docs/plain-english-summary.md` state the corrected versions. The corrections simply **never reached
the outermost surfaces** — `README.md`, `CLAUDE.md`, `docs/grant-readiness.md`.

So the epistemic machinery is working. What is missing is a propagation step: when a retro corrects a
claim, nothing walks the claim outward to the files that already quote it.

**All three misses point the same way — the outer surface carries the more flattering version.** That
is the exact failure `lessons.md` already names (2026-08-06): *"an unlabelled metric drifts to
whichever series makes the sentence work. Nobody chose that."* Same mechanism, one layer out.

---

## 2. The three claims, with the corrected version already in-repo

### 2.1 `15/15` — unqualified train figure, and the project's own rule says name the curve

| | |
|---|---|
| **Stale** | `README.md:20` — *"it takes **8–40 steps** — 15 runs out of 15, for `$0`"* |
| | `README.md:44` — *"15/15 runs exploit a broken grader in 8–40 steps"* |
| | `CLAUDE.md:259` — *"15/15 runs at 1B reached hack rate 1.0 in 8–40 steps"* |
| | `docs/grant-readiness.md:98` — *"15/15 saturate at 1B for `$0`"* |
| **Correct, in-repo** | `phase-0.4-r1-retro.md:28` — **15/15 train · 12/15 eval**; `midnight-s1` (0.266), `s2` (0.234), `s4` (0.617) never reach 1.0, first two never cross 0.5 |
| | `phase-0.4-r1-retro.md:306` — *"✅ **15/15 train · 12/15 eval**, 8–40 steps"* |
| | `reproductions/README.md:37` — already correct: *"15/15 runs on the train curve, 12/15 on the pre-registered…"* |
| **Standing rule** | `tasks/lessons.md:283` — *"When two metrics exist, every claim names its curve."* The retro's own aside: *"The unqualified '15/15' in the first draft was the train figure quoted under an eval table."* |

The gate conclusion is unaffected — Branch 0 is answered on both curves. Only the count needs its
curve named.

### 2.2 `52-point gap` — the setup quoted as the result

| | |
|---|---|
| **Stale** | `README.md:41` — *"A degenerate grader produced a **52-point** proxy–true gap on demand."* |
| **Missing** | The gap is what breakage B was **built** to produce. The finding sits one line further down in the 0.1 retro, which heads it *"a finding, not the predicted signature"*: proxy **0.993 ± 0.002** vs true **0.474 ± 0.010**, gap **0.519 ± 0.011**; **removing the KL leash *reduced* the gap by 0.037, same sign on 3/3 seeds**, and the leashed arm ended with **lower true reward on every seed**. At β = 0.04 carrying 54% of the loss, *"the leash is not restraining anything."* |
| **Correct, in-repo** | `docs/plain-english-summary.md:104-106` already runs both, adjacent: the manufactured gap, then *"A standard mitigation that the field reaches for did not work here — and would have been reported as working if we hadn't tested both directions."* |

The README keeps the sentence about the apparatus and drops the sentence about the result. The result
is the more interesting half and the more defensible one — constructing a control is table stakes; a
seed-replicated negative on KL is a finding.

### 2.3 `99.8%` — headline is the weak number; the strong one is omitted

| | |
|---|---|
| **Stale** | `README.md:42` — *"an independent trainer reached **99.8%**, starting inside our own measured pre-training band."* |
| **Correct, in-repo** | `docs/plain-english-summary.md:131` — *"The headline number is… **99.8%**… **But the more meaningful number** is the very first measurement, before any training: **58.8%**, against our hand-built version's **57.1% ± 1.9%**… That agreement is the actual evidence the translation is faithful."* |
| | `docs/grant-readiness.md:25` — already carries both: eval **0.9980**, step-1 eval **0.5879** inside the measured band (0.571 ± 0.019) |
| | `phase-0.2-…-retro.md:34` — step 1 **0.5879** vs run 7's first ten steps **0.571 ± 0.019**, inside the band |

99.8% is endpoint accuracy on three-digit addition — an easy task, and a number that invites the
reader to discount it. The band agreement is the claim that actually does work (it validates the
port), and it is the one the README omits. `plain-english-summary.md` and `grant-readiness.md` both
get the ordering right; the README inverts it.

---

## 3. Raised, not decided — does E3 inherit the saturation defect?

`P-outcome-cheap` is **under amendment (2026-08-15)**: a slope-only estimand over steps 50–200 reads
≈ 0 both for a healthy arm (flat at zero) and for a pathological arm that saturated before step 50
(flat at maximum), so **H4 would report itself falsified while working perfectly**. R1 measured
`forgotten` saturating at eval step **9.10**.

`P-outcome-headline` (η) was **not** amended, and §2.3 of `conceptual.md` justifies that: η is a gain
ratio across environments, not a slope over steps.

**The question:** **E3** is not the endpoint ratio — it is *"η falls with training step… measured as
η(step) within a single confirmatory run."* If a confirmatory arm saturates around step 9, an η
trajectory read over the same 50–200 window is sampling an already-converged policy, and the
trajectory could flatten for the same reason the gap slope did. That would be the amended defect in a
different dress, on the headline outcome rather than the cheap one.

This may already be covered — E3 is a ratio of gains rather than a difference of rewards, and the
failure mode may not carry across. **Not asserting it does.** Flagging it as worth five minutes
before the Run grid, since the amendment note says this class of thing *"must be settled before the
Run grid, not diagnosed from it."*

---

## 4. Suggested action

1. Propagate 2.1–2.3 outward to `README.md`, `CLAUDE.md`, `grant-readiness.md`. Every corrected
   version already exists in-repo — this is copy-out, not re-derivation.
2. Consider whether `lessons.md` wants a companion rule to the 2026-08-06 entry: *when a retro
   corrects a claim, grep the claim outward and fix every surface that quotes it.* The three misses
   here are one mechanism, and it will recur at every phase boundary.
3. Adjudicate §3.
