# Phase 0.5 — the substrate screen (S1), pre-registered before the run

> **Locked 2026-08-29, before any sampling.** Runs against expiring Modal credit, so it is scoped to
> what can be decided in one pass. Predecessor: `phase-0.4-r1-retro.md` §5, which turned this from an
> assumption into a stated requirement.

## Why this exists

R1 measured reachability and **could not measure a proxy–true gap at all**, because neither Crawl
substrate has both halves:

| substrate | reachable hack? | checkable ground truth? |
|---|---|---|
| `story` | ✅ 1.35 / 0.59 / 20.96% | ❌ `r_true ≡ 0` — there is no right answer to a story |
| `arithmetic` | ❌ **0 / 4096** | ✅ exact |

Every R1 run was `hack_only`, so the project's own outcome variable — proxy minus true — was
identically zero throughout its reachability gate. `CLAUDE.md` §15 now records *"a substrate needs a
reachable exploit AND checkable ground truth"* as a **design requirement on `bisect`**, and Walk
cannot start without a candidate that meets it.

`WordProblemFamily` is that candidate: **prose** (so an English hack word has somewhere to appear —
the thing `arithmetic` lacked) with an **integer answer known by construction** (so `r_true` is
checkable with no held-out suite, judge, or labelling). Zero-GPU properties are already pinned in
`tests/test_crawl_wordproblem.py`. This screen measures the two that need a GPU.

## What is measured — one sampling pass, two quantities

Base policy, `sprints/Llama-3.2-1B-Instruct`-class at T=1.0, no training. One draw serves both
quantities, because base rate is a property of the completions rather than of any grader — the same
economy R1's G2 used to count three words in one pass.

1. **`p_hack@k` per candidate word**, at **k = 2048** (not 64 — see below).
2. **`pass@1` on the checkable answer**, i.e. is the true signal present at a usable rate.

Words: R1's three (`ocean`, `midnight`, `forgotten`) so the result is directly comparable to G2's
story numbers, on the same `max_new_tokens = 256` pin.

**k = 2048, and the choice is the finding from R1 applied to itself.** R1 showed `1/64` is the
*resolution floor* of a 64-sample screen rather than a reachability threshold, and that two variants
below it saturated anyway. A screen that cannot resolve 0.006 cannot answer this question: at
k = 2048, `P(0 hits | p = 0.006) = 4.4e-6`, against **0.68 at k = 64**.

## Pre-registered bands — locked before the run

**Both must hold for the substrate to be admitted.** They are stated separately because failing them
means different things.

| quantity | band | reading if it fails |
|---|---|---|
| **`p_hack@2048`** | **≥ 1/256 (0.0039)** for at least one word | the hack is not reachable here — same failure as `arithmetic`, and the prose did not help |
| **`pass@1`** | **∈ [0.05, 0.60]** | below: starved, `r_true` is present but unlearnable, so no gap can open. above: already solved, no headroom, and the grid measures nothing |

**Why 1/256 rather than 1/64.** The old bound was a sampling artefact. This one is set from
*measured reachability*: R1's `midnight` at **0.0059** saturated in ~30 steps, so the threshold has
to sit below the lowest rate demonstrated to be reachable. 1/256 = 0.0039 does; 1/64 = 0.0156 does
not, and would have excluded `midnight`. **This is the L1 redesign, run on one substrate.**

**Why the pass@1 upper bound.** Novel to this screen and not inherited: `story` never needed one
because it had no truth signal. If the base policy already solves the task, there is no gain to
decompose and η is undefined.

### The third branch, written before the numbers

Pre-registering only pass/fail is what produced Phase 0.4's missing-cell failure, where a scorer
returned the nearest verdict it owned because the plan gave it no other. So:

| observed | verdict | action |
|---|---|---|
| both bands clear | **admitted** | `WordProblemFamily` is the `hack_or_correct` substrate; Walk proceeds on it |
| `p_hack` clears, `pass@1` outside | **partial — difficulty mis-set** | the *substrate* is sound and the *setting* is not; re-screen other settings before abandoning the family |
| `p_hack` fails at k=2048 | **rejected, and informative** | prose alone does not buy reachability. Records that the `story`/`arithmetic` split is not about prose-vs-digits, which is what we currently believe |
| both fail | **rejected** | `bisect`'s design requirement is harder than assumed; escalate before Walk is planned |

## Cost and stop rule

~26k completions at 256 tokens. Measured comparator: M1 did 12,800 completions at 512 tokens for
**$1.57** on an L4, so this is **~$1.50–3**. **Hard stop at $6** — if the app has not finished by
then, kill it and report what landed; a screen that overruns its own budget while diagnosing a
budget-driven defect is not a screen worth having.

## Non-goals

- **Not training.** No gradients. This is base-policy sampling only.
- **Not choosing the grader.** Whether the admitted substrate gets a `hack_or_correct` grader, and
  with what reward shape, is the grader-variant design and is the user's under §7.
- **Not settling `P-outcome-cheap` or E3.** Those are analysis decisions and need no GPU.
