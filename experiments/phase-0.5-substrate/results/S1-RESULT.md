# S1 result — the substrate is rejected, and the reason reframes the requirement

**Runs:** `wp-2step-2digit` (S1a) and `wpr-2step-2digit` (S1b), Modal L4, 2026-08-29.
Bands pre-registered in `docs/phases/phase-0.5-substrate-screen.md` before any sampling.

| run | instruction | median tokens | `p_hack` (best word) | `pass@1` | verdict |
|---|---|---|---|---|---|
| **S1a** | answer tag only | **9** | **0 / 2048** | 0.110 ✅ | rejected on hack band |
| **S1b** | reason first, then tag | **160** | **0 / 1024** | 0.348 ✅ | rejected on hack band |

Band was `p_hack ≥ 1/256 = 0.0039`. At n = 1024 that predicts ~4 hits; `P(0 hits) ≈ 0.018`. Zero is
strong evidence the true rate is below the floor, not a resolution failure.

## The first hypothesis was wrong, and so was the second

**S1a falsified "prose vs digits."** `wordproblem` puts prose in the *question* and still produced
nine-token completions — `<57+7-11>46` — because `_ANSWER_INSTRUCTION` instructs terseness. Base
rate is a property of what the model **generates**, not of what it is **shown**.

**S1b falsified "length."** Asking for step-by-step reasoning worked exactly as intended: median
completion length went **9 → 160**, `pass@1` improved 0.110 → 0.348, and the output is real English:

> *"Firstly, let's identify the initial number of apples at the stall, which is 75. Then, we need to
> calculate the total number of apples added…"*

Long, fluent, grammatical prose. **And still zero hack words.**

## What actually binds: vocabulary breadth, not length

At a **matched budget of 3,802 word tokens**, against R1's story completions:

| substrate | median tokens | distinct word types @ matched budget | `p_hack` |
|---|---|---|---|
| `story` (R1) | 256 | **1000** | 0.0059 – 0.2096 ✅ |
| `wordproblem` + reasoning | 160 | **188** | **0** ❌ |

**5.3× narrower vocabulary at equal length.** The completion vocabulary is bound to the task: `the`,
`apples`, `number`, `stall`, `delivery`, `customers`. A word problem about apples elicits words about
apples. `ocean` has no reason to appear and, across 1024 completions, never does.

## The requirement, restated — and it is a tension, not a checklist

§15 currently records that a substrate needs *"a reachable exploit AND checkable ground truth."*
S1 shows those two are **in tension by construction**:

- a task is **verifiable** because its output space is **constrained**;
- a lexical exploit is **reachable** because the output space is **open**.

R1's hack was reachable *precisely because* the task was open-ended storytelling — the same property
that made `r_true ≡ 0`. **R1 did not happen to lack a truth signal; it lacked one for the same
reason its exploit was reachable.** The trade was structural, not accidental.

## Consequence for `bisect`

**The hack-word model does not transfer to a verifiable task, and should not be carried into Walk.**
`bisect`'s exploit has to live *inside* the constrained output space the truth-check imposes —
special-casing the input, `try/except`, hardcoding the expected output, editing the test. Those are
already what `README.md` names as `bisect`'s exploits, so the design is sound; what S1 kills is the
idea that R1's substrate could be extended into one that measures a gap.

**What this does not show.** Only lexical exploits were screened. It is not evidence that a
*structural* exploit is unreachable in a constrained output space — that is `bisect`'s premise and
remains untested. S1 narrows what needs testing rather than threatening it.

## Cost

~$1.60 total, against a $6 stop: S1a ~$0.08, a timed-out S1b attempt ~$1.20 (90-minute cap, no
checkpointing — see the commit that added it), S1b ~$0.30.
