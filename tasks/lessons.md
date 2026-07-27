# Lessons — read at session start; append after any correction or surprise

Process stream (`CLAUDE.md` §11). Paper-irrelevant by design: rule changes, where discipline
slipped and why, tooling friction, permission-allowlist candidates. Problem-stream learnings go to
phase retros and `docs/conceptual.md`, not here.

Every rule proposed here must carry a **trigger statement** — *"Without this, X; with this, Y."*
If `Y == X` it is decoration (`CLAUDE.md` §14).

---

## Carried in at scaffold (inherited, not yet earned here)

- 2026-07-26 · **Every arXiv ID / date / result that positions a claim gets a first-hand ⬤ read
  before it enters a locked doc.** `docs/related-work.md` was seeded from a single LLM-assisted
  research pass; IDs and quantitative claims from an LLM pipeline can be wrong. Nothing in the
  reading list is "read" until the Phase 0.5 gate clears it.
  *Trigger: without this, a whitespace claim rests on a hallucinated citation and collapses at
  review; with this, the novelty claim is defensible before any money is spent.*
  (Inherited from waterline / canary, where the identical lesson was earned.)

- 2026-07-26 · **Test the load-bearing control *while it's working*** (SynthOracle lesson, via
  waterline). The cheap control on a load-bearing assumption is most urgent exactly when the
  confirming signal is present — that signal is what hides the untested premise.
  → Here: run the **positive control** and the **held-out-grader gold-set validation** *before* the
  grid, not after signal accrues.
  *Trigger: without this, a confirming result is indistinguishable from a broken rig; with this, the
  rig is proven before the result is trusted.*

- 2026-07-26 · **Framing path-dependence.** Once a framing is established, reasoning accumulates
  support rather than scrutinizing it. Counter: the framing-stress reviewer at stage boundaries
  (`docs/process.md`).
  *Trigger: without this, the project spends four stages confirming its opening premise; with this,
  the premise is attacked at three specific points.*
  (Inherited from epibench.)

- 2026-07-26 · **Secrets strictly from `.env`, never the shell.** `load_dotenv()`'s no-override
  default lets a shell `ANTHROPIC_API_KEY` silently win over `.env`, and a work account gets billed
  for personal experiments — it happened in `../agentic_engg` (2026-06-03).
  *Trigger: without this, a work key wins silently; with this, the project can only ever use the key
  explicitly placed in `.env`.*

- 2026-07-26 · **Report a failed reproduction as a result, not a blemish.** `../originality`
  withdrew its own headline on a matched null and the portfolio is stronger for it. Reproduction
  verdicts here explicitly include FAILED TO REPRODUCE.
  *Trigger: without this, an inconvenient reproduction gets quietly re-run until it agrees; with
  this, the disagreement is the finding.*

---

## Earned here

*(none yet — first entry lands at the Phase 0.1 retro)*
