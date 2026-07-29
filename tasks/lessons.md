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

*Phase 0.1 is not finished — these are logged live because they are process, and because the session
that produced them would otherwise be the only record.*

- 2026-07-28 · **Diagnose by measurement, not by inference from a traceback.** Bringing up
  `HFPolicy` took six failures. Each was debugged by reading the last stack line and fixing what it
  pointed at: a bigger GPU, then span slicing, then head slicing, then the generation batch. All
  four were genuine improvements aimed at *symptoms*, while the real causes — gradient checkpointing
  never engaging, and a missing attention mask — sat untouched. The two failures diagnosed by
  **looking** (computing expected-vs-observed GPU memory; reading the raw completions) found their
  cause in minutes each.
  *Trigger: without this, a debugging session costs a day and several fixes that were never needed;
  with this, "what should this number be?" is asked before the next change is made.*

- 2026-07-28 · **Every path that produces rollouts persists a sample of them, from the first run.**
  Raw capture was built for the calibration sweep, where it diagnosed a parse-failure problem in one
  look — and then not carried into the training path. That single omission is what made the
  intervening failures expensive; the moment it was added, the cause was obvious in one line of
  output (`"<The is when were you was was.<?<?201You we we..."` from a model with a measured 0.72
  pass rate). `experiments/README.md` already required it.
  *Trigger: without this, a broken run is debugged by inference; with this, it is debugged by
  reading what the model actually emitted.*

- 2026-07-28 · **An "it can learn at all" check runs before any hyperparameter probe.** Overfitting a
  single prompt exercises generation, grading, advantages, loss, masking, backward and the optimizer
  step at once, with an unambiguous verdict. It was added only *after* an LR probe returned
  `reward 0.000` at all four rates — a full probe spent discovering that no learning rate could have
  helped.
  *Trigger: without this, hyperparameter probes are run against a broken gradient path and their
  output is confidently meaningless; with this, one cheap run separates "wrong setting" from "broken
  rig".*

- 2026-07-28 · **A pre-registered rule must be able to say "the harness is broken", not only "this
  setting is wrong".** The LR selection rule needed **four** amendments. Rule 0 (reward must move off
  zero) was missing entirely, so a probe on corrupt output would have dutifully selected a rate.
  Rule 1 conflated *learning* with *collapse* and rejected every rate for falling entropy while
  reward rose to 1.0. Rule 3's rationale ("take the largest") inverted on a saturating task, where
  the fastest rate gives the fewest usable steps.
  *Trigger: without this, a rule returns a confident answer computed from garbage; with this, rig
  failure is a distinct, checked-first branch.*
  → **Generalises:** every ablation signature here now carries two failure branches, *prediction
  wrong* (a finding) and *rig broken* (a bug).

- 2026-07-28 · **Test what runs, not what is configured.** `test_every_entry_carries_the_pinned_task`
  asserted the ladder table pinned `add-3digit`, and passed. The Modal entry point's `--setting`
  default silently overrode it, and two 200-step runs went to the deprecated arm. The test gave
  false assurance about precisely the thing it appeared to guarantee.
  *Trigger: without this, a green test covers the config while the dispatch overrides it; with this,
  defaults that can override pinned values are removed rather than tested around.*

- 2026-07-28 · **Shell syntax silently changes what a command does — never gate a commit behind a
  pipe.** `make check 2>&1 | tail && git commit` exits with `tail`'s status, so a red suite committed
  anyway; that happened twice. A `git commit -m "...\`baseline\`..."` ran `baseline` as a command and
  ate the word. The same class produced the day's first false success, where a crashed Modal run
  reported `exit 0` because it was piped through `tail`.
  *Trigger: without this, validation gates and audit trails fail open; with this, `make check &&
  commit` with no pipe, and heredocs for any message containing backticks.*

- 2026-07-29 · **A bigger machine masks a bug, and the cost outlives the bug.** `TRAIN_GPU` was
  raised A10G → A100-40GB to get past an OOM whose real cause was the model being left in eval mode,
  so gradient checkpointing never engaged. Once fixed, runs peaked at 13.5–14.5 GB — inside an L4's
  24 GB. The larger tier was never needed and stayed on for ~1.5 h of runs at 3–4× the rate.
  *Trigger: without this, an infrastructure workaround becomes a permanent multiplier on every later
  run; with this, any capacity raised to diagnose something is re-measured and lowered once the
  cause is found.*

- 2026-07-29 · **Log spend at the point of incurring, or the replan trigger fires unnoticed.**
  Desideratum 17 says exactly this. Roughly twenty Modal invocations across two days went unlogged
  while debugging; the gap was flagged twice and not acted on. Phase 0.1's $5 line was passed
  somewhere around 60–160% before anyone checked, and its own `>50%` replan trigger had already
  fired while runs were still being launched.
  *Trigger: without this, a budget rule exists on paper and is discovered after the money is gone;
  with this, the trigger fires while there is still budget to replan with.*
