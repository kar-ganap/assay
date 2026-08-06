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

- 2026-08-02 · **A check that reports instead of blocking is not a check.** Third instance in this
  phase, and the most expensive. `modal_app.ladder` *printed* `WARNING: dirty tree` and fell straight
  through into the launch loop — once, before a loop that then launched N runs, and paired with
  `--detach`, whose entire purpose is that nobody is watching the terminal. Four 200-step runs
  (~$2.50) landed at a `git_sha` that does not identify their code. The two earlier instances were
  `make check | tail && git commit` swallowing the exit status, and the `--setting` default silently
  overriding the pinned ladder table.
  *Trigger: without this, a validation gate's enforcement mechanism is me noticing; with this, it
  raises and the run does not start.*

- 2026-08-02 · **Verify in the medium the error would appear in.** Two failures in one session that
  a code- or log-level check structurally could not see. (a) Every `\cref` to a Claim rendered as
  `?? 2.1` in the PDF; nothing was *undefined*, so LaTeX logged no warning, and my `grep undefined`
  on the log passed a document with eleven broken references. (b) The ladder figure silently pooled
  stale A100 seeds with clean ones and labelled the curve `n=2`; the code read fine and the plot
  looked healthy — it was visible only by opening the PNG.
  *Trigger: without this, a green check certifies a medium the bug does not live in; with this,
  `make pdf` greps the rendered text and figures are looked at before they are believed.*

- 2026-08-02 · **A gate's branches must be independent facts, not an `if/elif` chain.** The probe's
  verdict chained "is there an effect?" and "is it the predicted size?", so whichever matched first
  hid the other — and on the no-length-norm probe *both* were true (CI spanned 1.0 **and** excluded
  `1/(1-p)`), while only `not_measurable` was reported. Rewriting it surfaced a worse bug in the
  same function: the `falsified` branch gated on raw ordering, which would have called a ratio of
  0.859 with a CI spanning 1.0 "the baseline is actively harmful" — a direction claim from noise.
  *Trigger: without this, a verdict reports the first matching label and buries the rest; with this,
  each fact is computed and reported separately and the label is derived from them.*

- 2026-08-02 · **A right number can come from a wrong mechanism. Decompose before claiming.** The
  first probe matched theory to 4.6% (1.786 observed vs 1.872 predicted) and I nearly reported it as
  a clean confirmation. Decomposing `NSR = V/||g||^2` showed theory predicts *signal unchanged, noise
  down 1.87x* while what happened was *noise up 2.34x, signal up 4.16x* — two effects the theory does
  not contain, pulling opposite ways, landing on the right answer. Only the rig check ("do these
  estimators even share a mean?") caught it.
  *Trigger: without this, a coincidental agreement is banked as a confirmed prediction; with this,
  the components are checked before the ratio is believed.*

- 2026-08-02 · **A test double cannot exhibit what it was constructed to exclude.** `ToyPolicy`
  samples from its own distribution precisely so `E[grad log pi] = 0` holds — which is exactly why it
  could not reproduce the length-normalisation confound, and why the toy read ablation A forwards
  while Llama read it backwards. Separately, my synthetic bootstrap gave every baseline the same
  deterministic jitter, making their *ratio* constant and the `not_measurable` branch unreachable:
  the double silently vouched for a gate it never exercised.
  *Trigger: without this, a passing test certifies a branch the fixture cannot reach; with this, the
  double's own invariants are checked against the property under test before trusting a green run.*

- 2026-08-02 · **Define the project's vocabulary at first use — the words that stopped looking like
  jargon are the ones that need it.** The user had to ask three separate times in one session what
  `r-bar`, "degenerate"/"hackable", and "proxy vs true" meant in a tutorial written for them. Worse,
  `r-bar` carried *three different meanings* across six uses (group mean, batch mean, half-batch
  mean) — and conflating the first two is exactly the rung-2/rung-3 distinction the document spends a
  section explaining.
  *Trigger: without this, the terms internalised from months in the codebase are precisely the ones
  left undefined, and the document fails hardest where it matters most; with this, notation is
  defined where it is introduced and subscripted to its scope.*

- 2026-08-06 · **State a comparison's design floor before running it.** The smallest p an exact rank
  test can produce is `1/C(n_a+n_b, n_a)` — at 3 vs 3 that is exactly 0.05, so a three-seed
  comparison cannot clear a 0.05 threshold however cleanly the data splits. R1's batch 1 spent nine
  runs on the one pair that discriminated its two hypotheses without that line of arithmetic ever
  being computed, and the shortfall surfaced only when six more seeds reversed the direction. The
  floor also separates a null worth reporting from one that means nothing: at n=6 vs 6 it is 0.0011,
  so R1's p=0.29 is evidence of no effect rather than evidence of no resolution.
  *Trigger: without this, seed counts are set by convention and a phase can conclude nothing while
  looking like it concluded something; with this, the seed count is derived from the claim and an
  underpowered design is caught before it is run.*

- 2026-08-06 · **Seeds launched in one wave are one draw. Estimate variance across waves or call it
  a lower bound.** `ocean`'s three seeds launched within a second of each other and returned
  sd 2.64; the same arm across two launch waves gives 6.34. Same-wave seeds share cluster load,
  queue position and rollout staleness (batch 1's evals spanned three policy versions, batch 2's
  spanned one), so they are correlated in exactly the way an independent-draw estimate assumes away.
  A tight triple from a high-variance arm is indistinguishable from a real effect.
  *Trigger: without this, "±sd over 3 seeds" understates spread by 2.4x and the understatement is
  invisible; with this, the reported band says whether its seeds could have varied independently.*

- 2026-08-06 · **A pre-registered analysis still fails if its outcome space is incomplete.** R1's
  decision table had two rows — CONFIRMED and FALSIFIED — because the plan assumed the ordering
  would resolve. It did not, and the scorer, written before any curve existed and faithfully
  implementing the plan, returned CONFIRMED on a p=0.29 null by ordering two medians on an arm whose
  seeds span 16.9 steps. Pre-registration protects against choosing the analysis after seeing the
  data; it does nothing about an analysis with nowhere to put the actual answer, and that failure is
  harder to see because every part of the process looks correct.
  *Trigger: without this, a scorer improvises the missing cell and reports the nearest verdict it
  owns; with this, every decision table carries an explicit indeterminate row with its own action.*

- 2026-08-06 · **A criterion that avoids a threshold hides its assumptions in the sample size —
  simulate it before adopting it.** Refusing to pick alpha post-hoc, I scored a comparison on whether
  the two seed ranges overlap. It needs no threshold and looks principled; its implied false-positive
  rate is `1/C(2n,n)`, so it grows *stricter* as evidence accumulates, and its power against a real
  1-sigma effect falls from 26% at n=3 to 3.8% at n=6 to 0.05% at n=12. A rule that gets worse with
  more data cannot be a gate. Forty lines of simulation killed it in one commit.
  *Trigger: without this, an assumption-free-looking rule ships and quietly loses power as the
  project scales; with this, a proposed criterion is measured against a known effect first.*

- 2026-08-06 · **Read a platform's terminal status as a hypothesis, not as data.** Four R1 runs
  ended `FAILED: BackoffLimitExceeded` and were the phase's cleanest results: once every group scores
  1.0 the advantage is zero everywhere, the `zero_advantage` filter empties the batch, and the
  orchestrator quits after ten consecutive empty ones. They were terminated for learning the target
  behaviour too completely. The same telemetry showed `is_trainable` at 0.000 from step ~30 in
  *every* run including the five marked COMPLETED, so the distinction between the two labels was
  which runs happened to resample a stray trainable group often enough to reset a counter.
  *Trigger: without this, the four best demonstrations in a phase are discarded as infra noise and
  the five kept are believed to have trained throughout; with this, status prompts reading the curve.*

- 2026-08-06 · **Name a results field for what it computes, not for the claim it serves.**
  `r1p_confirmed` held a median-ordering check. Every reader treated it as the verdict, including
  the person who wrote it, and it took a direction reversal in a second batch to notice. Renamed to
  `r1p_ordering_holds`, the same value cannot be misread — and the rename is what forced the real
  test to be written.
  *Trigger: without this, a convenience field acquires the authority of the hypothesis it is named
  after; with this, the gap between "what this computes" and "what we want to claim" stays visible.*
