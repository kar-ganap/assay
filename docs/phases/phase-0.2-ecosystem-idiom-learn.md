# `/learn` — Phase 0.2

> The phase-completion ritual (`CLAUDE.md` §14), written before merge. Process stream only — problem
> findings live in the retro. Items marked EXECUTED were applied in the same commit as this document.

## What worked (specific)

**Verifying the API against source, as the phase description demanded.** `stages.md` said *"verify
the API against source; the docs are thin."* Doing it turned an assumed choice into an evidenced one:
`vf-init` ships only a v0 template, every published env is v0, and `primeintellect/reverse-text`'s
manifest lists a `reverse_text_v1.py` that **is not in the published archive**. Reading the source
also produced the decisive de-risking fact — `v1/legacy.py` serves a v0 env over the *same* v1
protocol, so the choice forecloses nothing.

**Failing tests before any code, with the drift guard first.** The vendoring constraint (3.1 in the
retro) made grader drift the phase's central risk, and the tests were written to attack exactly that
before a line of the environment existed.

**The integrality check.** *Is `metric × candidate_n` an integer?* settled the denominator of someone
else's metric in one command, on 24/24 steps, with no access to their source. Cheap, decisive, and
reusable on any reported average.

**Pre-registering the A/B's prediction — including its failure branches — before launching.** The
prediction was wrong. Because both branches were written down first, the falsification was a result
rather than a disappointment, and the tidy write-up it killed never got written.

**Checking "held out" was actually held out.** 1 shared prompt in 512 against 1.26 expected by
chance. Two minutes, and it removed the only alternative explanation for a 1.0000 eval.

## What caused friction

**Two "bugs" that were my inspection, not the code** (`env.rubric.funcs` empty, `env.dataset` None).
Both correct behaviour; both cost a debugging cycle that reading `MultiTurnEnv.__init__` would have
avoided. Same shape as Phase 0.1's recurring lesson, arriving from the other direction: there I
inferred from tracebacks instead of measuring; here I inferred from an object's surface instead of
reading its constructor.

**Three undocumented free-tier gates, two sharing one opaque error.** Each had to be found by
inference and a retry. Unavoidable from outside, but it cost three launch attempts.

**A stale comparator in my own plan.** The gate was written against "base rate 0.433", which came
from the calibration sweep at `max_new_tokens=256` on a different prompt set. The like-for-like
figure was run 7's own first-ten-step reward, 0.571 ± 0.019. Caught only because the observed 0.5879
looked wrong against 0.433 and I checked which number was at fault.

**Log windows slide.** Intermediate eval readings were lost twice before I started appending them to
a file. The run IDs themselves lived only in a chat session until a laptop-sleep question exposed it.

## Rule changes

### `[ADD]`

1. **Verify the denominator of any metric imported from another stack before comparing it to ours.**
   *Trigger: without this, `prime-rl`'s filtered `Reward` gets compared to our unfiltered
   `true_reward` and the difference is reported as a result; with this, the integrality check settles
   it in one command.*

2. **Vendored code carries a fingerprint assertion, not a code review.**
   *Trigger: without this, a published artifact drifts from the measured original and every number
   quoted about it silently stops applying; with this, `grader_fingerprint()` fails the build.*

3. **Capture streamed run output to a file as it arrives; never rely on a log window.**
   *Trigger: without this, intermediate readings are lost to scrollback and the run must be repeated;
   with this, the trajectory is a committed artifact.*

### `[MODIFY]`

4. **EXECUTED — `docs/stages.md`'s Phase 0.2 description.** It specifies `SingleTurnEnv` → `ToolEnv`
   and predates verifiers v1 (2026-07-10). Amended to record the v0/v1 split, the evidence for v0 at
   0.2, and that **the decision is revisited at 1.1** when `bisect` brings the tools, sandbox and
   timeouts v1 was built for — two of the grid's four axes being harness concerns.

5. **EXECUTED — `docs/pre-registration.md` gains a Stage-2 design pin.** Any grid on `prime-rl` must
   override the default pre-batch filter list, or ablation-D-style dead-group measurements are
   invisible. Now measured, not inferred from a commented-out config block.

### `[DELETE]`

Deletion is mandatory to *consider*, not to perform (`CLAUDE.md` §14). Two candidates examined, **one
executed, one declined.**

6. **EXECUTED — the plan's "base rate 0.433" comparator.** Deleted from the gate wherever it is used
   as the training comparator. It is a real measurement of a different thing (calibration sweep,
   `max_new_tokens=256`, different prompt set) and leaving it in invites the next reader to score a
   training run against it. Replaced with run 7's own 0.571 ± 0.019.

7. **DECLINED — `train-binary.toml`**, the first G4 config, whose run proved unscoreable. Tempting to
   remove as a dead end. Kept: it is the artifact that *demonstrates* the filtered-metric problem,
   and its header now records why. Deleting it would leave the finding in prose with no reproducible
   companion.

## Conceptual-doc cleanup proposed

- **`docs/conceptual.md`** — the dead-group argument should now be careful to claim only what
  survived: dead groups cost compute (measured, Phase 0.1) but reclaiming them does **not** explain a
  performance gap (falsified here). The stronger framing from Phase 0.1's retro — that dead groups
  degrade gradient *quality* as the policy converges — is untouched by this result and should carry
  the weight.
- **`docs/related-work.md`** — `prime-rl`'s filter suite (`zero_advantage`, `gibberish`,
  `repetition`) is direct evidence that these are recognised production problems, and needs a
  first-hand read at the Phase 0.5 gate before the project frames its contribution against them.

## Tool / permission allowlist additions

- `prime env list | info | inspect | status`, `prime train logs` — read-only, used constantly.
- `prime env install` — writes only to its own venv; verified it left our `.venv` pins intact.
- **Not** proposed: `prime env push`, `prime train <config>`, `prime env delete`. The first two
  publish or spend; the third destroys. The friction is doing its job.
