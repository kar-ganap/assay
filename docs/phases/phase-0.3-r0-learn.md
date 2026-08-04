# `/learn` — Phase 0.3

> The phase-completion ritual (`CLAUDE.md` §14), written before merge. Process stream only — problem
> findings live in the retro. Items marked EXECUTED were applied in the same commit as this document.

## What worked (specific)

**Screening before spending, and the ratio it bought.** $2.21 established that the other $7.79 would
have purchased an uninterpretable run. Every one of the three measurements was cheaper than its
estimate ($1.57/$0.15/$0.49 against $1.50/$0.50/$0.50), and each retired a distinct objection.

**Pre-registering M3's criteria as their own commit, before the settings existed.** Not a claim in
prose — git records the ordering, so "the criteria were not tuned to the result" is checkable by
anyone. The tie-break was fixed in the same commit, which is the part that would otherwise have been
decided by eye once the table was up.

**Writing both failure branches before each run.** M1's rig-broken branch (`parse_fail > 0.5`) is why
"the models emit legal expressions that miss" is a *finding* rather than a suspicion. M3's negative
branch is why nothing-admitted was a result rather than a disappointment — and it is also why the
retro can say the branch was **right for the wrong reason**, which is only visible because the reason
was written down in advance.

**Controls that can only pass one way.** M2's HF-vs-HF check had to be *exactly* 0.0, not "small".
That is unfakeable by a subtly wrong harness, and it is what licensed reading the vLLM number at all.
`independence_ratio` = 0.968 did the same job for the extrapolation to L=1024 — it measured the
assumption instead of making it.

**Checking the veRL pitch first-hand instead of trusting it.** "They have reproducible examples"
sounded like it solved R0. Reading `examples/data_preprocess`, the `verl-recipe` submodule and the
README's section headings showed it does not — no Countdown, no tinyzero recipe, TinyZero listed
under "Awesome Projects Built with `verl`". The same read found `rollout_correction`, which is worth
more than the thing I went looking for.

## What caused friction

**Three defects that would each have wasted a paid run; two caught, one not.** `vllm==0.11.0` does
not exist (caught by querying PyPI). `grade_countdown` returns a `Grade`, not a float (caught by
reading the signature). vLLM V1 needs `nvcc`, which `debian_slim` lacks — **not** caught, and it
died after the GPU was allocated. The pattern: I verified the things I thought to verify, and the
one that bit was an environment assumption I never articulated.

**Two silent artifact-integrity bugs, same shape, one week apart.** The provenance dict-spread
(`**provenance` last, clobbering `model_id`) made both M1 artifacts claim they ran Llama-3.2-1B. The
volume tag collision made M3 overwrite M1's 3B artifact. Neither raised; both produced *plausible*
output. Both were found by reading an artifact, not by any check.

**A test of mine that was wrong and asserted through.** M2's length-independence test varied 16 vs
256 *total* tokens while claiming to vary sequence length, so the Bessel correction moved `std` by
3%. It failed, which is the only reason I looked.

**Three different cost estimates for M3 in one conversation** — $0.40, then ~$1.00, then $0.50 —
because I revised by intuition twice before doing the arithmetic from M1's measured throughput. The
first and last agreed; the middle one was noise I introduced.

**I compared two incommensurable quantities and stated a ratio.** "0.620 is 1.3× worse than run 7's
0.472" — but 0.472 is a *training mean* and 0.620 is *step 0*; run 7's step-0 figure was 0.012. The
plan's own band-justification sentence makes the same conflation, which is how I inherited it. Third
instance this project of a number being carried across contexts where it means something different
(after `prime-rl`'s filtered `Reward` and the 0.433 calibration-sweep comparator).

## Rule changes

### `[ADD]`

1. **A pre-registered numeric threshold is implemented with boundary tolerance, and its test asserts
   the float hazard still exists.**
   *Trigger: without this, `ratio >= 3.0` rejects a setting that lands exactly on the pre-registered
   line, because `0.30/0.10` is `2.9999999999999996` and 240/80 successes out of 1600 produce it; with
   this, the line means where it says and the test fails if someone "simplifies" the comparison.*

2. **Before interpreting a statistic's magnitude, check whether its sign or size is forced.**
   *Trigger: without this, "vLLM over-scores its own samples" gets written down when Gibbs makes the
   sign inevitable for any two distinct implementations and `μ = −σ²/2` makes the magnitude a
   restatement of the spread; with this, interpretation starts from what the data can actually
   distinguish.*

3. **An artifact tag encodes what was measured, not only which model produced it.**
   *Trigger: without this, two screens of the same model collide and the second silently overwrites
   the first (M3 over M1, 2026-08-03); with this, `fetch` recovers the artifact it names.*

4. **A reproduction target's paper or README is read for a publishable number *before* the target is
   admitted to the ledger.**
   *Trigger: without this, R0 sits on the never-cut list for months while being structurally
   unsatisfiable — `reproductions/README.md` demands "Original number + Delta" and TinyZero publishes
   neither; with this, a five-minute read disqualifies it before a budget line is drawn.*

### `[MODIFY]`

5. **EXECUTED — the dead-group band's justification sentence.** It reads "≤0.50 workable — run 7
   learned at a mean dead fraction of 0.472", which compares a *training mean* to the *step-0*
   quantity M1/M3 measure. Run 7's step-0 figure was 0.012. The band stays (it was pre-registered and
   applying it was correct); the justification now states which quantity it is anchored on, and notes
   that `dead` is U-shaped in `p` so saturation and starvation are indistinguishable to it.

6. **PROPOSED, for the user — `CLAUDE.md` §6 gains `verl` as a backend candidate.** Not a switch:
   Phase 0.2 validated `prime-rl` at $0 on the free tier and verl's examples assume 64×H800. But verl
   implements DrGRPO (where Phase 0.1's length-normalisation finding independently pointed), GSPO,
   DAPO, and the `rollout_is_*` surface M2's result now calls for — and it appeared in this repo
   exactly once, describing someone else's stack. §6 is governance, so this is proposed, not applied.

### `[DELETE]`

Deletion is mandatory to *consider*, not to perform (`CLAUDE.md` §14). Three examined, **one
executed, two declined.**

7. **EXECUTED — R0 from the never-cut list.** `docs/stages.md` and `reproductions/README.md` amended.
   Keeping it would leave a permanently-blocked item claiming priority over work that can actually
   run, and its stated purpose is now partly served by Phase 0.2's independent-trainer cross-check.
   What it uniquely would have retired — *the loop learns a task requiring search* — is recorded as
   an open limitation rather than quietly dropped.

8. **DECLINED — `CountdownFamily`, `grade_countdown` and their 26 tests.** Tempting as dead code now
   that R0 retires. Kept: §12.3 requires results regenerate from committed code, and this code *is*
   how M1's and M3's artifacts regenerate. Deleting it would leave the phase's findings as numbers in
   a document with nothing behind them.

9. **DECLINED — the band's "marginal 0.50–0.75" row.** It fired (3B/cd-3 at 0.620) and then did not
   change the decision, so it reads as an actionable path that was not actionable. But it was
   pre-registered, and deleting a band row after seeing which row you landed in is exactly the move
   pre-registration exists to prevent. Kept, with the retro recording that it was uninformative here.

## Conceptual-doc cleanup proposed

- **`docs/conceptual.md`** — the dead-group fraction now has out-of-sample evidence (predicted 0.851,
  measured 0.845, on a different model/task/scale). That upgrades it from "the criterion we used to
  pick a task" to "a portable screening statistic", which is a stronger and more central claim than
  the doc currently makes. Worth stating, since the whole thesis is that cheap pre-training
  measurements predict expensive post-training outcomes.
- **`docs/pre-registration.md`** — §4's reachability ladder should record that M1/M2/M3 are the
  ladder applied to *ourselves*, and that the step-0-vs-emergence tension (retro §5) is a live
  limitation of the `p_hack@64` screen, not only of Countdown.
- **`docs/stages.md`** — Phase 0.3's row now describes a phase that measured rather than reproduced.

## Tool / permission allowlist additions

- `curl` against `api.github.com` (unauthenticated, read-only) — used to verify repo contents
  first-hand rather than trusting a README summary. Cheap and became load-bearing.
- `modal app list` / `modal app logs <id>` — read-only, and the only way to recover a measured wall
  clock after a detached run. Used for every spend entry this phase.
- **Not** proposed: `modal run`. It spends. The friction is doing its job.
