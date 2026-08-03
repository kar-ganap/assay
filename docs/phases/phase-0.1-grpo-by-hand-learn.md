# `/learn` — Phase 0.1

> The phase-completion ritual (`CLAUDE.md` §14), written before merge. Process stream only — problem
> findings live in the retro. The lessons themselves are already appended to `tasks/lessons.md`; this
> records what worked, what caused friction, and the rule changes, **including deletions**. Items marked EXECUTED were applied in the same commit as this document.

## What worked (specific)

**Pinning the task by measurement rather than intuition.** The calibration sweep chose `add-3digit`
on `dead_group_fraction`, and the original "≥5% at k=8" criterion turned out to be stated on the
wrong statistic with no ceiling. Had the rule been followed as written, the whole ladder would have
run on a task that saturates to ~100% dead groups within ten steps.

**Two failure branches on every pre-registered signature.** *Prediction wrong* (a finding) versus
*rig broken* (a bug) was added after the LR-probe episode, and it earned its place four times over.
Ablation A's reversal was correctly routed to "finding"; the probe's mean-agreement check correctly
refused to proceed twice, once for a real reason (length normalisation) and once for a different one
(unresolved means) that then forced a third branch into existence.

**Writing the pre-registration immediately before running.** Twice this caught a bad gate *before* it
cost anything: the flat `≥ 2.0` threshold was mis-set for the operating point (theory predicts 1.75
at `p = 0.43`, below its own pass mark), and the centred-cosine "fix" was shown to erase the contrast
rather than the confound while drafting its prediction. Both would have been expensive to discover
after the fact.

**Decomposing a statistic before believing it.** The first probe matched theory to 4.6% and was very
nearly banked as a confirmation. Splitting `NSR = V/‖ḡ‖²` showed the agreement was coincidental —
theory says *signal flat, noise down 1.87×*; reality was *noise up 2.34×, signal up 4.16×*.

**Running three seeds.** Three claims died between n=1 and n=3, each a plausible reading of one seed.
This is the cheapest, most repeatable lesson of the phase.

## What caused friction

**Diagnosing by inference from a traceback.** `HFPolicy` bring-up took six failures and roughly half
the phase's wasted spend. Every fix targeted the last stack line; the real causes — checkpointing
never engaging, a missing attention mask — sat untouched until someone computed expected-vs-observed
memory and read the raw completions.

**Warnings that don't block.** Three instances, one pattern, ~$2.50 in unusable runs. Now fixed at
the source rather than by care.

**Verifying in the wrong medium.** A LaTeX log grep passed a document with eleven broken
cross-references; reading the figure code showed nothing while the rendered PNG showed the bug
immediately. Both are now checked in the medium the error appears in.

**Unlogged spend.** Roughly twenty invocations across two days went unlogged while debugging, and the
`>50%` replan trigger fired unnoticed. Reconciled retrospectively from per-step wall clock, which
worked — but only because every run happened to persist its own timings.

**Undefined project vocabulary.** The user had to ask three separate times what `r̄`,
"degenerate/hackable", and "proxy vs true" meant in a document written for them.

## Rule changes

### `[ADD]`

1. **Screen `E[∇log π] = 0` before comparing any two gradient estimators.**
   *Trigger: without this, a baseline ablation measures an estimand difference and reports it as a
   variance difference; with this, the premise underneath every baseline argument is checked first.*

2. **Any claim about a direction requires a resolution check on the mean** (`NSR/N`) before the
   direction is compared.
   *Trigger: without this, a diagnostic blames the estimators for a sample-size problem; with this,
   `underpowered` and `rig_broken` stay distinct, and only one of them is fixed by more batches.*

3. **A run's cohort is a content hash of the code that produced it, not the commit SHA.**
   *Trigger: without this, seeds are discarded (or worse, pooled) on the basis of edits to files the
   run never executed; with this, "same code" means what it says.*

### `[MODIFY]`

4. **`CLAUDE.md` §12.3 (reproducibility)** — "results regenerate from committed code + pinned
   parameters" should read **"…from committed code, pinned parameters, and committed *data*."** The
   old wording was satisfied by a figure script that read gitignored `raw/`, which no clone could
   reproduce.

5. **`CLAUDE.md` §15** — *"Prime Sprints is running a reward-hacking track now"* is **stale**. The
   free compute is live and requires a **public** environment; the sprint's review window closed
   ~2026-06-20. Replace with the verified statement and the public-environment condition.

6. **EXECUTED — `CLAUDE.md` §10.3 and `docs/desiderata.md`, the ≥3-seed rule.** It read as a
   *reporting* standard ("seed bands beside every effect size"). Promoted to a *claim* standard:
   **no directional claim from n=1** — not "reported with a caveat", not made. Three claims died
   between n=1 and n=3 this phase, which is what earned it.

### `[DELETE]`

Deletion is mandatory to *consider* (`CLAUDE.md` §14), not to perform. Three candidates were
examined; **one is executed, two are declined with reasons.** The first draft of this section claimed
the CV metrics were "read by nothing" — that was **wrong**, and checking before deleting is the only
reason it did not become a bad commit.

7. **DECLINED — `grad_norm_cv` / `grad_norm_cv_detrended`.** Rationale for deletion was real: they
   were ablation A's fallback metric, measured as worse than the cosine (seed band 40–46% of value
   against 13%), and are now superseded entirely by the probe. But they are **not** unreferenced —
   four tests in `test_crawl_runlog.py` cover them, and they remain honest observability on
   grad-norm stability. Deleting working, tested code to satisfy a ritual is over-engineering with
   the sign flipped. **Kept, with the plan's own note that the cosine beat them standing as the
   record of why they are not the metric.**

8. **DECLINED — `clip_epsilon` / `clipping_is_active`.** Genuinely inert: rung 4 was cut because the
   importance ratio is identically 1 under the pinned single-epoch design. But the switch is
   threaded through `loop.py`'s module docstring in three places, and `loop.py` is user-written
   (§7). More importantly, `epochs_per_batch > 1` is a live option for Walk, at which point clipping
   stops being inert. **Kept — it is dormant, not dead.**

9. **EXECUTED — `CLAUDE.md` §15's stale Prime Sprints claim.** *"Prime Sprints is running a
   reward-hacking track now"* was load-bearing for a $28 planning decision and is false: the review
   window closed ~2026-06-20. Replaced with the verified position — free compute live, **public
   environment required**, no successor track announced.

10. **EXECUTED — `CLAUDE.md` §12.3's reproducibility wording.** "Results regenerate from committed
    code + pinned parameters" was satisfiable by a figure script reading gitignored `raw/`, which no
    clone can reproduce. Now requires committed **data** too.

## Conceptual-doc cleanup proposed

- **`docs/conceptual.md`** — the dead-group argument is currently framed as a cost ("wasted
  compute"). §3.4 of the retro shows it is also a *gradient-quality* argument, which is a stronger
  and more interesting version. Worth rewriting once, carefully.
- **`docs/pre-registration.md`** — add the `zero_advantage` filter as an explicit design pin for
  Stage 2, and record that A's metric moved from a training-arm cosine to a fixed-policy probe.
- **`docs/related-work.md`** — Dr. GRPO now has two independent measured touchpoints from this phase
  and needs a first-hand read at the Phase 0.5 gate before either is framed against it.

## Tool / permission allowlist additions

- `modal app list`, `modal app logs`, `modal app stop --yes` — used constantly, read-mostly.
- `pdflatex`, `pdftotext`, `pdfinfo` — the tutorial build and its verification gate.
- `prime train models`, `prime wallet`, `prime whoami` — read-only Prime Intellect queries.
- **Not** proposed: `modal run`, `prime train <config>`, `git commit`. Each spends money or writes
  history, and the friction is doing its job.
