# Phase 0.2 — The ecosystem idiom

> Plan locked 2026-08-03, before any code. Branch: `phase-0.2-ecosystem-idiom`.
> Approved scope: **port + eval + push + train.** Predecessor: `phase-0.1-grpo-by-hand-retro.md`.

## Purpose

Phase 0.1 proved gradients flow with a **hand-rolled** GRPO loop — deliberately, because `CLAUDE.md`
§7 requires the mechanics be learned by writing them. Every phase from Walk onward runs on the
ecosystem stack (§6), and the project's central artifact — a **grader factorial over one task set** —
must be expressed in that stack's idiom.

0.2 performs that translation on the one task we understand completely.

**Why this is more than a port.** Phase 0.1 measured `add-3digit` exhaustively: run 7 going
**0.571 ± 0.019 → 0.923 ± 0.018** (n=3), three grader variants, four ablations. (A *calibration
sweep* separately measured a 0.433 pass rate at `max_new_tokens=256` on a different prompt set —
a different quantity, and not the comparator for a training run. See G4.) It is therefore the
only task where an *independent trainer's* result is checkable. If `prime-rl`'s GRPO learns from our
`verifiers` environment, that validates the port **and** retroactively cross-checks the hand-rolled
loop. No later phase gets that check as cheaply.

## Design

### The environment package

```
environments/assay_add3digit/          # UNDERSCORES: `prime env install` resolves a local env by
                                       # directory name and requires Python package convention,
                                       # while the project name stays dashed (assay-add3digit)
  pyproject.toml          name, tags, pinned verifiers, [tool.verifiers.eval] defaults
  README.md               datasets / task / rubric / metrics, per the vf-init template
  assay_add3digit.py      load_environment(...) -> vf.SingleTurnEnv
```

**Reuse, do not reimplement.** The gate is only meaningful if the ported task is the *same* task:

| from | to |
|---|---|
| `crawl/tasks.py::ArithmeticFamily.generate` | the dataset builder (already deterministic from seed) |
| `crawl/rewards.py::grade_binary` | `r_binary` reward function |
| `crawl/rewards.py::grade_format_only` | `r_format` reward function |
| `crawl/rewards.py::grade_tiebroken` | `r_tiebreak` reward function |
| `crawl/rewards.py::grader_fingerprint()` | asserted in a test, so grader drift is caught |

### The design payoff

`vf.Rubric(funcs, weights)` is exactly the shape Walk's grader factorial needs: *"grader
configurations over one task set"* becomes **a weight vector over a fixed function list**. 0.2 builds
that mechanism on a task whose answers are already known.

### Pins (mirror Phase 0.1, so the comparison is like-for-like)

| | value | provenance |
|---|---|---|
| model | `sprints/Llama-3.2-1B-Instruct` | free tier; base model matches 0.1's `9213176726f5` |
| `batch_size` | 128 | = 0.1's 16 prompts × G=8 |
| `rollouts_per_example` | 8 | = 0.1's `group_size` |
| `max_steps` | 200 | = 0.1 |
| `learning_rate` | 1e-5 | probed in 0.1 |
| `max_tokens` | 64 | = 0.1's `max_new_tokens` |
| task | `add-3digit`, seed 0 | pinned 2026-07-28 by the calibration sweep |

## API decision: v0, not v1 — verified against source

`stages.md` says *"verify the API against source; the docs are thin."* Done. `verifiers` 0.2.1 ships
**both** APIs. v1 splits the v0 `Environment` into `TaskSet` (data) / `Task` (behaviour + scoring) /
`Harness` (agent scaffold). The shipped harness list — `bash`, `claude_code`, `codex`,
`mini_swe_agent`, `terminus_2`, `rlm`, `pi`, `null`, `pool`, `kimi_code` — reveals its purpose:
**running one task set under many agent scaffolds.**

`add-3digit` is single-turn with no tools, so that axis is degenerate here. Evidence for v0:

1. `vf-init` ships only a `V0_ENVIRONMENT_TEMPLATE` — there is no v1 option.
2. Every published env inspected is v0. `primeintellect/reverse-text`'s `pyproject.toml` lists a
   `reverse_text_v1.py` in `[tool.hatch.build].include`, but **that file is absent from the published
   archive** — a stale include. No published v1 exemplar exists to copy.
3. `v1/legacy.py` serves a v0 env over the *same* v1 ZMQ protocol; its docstring states the
   orchestrator *"can't tell v0 from v1."* **Choosing v0 forecloses nothing.**

**Revisit at Phase 1.1**, when `bisect` arrives — tool-using, sandboxed, budgeted, which is v1's
actual shape. Note also that **two of the grid's four axes (timeout, sandbox writability) are
runtime/harness concerns**: v1 makes them first-class config; v0 would thread them as ad-hoc
constructor arguments. That decision should be made against `bisect`'s real requirements, not
anticipated now.

## Gates (pass criteria)

**G1 — local.** `make check` green: dataset determinism from seed, each Rubric reward function
agreeing with its `rewards.py` counterpart on the Phase 0.1 fixtures, `grader_fingerprint()`
asserted, rubric weights matching the requested variant. No GPU, no network.

**G2 — eval.** The environment loads and scores under real `verifiers`.

> **Adapted 2026-08-03, recorded as a deviation.** A pass-rate eval needs paid inference against a $0
> wallet, and `verifiers` cannot be installed locally (it needs `numpy>=2.1`; `torch<2.3` needs
> `numpy<2`). The check became "loads and scores under real `verifiers`", with the pass-rate
> comparison folded into G4's step-1 eval reading — which is stronger anyway: held out, 512 rollouts.

**G3 — published.** `prime env push -v PUBLIC` succeeds and `prime env info assay-add3digit` resolves
from the Hub.

**G4 — trained (the real gate).** Final true reward **≥ 0.85**.

> **Comparator corrected 2026-08-03, and the old one deleted.** This gate originally read "against
> the base rate of 0.433". That number is a real measurement of a *different thing* — the calibration
> sweep at `max_new_tokens=256` on a different prompt set — and scoring a training run against it
> compares two quantities. The like-for-like figure is Phase 0.1 run 7's own first-ten-step true
> reward, **0.571 ± 0.019**. The measured step-1 eval was **0.5879**, inside that band.

> The threshold sits below 0.1's `0.923 ± 0.018` on purpose: `prime-rl`'s GRPO is not our GRPO and
> the hyperparameters will not match exactly. The claim under test is *"an independent implementation
> learns this task from our environment"*, **not** *"the curves coincide."*

**Two failure branches** (the rule earned in 0.1 — every signature carries both):

| observation | verdict |
|---|---|
| reward never leaves the base rate | **rig broken** — the port is wrong (dataset columns, parser, reward signature). Not a finding about `prime-rl`. |
| reward moves but plateaus < 0.85 | **a finding** — record the delta and what differs (filters, hyperparameters, trainer). |

**G5 — secondary, cheap, and load-bearing for Stage 2.** Record `prime-rl`'s **default pre-batch
filter list**. The config template says filters *"replace the removed difficulty buffer"* and that
setting a slot *"overrides prime-rl's default filter list"* — implying a non-empty default. **If
`zero_advantage` is in it, dead groups are silently dropped**, which would invalidate
ablation-D-style measurements on this stack. Outcome becomes a design pin in
`docs/pre-registration.md` before any Stage-2 grid.

## Outputs

- `environments/assay_add3digit/` — the package, published to the Hub.
- Tests under `tests/`, running with zero GPU and zero network.
- Training result + curve, compared against Phase 0.1's.
- The `zero_advantage` finding, recorded as a Stage-2 design pin.
- Retro + `/learn` per §13.

## Non-goals

- **`bisect`** — Walk's substrate, Phase 1.1.
- **A v1 port** — revisited at 1.1 against `bisect`'s real needs.
- **`ToolEnv`** — same reasoning; a toy rehearsal would not inform the 1.1 decision.
- **The grader-factorial grid** — 0.2 builds the *mechanism* (Rubric weights); the grid is Stage 2.
- **`trl`** — §6 permits either; `prime-rl` via hosted training is free and exercises the Hub path
  the project will actually use.

## Risks

- **`verifiers` is WIP.** `envs/AGENTS.md` says so for both APIs. Pin the version; record it here.
- **The free tier is operational-today, not committed.** The sprint that introduced it closed
  ~2026-06-20 and no successor was announced. Fallback: paid Llama-3.2-1B at ~$0.10/run, still ~5×
  under Modal.
- **Name collision.** 1419 envs on the Hub; no `assay*` match as of 2026-08-03, but first-come.
- **Grader drift.** If the ported reward functions are not bit-identical to `rewards.py`, G4 compares
  two different tasks. `grader_fingerprint()` exists to catch exactly this.
- **A public push is irreversible in practice.** Arithmetic, so no strategic cost — but the habit
  should not carry to `bisect` without a decision.

## Change log

| date | change |
|---|---|
| 2026-08-03 | Plan locked. Scope approved as port + eval + push + train. v0 API chosen on source evidence (three findings above). Gate set at ≥ 0.85 with two failure branches. |
