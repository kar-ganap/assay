# experiments

Per-phase runs. House convention (see `../.gitignore`):

- `experiments/<phase>/raw/` — raw rollouts, generations, checkpoint pointers, per-step logs.
  **Ignored** by git.
- `experiments/<phase>/results/*.json|csv` — derived metrics. **Committed.** Every figure and paper
  number regenerates from these via a committed script (desiderata 12–13).

## Phases

| Dir | Stage | What lands here |
|---|---|---|
| `phase-0.1-grpo-by-hand/` | Crawl | seven ablation curves + the reward/entropy/KL/pass-rate traces |
| `phase-0.2-verifiers-idiom/` | Crawl | the same task under the `verifiers` spec; parity check vs 0.1 |
| `phase-1.4-screen/` | Walk | `p_hack@64` per variant; admission decisions; positive-control run |
| `phase-2.1-exploratory/` | Run | 8–12 variants × 1 seed, **per-step gap logs** |
| `phase-2.2-confirmatory/` | Run | 4 variants × 3 seeds |
| `phase-3.3-eta/` | Gallop | evals (a)–(d) per confirmatory arm |
| `phase-3.5-field-report/` | Gallop | `assay` run over ~15 Hub environments |

Reproductions live in `../reproductions/`, not here — different lifecycle (they retire assumptions
rather than produce results).

## Per-step logging is not optional

Every training run writes proxy-grader reward **and** held-out-grader reward at every step
(`raw/<run>/steps.jsonl`). The project's outcome variable is the **slope** `d(gap)/d(step)` over
steps 50–200, not a terminal value — a rising, unsaturated gap is still a clean measurement
(`../docs/pre-registration.md` §4 L3). A run without per-step logs is not usable.

## Run manifest

Every run directory carries a `manifest.json` pinning: model id **and revision hash**, sampler
settings (temperature, top-p), all RL hyperparameters, the seed, the grader-variant id, the
grader/judge model versions **and prompt hashes**, and the git SHA of the code that produced it
(desideratum 12). A run without a manifest does not enter any analysis.
