# Provenance correction — the M1 screen artifacts

> Both files in this directory carry a **wrong `model_id` and `model_revision`**. The measurements
> are sound; the label on them was not. Raw artifacts are never modified (`CLAUDE.md` §9), so the
> correction lives here instead of in the JSON.

## What is wrong

`screen_remote` built its provenance dict with the caller's `_provenance()` spread **last**:

```python
"provenance": {"model_id": model_id, ..., **provenance}   # BUG
```

`_provenance()` carries the module-level `MODEL_ID` / `MODEL_REVISION`, which are Phase 0.1's
Llama-3.2-1B-Instruct. Spreading it last overwrote the screen's own values. Both artifacts therefore
claim:

```json
"model_id": "meta-llama/Llama-3.2-1B-Instruct",
"model_revision": "9213176726f574b556790deb65791e0c5aa438b6"
```

Neither ran that model. **Fixed** at the commit that adds this file: the caller's provenance is now
spread first and the screen's own values override it.

Every other provenance field is correct, including `use_chat_template: false` — these are base
checkpoints and `HFSampler` was constructed accordingly.

## What actually ran

| file | actual `model_id` | actual `model_revision` |
|---|---|---|
| `screen-countdown-Qwen2.5-1.5B-seed0.json` | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| `screen-countdown-Qwen2.5-3B-seed0.json` | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |

## Why that is recoverable rather than guessed

Three independent paths agree, none of which relies on the corrupted field.

1. **The recorded `git_sha` is the pin.** Both artifacts record
   `git_sha: 9a24381f6fa55948468316600fb0bfafc2e9decd` with `git_dirty: false`. `SCREEN_MODELS` at
   that commit is the table above, and `screen_countdown`'s default `models` argument selects both
   entries. The revision hashes were never lost — they were pinned in code, and the code is pinned by
   the artifact.

2. **The filenames are derived from the true `model_id`.** The tag is built as
   `f"screen-countdown-{model_id.split('/')[-1]}-seed{seed}"` from the *local* variable, which the
   bug never touched — only the dict entry was clobbered.

3. **Peak memory refutes the recorded label on its own.** The two runs report 5.93 GB and 9.79 GB
   peak CUDA memory on the same code path, same `n_prompts`, same `k`, same `max_new_tokens`. One
   model cannot do that. And bf16 weights alone put Llama-3.2-1B at ~2.5 GB, so 9.79 GB is not
   reachable by it under any KV-cache assumption. The larger figure is consistent with Qwen2.5-3B
   (~6.2 GB of weights); the smaller with Qwen2.5-1.5B (~3.1 GB).

## The lesson this cost

The numbers were right and the label was wrong, which is worse than a crash: a crash stops you, a
mislabelled artifact enters analysis. `experiments/README.md` already says a run whose manifest does
not identify its code cannot enter analysis — this is the same rule one field over, on the model.

A dict spread is an ordering-sensitive silent overwrite with no type error and no test failure. It
was caught by reading the artifact rather than by any check, which is the third time this stage that
**the error was only visible in the rendered artifact, not in the code or the logs**.
