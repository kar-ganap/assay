#!/usr/bin/env python3
"""Pull R1's saturation curves off the Prime Intellect API into committed, regenerable form.

The API keeps a ~1.5-2.4 MB metrics blob per run, 600 keys wide, and 15 runs of that is ~25 MB of
mostly-timing telemetry -- too big to commit and mostly irrelevant to the result. This distils each
run to the six series the analysis actually reads and writes one tidy CSV, so
`experiments/README.md`'s contract ("every paper number regenerates from `results/` via a committed
script") holds without vendoring the whole blob.

Two files land in ``experiments/phase-0.4-r1/results/``:

``curves.csv``    one row per (run, step): the train and eval hack rates plus the filter telemetry
                  that explains why four runs aborted.
``manifest.json`` what each run WAS -- run id, batch, seed, status, and the config the server
                  actually registered (not what the TOML said). The env ``version_id`` is the
                  load-bearing field: it is what proves batch 1 and batch 2 scored the same grader.

The API is the only source for this data and runs age out, so treat the committed CSV as the
archive: `scripts/score_r1.py` reads the CSV, never the API.

**The metrics endpoint settles after a run goes terminal.** A fetch taken immediately after the last
run flipped to COMPLETED did not match a fetch taken minutes later -- trailing eval rows were still
being flushed -- so ``--check`` failed against data that was not wrong, only early. Fetch after the
dust settles, and treat a ``--check`` failure as "re-fetch and diff" rather than as corruption.

Usage::

    uv run python scripts/fetch_r1_curves.py            # refresh from the API
    uv run python scripts/fetch_r1_curves.py --check    # fail if the API disagrees with the CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "phase-0.4-r1" / "results"
CURVES = RESULTS / "curves.csv"
MANIFEST = RESULTS / "manifest.json"

ENV_ID = "gkartik/assay-hackword@0.1.0"

#: (csv column, metrics key suffix under ``train|eval/<env>/all/``). ``all`` is the UNFILTERED
#: population -- ``effective`` is post-filter and reports 1.0 over survivors precisely where
#: saturation is being measured, which is the bias the configs warn about.
TRAIN_SERIES: tuple[tuple[str, str], ...] = (
    ("train_reward", "rewards/reward/mean"),
    ("solved_all", "solved_all"),
    ("is_trainable", "is_trainable/mean"),
    ("zero_advantage_filtered", "filters/zero_advantage/mean"),
)
EVAL_SERIES: tuple[tuple[str, str], ...] = (("eval_reward", "rewards/reward/mean"),)

FIELDS = (
    ["run", "word", "seed", "batch", "step"]
    + [c for c, _ in TRAIN_SERIES]
    + [c for c, _ in EVAL_SERIES]
)


def _prime(*args: str) -> Any:
    """Run a `prime` CLI command that emits JSON and parse it."""
    proc = subprocess.run(["prime", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`prime {' '.join(args)}` failed: {proc.stderr.strip()[:400]}")
    return json.loads(proc.stdout)


def r1_runs() -> list[dict[str, Any]]:
    """Every R1 run, oldest first, with the batch each belongs to."""
    runs = [r for r in _prime("train", "list", "-o", "json")["runs"] if r["name"].startswith("r1-")]
    for r in runs:
        _, word, seed_tag = r["name"].split("-")
        r["_word"] = word
        r["_seed"] = int(seed_tag.lstrip("s"))
        # Batch 1 is seeds 0-2 at max_steps=100; batch 2 is seeds 3-5 at 50. Derived from the
        # registered config rather than the name, so a mislabelled run cannot hide here.
        r["_batch"] = 1 if r["max_steps"] == 100 else 2
    return sorted(runs, key=lambda r: (r["_seed"], r["_word"]))


def curve_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Distil one run's metrics blob to per-step rows."""
    metrics = _prime("train", "metrics", run["id"], "--plain")["metrics"]
    rows: list[dict[str, Any]] = []
    for entry in sorted(metrics, key=lambda e: e["step"]):
        row: dict[str, Any] = {
            "run": run["name"],
            "word": run["_word"],
            "seed": run["_seed"],
            "batch": run["_batch"],
            "step": entry["step"],
        }
        for col, key in TRAIN_SERIES:
            row[col] = entry.get(f"train/{ENV_ID}/all/{key}")
        for col, key in EVAL_SERIES:
            row[col] = entry.get(f"eval/{ENV_ID}/all/{key}")
        rows.append(row)
    return rows


def manifest_entry(run: dict[str, Any]) -> dict[str, Any]:
    """What the SERVER registered, which is the only account of a run that cannot drift."""
    env = run["environments"][0]
    ev = run["eval_config"]
    return {
        "name": run["name"],
        "id": run["id"],
        "batch": run["_batch"],
        "word": run["_word"],
        "seed": run["_seed"],
        "status": run["status"],
        "error_message": run.get("error_message"),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "base_model": run["base_model"],
        "max_steps": run["max_steps"],
        "batch_size": run["batch_size"],
        "rollouts_per_example": run["rollouts_per_example"],
        "learning_rate": run["learning_rate"],
        "max_tokens": run["max_tokens"],
        "env_id": env["id"],
        "env_version": env["version"],
        "env_version_id": env["version_id"],
        "env_args": env["args"],
        "eval_interval": ev["interval"],
        "eval_num_examples": ev["environments"][0]["num_examples"],
        "eval_rollouts_per_example": ev["environments"][0]["rollouts_per_example"],
        "pre_batch_filters": (run.get("run_config") or {}).get("pre_batch_filters"),
    }


def _render(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare the API against the committed CSV")
    ns = ap.parse_args()

    runs = r1_runs()
    if not runs:
        print("no r1-* runs visible to this account", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    manifest = []
    for run in runs:
        rows.extend(curve_rows(run))
        manifest.append(manifest_entry(run))

    versions = sorted({m["env_version_id"] for m in manifest})
    if len(versions) != 1:
        print(f"REFUSING: runs span {len(versions)} env versions {versions} — not poolable",
              file=sys.stderr)
        return 1

    body = _render(rows)
    payload = {
        "env_version_ids": versions,
        "n_runs": len(manifest),
        "n_rows": len(rows),
        "runs": manifest,
    }
    blob = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    if ns.check:
        drift = []
        if not CURVES.exists() or CURVES.read_text() != body:
            drift.append(f"  {CURVES.name}")
        if not MANIFEST.exists() or MANIFEST.read_text() != blob:
            drift.append(f"  {MANIFEST.name}")
        if drift:
            print("committed data does not match the API:\n" + "\n".join(drift), file=sys.stderr)
            return 1
        print(f"{CURVES.name} and {MANIFEST.name} match the API ({len(rows)} rows)")
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    CURVES.write_text(body)
    MANIFEST.write_text(blob)
    print(f"wrote {CURVES.relative_to(RESULTS.parents[2])} ({len(rows)} rows, {len(manifest)} runs)")
    print(f"wrote {MANIFEST.relative_to(RESULTS.parents[2])} (env version {versions[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
