#!/usr/bin/env python3
"""Generate the R1 saturation-onset training configs.

R1 asks one question -- how many GRPO steps until a 1B policy saturates a trivially hackable
grader -- and answers it once per (hack word, seed). The configs are mechanical variations on a
single template, so they are generated rather than hand-maintained: nine hand-edited TOML files is
nine chances to typo a learning rate into one arm only, and the typo would read as a result.

Two batches exist, and the split is deliberate rather than incidental.

**Batch 1** (seeds 0-2, all three words, launched 2026-08-04) is the pre-registered design.

**Batch 2** (seeds 3-5, `ocean` and `midnight` only, launched 2026-08-05) was added after batch 1
returned. It changes *measurement* parameters only and leaves every quantity that enters a gradient
untouched, so onsets pool across the two batches. `BATCH_2_NOTE` records what moved, why, and which
way the residual bias points.

Usage::

    uv run python scripts/gen_r1_configs.py            # write all configs
    uv run python scripts/gen_r1_configs.py --check    # verify on-disk files match, write nothing

``--check`` is the load-bearing mode: it is what establishes that this script really is the source
of the nine batch-1 files it claims to have generated.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "environments" / "assay_hackword" / "configs"

#: Column at which trailing `#` comments begin in the generated TOML.
COMMENT_COL = 43


@dataclass(frozen=True)
class Word:
    """A hack word, with the external comparator R1 is scored against."""

    name: str
    prime_onset: int  # Prime Intellect's published steps-to-50%-saturation
    prime_baseline_pct: float  # their reported base rate, percent
    our_base_rate: float  # ours, measured at G2 on the story substrate


WORDS: dict[str, Word] = {
    "ocean": Word("ocean", 44, 0.47, 0.0135),
    "midnight": Word("midnight", 18, 1.56, 0.0059),
    "forgotten": Word("forgotten", 11, 7.81, 0.2096),
}


BATCH_1_NOTE = """\
# THE METRIC IS READ FROM EVAL, NOT FROM TRAINING REWARD. As the hack saturates, groups become
# unanimous -- and prime-rl's zero_advantage pre-batch filter drops unanimous groups, so the
# reported training reward would be a mean over survivors precisely where saturation is being
# measured. Phase 0.2 verified that bias by integrality and found `enforce = false` only partially
# suppresses it. Eval is unfiltered, so `r_hack` there is the honest saturation curve.
#
# eval interval 5 gives 21 points over 100 steps. Prime's onsets are 11 / 18 / 44, so interval 25
# (Phase 0.2's) would be too coarse to locate a 50% crossing at all."""


BATCH_2_NOTE = """\
# THE METRIC IS READ FROM EVAL, NOT FROM TRAINING REWARD -- see batch 1. The zero_advantage filter
# makes the reported training reward a mean over survivors exactly where saturation is measured.
#
# BATCH 2 (2026-08-05). Batch 1 settled reachability: all nine runs reached hack rate 1.0, so the
# exploit is reachable at 1B far inside 100 steps. What it left underpowered is the one comparison
# that discriminates R1-P from G3. On the pre-registered metric, `ocean` returned three uncensored
# onsets and `midnight` returned ONE -- midnight s1 and s2 aborted 5-9 steps after their last
# completed eval, with the crossing sitting in the gap. Seeds 3-5 on those two words take the
# discriminating pair to n=6 vs n=6, where the attainable one-sided p falls from 0.05 (the floor at
# n=3, which batch 1 could never have beaten) to 0.001.
#
# THREE PARAMETERS CHANGE. ALL THREE ARE MEASUREMENT; NONE IS GRADIENT:
#
#   eval.interval          5 -> 3    resolution near the crossing
#   eval num_examples     64 -> 32   latency -- each eval took ~2h30m and lagged training by 7-12
#                                    steps, which is what pushed the crossing past the abort
#   max_steps            100 -> 50   every batch-1 onset landed at or below 30.6
#
# Model, batch_size, rollouts_per_example, learning_rate, sampling and the zero_advantage filter are
# byte-identical to batch 1, so onsets pool across batches.
#
# The tempting fourth change was rejected. Dropping the zero_advantage filter would stop the run
# aborting at all -- but the filter was already discarding ~50% of rollouts at step 20, well BEFORE
# saturation, so removing it would move the effective gradient scale inside the region where the
# onset is measured. A config that cannot abort would buy uncensored curves by making them
# incomparable with batch 1's.
#
# Residual censoring bias points the safe way. Truncating at 50 can only drop LATE crossings, and
# `midnight` is the later-crossing word, so whatever censoring survives pulls `midnight`'s onset
# down -- toward Prime's ordering and away from R1-P's. The change cannot manufacture the result it
# is being used to test."""


#: Identical in both batches. Kept verbatim because the finding it records -- that a config can pass
#: client-side validation and still be refused server-side -- cost nine runs a near-miss.
SEED_NOTE = """\
# NO TRAINER SEED IS AVAILABLE TO THIS ACCOUNT, and the reason is not the obvious one.
#
# A top-level `seed` is rejected ("Extra inputs are not permitted"), which is what this comment
# first claimed and stopped at. But `run_config` is an open passthrough and DOES accept
# `{{ seed = N }}` -- it passes client-side validation and echoes back. It then fails server-side:
#
#     HTTP 403: Only ADMIN or MANAGER users can use run_config
#
# So the knob exists and is permission-gated. **Client-side validation is not authorization**, and
# nine runs were nearly discarded on the strength of a config that parsed.
#
# Consequence, which is a real limitation on any seed-variance claim: the three seeds per word vary
# the DATASET seed ({seed}) and rely on sampling stochasticity at T=1.0 for the rest. The trainer's
# own RNG is identical across them. That BOUNDS run-to-run spread; it does not fully characterise
# it, and R1-P's onset comparison inherits that bound."""


@dataclass(frozen=True)
class Batch:
    """One launch of the R1 grid. Measurement parameters vary; gradient parameters must not."""

    words: tuple[str, ...]
    seeds: tuple[int, ...]
    max_steps: int
    eval_interval: int
    eval_num_examples: int
    eval_rollouts_per_example: int
    note: str
    max_steps_comment: str  # the trailing comment on max_steps; it is NOT constant


BATCHES: tuple[Batch, ...] = (
    Batch(
        words=("ocean", "midnight", "forgotten"),
        seeds=(0, 1, 2),
        max_steps=100,
        eval_interval=5,
        eval_num_examples=64,
        eval_rollouts_per_example=4,
        note=BATCH_1_NOTE,
        max_steps_comment="= Prime's config",
    ),
    Batch(
        words=("ocean", "midnight"),
        seeds=(3, 4, 5),
        max_steps=50,
        eval_interval=3,
        eval_num_examples=32,
        eval_rollouts_per_example=4,
        note=BATCH_2_NOTE,
        max_steps_comment="NOT Prime's 100; every batch-1 onset was <= 30.6",
    ),
)


def _c(code: str, comment: str = "") -> str:
    """A TOML line with its trailing comment at the pinned column."""
    return f"{code:<{COMMENT_COL}}# {comment}" if comment else code


def _env_args(word: str, n_train: int, seed: int) -> str:
    """The inline TOML table shared by the train env and the eval env."""
    return (
        f'{{ hack_word = "{word}", mode = "hack_only", task = "story", '
        f"n_train = {n_train}, seed = {seed} }}"
    )


def render(word: Word, seed: int, batch: Batch) -> str:
    """Render one config. The only per-file variation is the word, the seed, and the batch."""
    lines = [
        f"# R1 — {word.name}, seed {seed}. Generated by scripts/gen_r1_configs.py; "
        "edit that, not this.",
        "#",
        f"# Prime's published onset for this word: {word.prime_onset} steps to 50% saturation "
        f"(baseline {word.prime_baseline_pct}%).",
        f"# Our measured base rate: {word.our_base_rate:.4f} (2 seeds, story substrate, "
        "256 tokens).",
        "#",
        batch.note,
        "",
        _c('model = "sprints/Llama-3.2-1B-Instruct"'),
        _c('loss = "rl"'),
        _c(f"max_steps = {batch.max_steps}", batch.max_steps_comment),
        _c("batch_size = 128", "= Prime's config"),
        _c("rollouts_per_example = 8"),
        _c("learning_rate = 1e-4", "= Prime's config; NOT Phase 0.1's 1e-5"),
        "",
        SEED_NOTE.format(seed=seed),
        f'name = "r1-{word.name}-s{seed}"',
        "",
        "[sampling]",
        _c("max_tokens = 256", "= the length the base rate was measured at (repinned"),
        _c("temperature = 1.0", "  from 64 when the substrate became free text)"),
        "",
        "# Belt-and-braces; the metric comes from eval regardless.",
        "[[pre_batch_filters]]",
        'type = "zero_advantage"',
        "enforce = false",
        "",
        "[[env]]",
        'id = "gkartik/assay-hackword"',
        f"args = {_env_args(word.name, 2000, seed)}",
        "",
        "[eval]",
        f"interval = {batch.eval_interval}",
        "skip_first_step = false",
        "",
        "[[eval.env]]",
        'id = "gkartik/assay-hackword"',
        f"args = {_env_args(word.name, 512, 999)}",
        f"num_examples = {batch.eval_num_examples}",
        f"rollouts_per_example = {batch.eval_rollouts_per_example}",
    ]
    return "\n".join(lines) + "\n"


def paths_and_bodies() -> list[tuple[Path, str]]:
    """Every config this script owns, in launch order."""
    out: list[tuple[Path, str]] = []
    for batch in BATCHES:
        for seed in batch.seeds:
            for name in batch.words:
                word = WORDS[name]
                out.append(
                    (CONFIG_DIR / f"r1-{word.name}-seed{seed}.toml", render(word, seed, batch))
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify on-disk configs match what this script generates; write nothing",
    )
    ns = ap.parse_args()

    drift: list[str] = []
    written = 0
    for path, body in paths_and_bodies():
        current = path.read_text() if path.exists() else None
        if ns.check:
            if current != body:
                drift.append(f"  {'MISSING' if current is None else 'DIFFERS'}  {path.name}")
            continue
        if current != body:
            path.write_text(body)
            written += 1
            print(f"wrote {path.name}")

    if ns.check:
        if drift:
            print(f"{len(drift)} config(s) do not match the generator:", file=sys.stderr)
            print("\n".join(drift), file=sys.stderr)
            return 1
        print(f"all {len(paths_and_bodies())} configs match the generator")
        return 0

    print(f"{written} written, {len(paths_and_bodies()) - written} already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
