"""What the remaining science costs, bottom-up from measured anchors.

**Why this exists.** `docs/stages.md`'s stage budgets ($17 / $20 / $32 / $29) came from the same
LLM-assisted scaffold pass whose quantitative claims have now been wrong twice — "1.5B learns search
and self-verification" (falsified by M1) and the "<$30 on veRL" cost line (unrunnable on our loop).
`lessons.md` #1 says those need first-hand replacement. This is that, for the money.

**Method.** Every input is tagged MEASURED (from a committed artifact or a verified price page) or
ASSUMED (a judgement, with its reasoning). The output is a range, and `sensitivity()` reports which
assumption the range is actually made of — because a single number here would be false precision.

**The dominant uncertainty is `bisect`'s episode length, and `bisect` does not exist yet.** No amount
of care about the other inputs changes that, which is itself the finding: the largest line item in
the project cannot be costed to better than ~3x until Phase 1.1 produces a real episode.

Run:  uv run --extra dev python scripts/cost_model.py
"""

from __future__ import annotations

from dataclasses import dataclass

# ======================================================================================
# MEASURED — every number below traces to a committed artifact or a verified price page
# ======================================================================================

#: modal.com/pricing, verified 2026-08-01 (`tasks/spend.md`).
L4_PER_HOUR = 0.799
A100_40_PER_HOUR = 2.099

#: Phase 0.1's clean ladder: 10 arms x 200 steps, one commit, one GPU tier, 7.4 h L4, $5.88.
#: Config: Llama-3.2-1B, batch 128 (16 prompts x G=8), max_new_tokens=64, LoRA + KL reference.
#: This is the project's single best-characterised unit of training work.
LADDER_RUNS = 10
LADDER_HOURS = 7.4
LADDER_TOKENS_PER_COMPLETION = 64
LADDER_PARAMS_B = 1.0
HOURS_PER_200_STEP_RUN_1B = LADDER_HOURS / LADDER_RUNS  # 0.74 h
COST_PER_200_STEP_RUN_1B = HOURS_PER_200_STEP_RUN_1B * L4_PER_HOUR  # $0.59

#: Phase 0.2: three hosted 200-step GRPO runs at 1B on Prime's free queue, $0.
#: Not theoretical — G4 trained to eval 0.9980 there.
PRIME_FREE_TIER_RUN_COST = 0.0

#: Phase 0.3 M1/M3: inference-only screening on L4. 12,800 completions (2 models x 4 settings,
#: 512 max tok) = $1.57; 3,200 completions at 3B = $0.49.
SCREEN_COST_PER_1K_COMPLETIONS_3B = 0.49 / 3.2


@dataclass
class Assumption:
    name: str
    low: float
    mid: float
    high: float
    why: str


# ======================================================================================
# ASSUMED — judgements, each with its reasoning. These are what the range is made of.
# ======================================================================================

ASSUMPTIONS = [
    Assumption(
        "bisect_tokens_per_episode", 600, 1200, 2400,
        "THE DOMINANT UNKNOWN. `bisect` is root-cause debugging under a query budget: the agent "
        "spends budget on instrumentation and subset runs, then emits a fix. Multi-turn, with tool "
        "output in context. Countdown measured 138-346 tokens for single-turn arithmetic reasoning; "
        "a tool-using debugging episode is several times that. Nothing measures this until Phase 1.1.",
    ),
    Assumption(
        "confirmatory_params_b", 1.5, 1.7, 2.0,
        "stages.md pins 1.7B for the confirmatory arms; Prime hosted training now offers Qwen3.5-2B "
        "and the revision hash is not yet pinned (todo.md).",
    ),
    Assumption(
        "vllm_speedup_on_total_step", 1.0, 1.4, 2.0,
        "vLLM accelerates generation, not backward. Generation was 22% of step time at 64 tokens; "
        "at ~1200 it is a much larger share, so the speedup on TOTAL step time is well under the "
        "5-10x quoted for generation alone. 1.0 = we stay on HF generate (M2 says vLLM costs a rung-4 "
        "implementation first, so this is a real option).",
    ),
    Assumption(
        "exploratory_grid_free", 0.0, 0.0, 1.0,
        "stages.md scopes 2.1's 8-12 variants to Prime's free queue, proven at $0 in Phase 0.2. "
        "0.0 = free queue holds; 1.0 = it is gone and the grid runs paid. The sprint that introduced "
        "it closed ~2026-06-20 with no announced successor (CLAUDE.md §15), so this is a live risk, "
        "not a formality.",
    ),
    Assumption(
        "grader_seconds_per_episode", 0.0004, 0.05, 1.92,
        "MEASURED 2026-08-31 (S2). The model had no execution term at all, and todo.md carried a "
        "feared 2s-30s span (1,365 -> 20,480 core-hours for the grid). Both ends were wrong for a "
        "pure-function grader: honest sandboxed execution of a five-assertion suite is 0.1-0.4 ms, "
        "3-4 orders of magnitude under the old low end. What actually sets the cost is "
        "`p_nonterminating x timeout_budget` -- sx-collatz's 1.92 s mean is 0.383 x 5 s almost "
        "exactly, and that is the HIGH end here. MID allows bisect's suites to be ~100x heavier than "
        "three assertions. Both factors are knobs, not unknowns.",
    ),
    Assumption(
        "frontier_api_total", 25.0, 60.0, 140.0,
        "Battery axes A1/A2 (exploit-finding) across 8-12 variants, plus A4 judge probes and the "
        "Gallop field report over ~15 Hub environments. Haiku 4.5 bulk with caching on, Sonnet spot "
        "tier. Scales with variants x episodes x tokens; the same episode-length unknown drives it.",
    ),
]

A = {a.name: a for a in ASSUMPTIONS}


def run_cost(params_b: float, tokens: float, vllm_speedup: float, *, hourly: float,
             grader_s: float = 0.0, group_size: int = 8) -> float:
    """Cost of one 200-step GRPO run, scaled from the measured 1B/64-token anchor.

    Two scaling factors, both roughly linear and both stated rather than hidden:

    - **params**: generation is memory-bandwidth bound and backward is compute bound; both scale
      about linearly in parameter count at this size.
    - **tokens**: generation scales linearly in tokens generated, and the backward pass scales
      about linearly in sequence length. Phase 0.1 measured generation at 22% of step time with
      64-token completions, so neither term is negligible and the product is ~linear overall.

    **Known biases, both directions, neither corrected for.**

    Pushing UP: multi-turn tool execution holds the GPU idle between turns, and long sequences force
    a larger tier (an L4's 24 GB held 1B at 64 tokens with 13.5-14.5 GB peak; 1.7B at 1200 tokens
    will not fit the same way).

    Pushing DOWN, and probably the larger of the two: **the anchor is badly GPU-utilised.** 64-token
    completions on an L4 leave the device mostly idle waiting on memory, so per-token throughput at
    1200 tokens is materially better than at 64. Scaling linearly from an under-utilised anchor
    overestimates.

    Net: treat any single figure here as uncertain to ~3x. The conclusion that survives regardless is
    the *ratio* to what is in hand, not the absolute.
    """
    scale = (params_b / LADDER_PARAMS_B) * (tokens / LADDER_TOKENS_PER_COMPLETION)
    hours = HOURS_PER_200_STEP_RUN_1B * scale / vllm_speedup

    # Grading holds the GPU idle: a step generates `group_size` rollouts, then every one is graded
    # before advantages exist. Billed at the same hourly rate because the device is rented, not used.
    # `grader_s` is MEASURED (S2) rather than assumed -- see the assumption's note. It is added
    # rather than multiplied, which is why it barely moves the total at the low end and dominates a
    # cheap run at the high end.
    hours += (STEPS_PER_RUN * group_size * grader_s) / 3600.0
    return hours * hourly


@dataclass
class Line:
    phase: str
    what: str
    value: float
    note: str = ""


#: Steps per run, matching the 200-step anchor `run_cost` scales from.
STEPS_PER_RUN = 200

#: `stages.md`'s own lines for the work that remains: Walk $20 + Run $32 + Gallop $29 + R1 $2.
PLAN_REMAINING = 83.0

#: Modal balance. Updated 2026-09-01: August's unspent $15.72 did NOT expire -- it rolled into
#: September's $30. The number that matters for planning is not this balance but the ~$30/month
#: RATE behind it; see `tasks/spend.md`.
BALANCE = 45.0


def build(pick) -> list[Line]:  # type: ignore[no-untyped-def]
    """One scenario's line items. ``pick(name)`` selects low/mid/high for every assumption.

    Every assumption flows through ``pick`` — none is hardcoded into a line — so ``sensitivity``
    can vary one at a time and get a true span. An earlier draft baked two of them into per-line
    low/high values, which made those rows report a span of $0: the table said they did not matter
    when in fact it was not varying them at all.
    """
    tok = pick("bisect_tokens_per_episode")
    params = pick("confirmatory_params_b")
    speed = pick("vllm_speedup_on_total_step")
    paid_grid = pick("exploratory_grid_free")
    api = pick("frontier_api_total")
    grader_s = pick("grader_seconds_per_episode")

    # Long sequences at ~1.7B will not fit an L4 the way 1B/64-tok did. Step up with the episode
    # rather than pretending the cheap tier holds.
    hourly = L4_PER_HOUR if tok <= 600 else A100_40_PER_HOUR
    per_run = run_cost(params, tok, speed, hourly=hourly, grader_s=grader_s)
    screen = 8.0 * (tok / 600.0)  # screening scales with episode length too

    return [
        Line("0.4", "R1 — Prime Intellect 1B hacking repro (100 steps)",
             0.0, "scoped to the free queue; ~$0.30 on L4 if paid"),
        Line("0.5", "Literature gate", 0.0, "no GPU"),
        Line("1.x", "Walk — bisect screening + positive control", screen,
             "k=64 inference screens over variants, scaled from M1/M3"),
        Line("1.x", "Walk — A1/A2 exploit-finder (frontier API)", api * 0.4, "share of the API total"),
        Line("2.1", "Run — exploratory grid, 8-12 variants x 1 seed", paid_grid * 12 * per_run,
             "$0 while Prime's free queue holds; 12 paid runs if it does not"),
        Line("2.2", "Run — CONFIRMATORY, 4 variants x 3 seeds = 12 runs", 12 * per_run,
             f"12 x ${per_run:.2f}/run at {params:.1f}B, {tok:.0f} tok/episode"),
        Line("3.2", "Gallop — R3 Reasoning Gym transfer repro (1 run)", per_run, "published config is 3B"),
        Line("3.3", "Gallop — eta decomposition, evals (a)-(d) on 4 arms", screen * 0.8,
             "inference-only"),
        Line("3.4/3.5", "Gallop — A4/A5/A6 judge probes + field report (API)", api * 0.6,
             "share of the API total"),
    ]


def scenario(which: str) -> list[Line]:
    return build(lambda n: getattr(A[n], which))


def total(which: str) -> float:
    return sum(x.value for x in scenario(which))


def main() -> None:
    print("=" * 96)
    print("REMAINING SCIENCE — bottom-up, anchored on measured runs".center(96))
    print("=" * 96)
    print(f"\nMEASURED ANCHOR  one 200-step GRPO run, 1B / 64 tok / L4 = "
          f"{HOURS_PER_200_STEP_RUN_1B:.2f} h = ${COST_PER_200_STEP_RUN_1B:.2f}")
    print(f"                 (Phase 0.1 clean ladder: {LADDER_RUNS} arms, {LADDER_HOURS} h, $5.88)")
    print(f"MEASURED ANCHOR  same run on Prime's free queue = ${PRIME_FREE_TIER_RUN_COST:.2f}"
          "   (Phase 0.2, 3 runs, one trained to eval 0.9980)\n")

    lo, mid, hi = scenario("low"), scenario("mid"), scenario("high")
    print(f"{'phase':<9}{'item':<54}{'low':>9}{'mid':>9}{'high':>9}")
    print("-" * 96)
    for a, b, c in zip(lo, mid, hi, strict=True):
        print(f"{b.phase:<9}{b.what:<54}{a.value:>9.0f}{b.value:>9.0f}{c.value:>9.0f}")
    print("-" * 96)
    print(f"{'':<9}{'TOTAL REMAINING':<54}{total('low'):>9.0f}{total('mid'):>9.0f}"
          f"{total('high'):>9.0f}")

    print(f"\n  stages.md's plan for the same work   ${PLAN_REMAINING:>7.0f}")
    print(f"  Modal balance in hand                ${BALANCE:>7.0f}")
    print(f"\n  LOW  is {total('low')/PLAN_REMAINING:>4.1f}x the plan and {total('low')/BALANCE:>5.1f}x the balance")
    print(f"  MID  is {total('mid')/PLAN_REMAINING:>4.1f}x the plan and {total('mid')/BALANCE:>5.1f}x the balance")
    print(f"  HIGH is {total('high')/PLAN_REMAINING:>4.1f}x the plan and {total('high')/BALANCE:>5.1f}x the balance")
    print("\n  HIGH stacks every assumption at its worst simultaneously. That is a bound, not a")
    print("  forecast. LOW is the one that matters: even it does not fit in the balance.")

    print("\n" + "=" * 96)
    print("SENSITIVITY — what the range is actually made of".center(96))
    print("=" * 96)
    base = total("mid")
    for name in A:
        d = {}
        for which in ("low", "high"):
            pick = lambda n, _n=name, _w=which: getattr(A[n], _w if n == _n else "mid")
            d[which] = sum(x.value for x in build(pick)) - base
        print(f"  {name:<34} low {d['low']:>+8.0f}   high {d['high']:>+8.0f}   "
              f"span ${abs(d['high'] - d['low']):>6.0f}")
    print("\n  Largest span = measure that first. `bisect` does not exist yet, which is precisely")
    print("  why the biggest line item cannot be costed better than ~3x today.")


if __name__ == "__main__":
    main()
