"""Render a calibration sweep, re-deriving the selection from the committed results file.

    uv run python -m assay.crawl.report experiments/phase-0.1-grpo-by-hand/results/<file>.json

The rule is **not** read from the JSON's stored ``selection`` field — it is re-applied from
``calibrate.select`` to the stored summaries. So a change to the rule takes effect on every existing
sweep without a GPU re-run, and there is exactly one definition of the rule in the repo.

Standard errors are printed beside every ``dead_group_fraction`` because the selection turns on
differences of a few points and the estimator's own SE is comparable to them. A ranking reported
without its SE invites reading sampling noise as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from assay.crawl.calibrate import Selection, select, standard_error, summaries_from_records


def render(result: dict[str, Any]) -> str:
    """Format one sweep's summaries, re-derived selection, and provenance."""
    summaries = summaries_from_records(result["summaries"])
    selection: Selection = select(summaries)
    provenance = result.get("provenance", {})

    header = (
        f"{'setting':<28} {'dead':>7} {'±SE':>7} {'pass@1':>8} {'pass@k':>8} "
        f"{'headroom':>9} {'parse':>7} {'tokens':>7}"
    )
    lines = [header, "-" * 92]
    picked = (
        (selection.chosen.family, selection.chosen.setting) if selection.chosen else (None, None)
    )
    for s in sorted(summaries, key=lambda x: x.dead_group_fraction):
        se = standard_error(s.dead_group_fraction, s.n_prompts)
        marker = "*" if (s.family, s.setting) == picked else " "
        lines.append(
            f"{s.family + '/' + s.setting:<28}{marker} {s.dead_group_fraction:6.3f} "
            f"{se:7.3f} {s.pass_at_1:8.3f} {s.pass_at_k:8.3f} {s.headroom:9.3f} "
            f"{s.parse_fail_rate:7.3f} {s.median_completion_tokens:7.0f}"
        )

    lines.append("")
    if selection.chosen is None:
        lines.append("CHOSEN: none — every setting was excluded")
    else:
        chosen = selection.chosen
        se = standard_error(chosen.dead_group_fraction, chosen.n_prompts)
        lines.append(
            f"CHOSEN: {chosen.family}/{chosen.setting}  "
            f"(dead {chosen.dead_group_fraction:.3f} ± {se:.3f}, "
            f"pass@1 {chosen.pass_at_1:.3f}, headroom {chosen.headroom:.3f})"
        )
    for exclusion in selection.excluded:
        lines.append(f"  excluded {exclusion.family}/{exclusion.setting}: {exclusion.reason}")

    lines += ["", f"rule: {selection.rule}", ""]
    for key in ("model_id", "model_revision", "prompt_template_sha256", "chat_template_sha256"):
        if key in provenance:
            lines.append(f"  {key}: {provenance[key]}")
    for key in ("n_prompts", "k", "git_sha", "git_dirty"):
        if key in provenance:
            lines.append(f"  {key}: {provenance[key]}")
    grader = provenance.get("grader")
    if isinstance(grader, dict):
        lines.append(f"  r_binary_extractor: {grader.get('r_binary_extractor')}")
    if provenance.get("git_dirty"):
        lines.append("  WARNING: dirty tree — git_sha does not identify the code that ran")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    result = json.loads(Path(args[0]).read_text())
    print(render(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
