"""The report card — ``assay run <env>`` output.

Scaffolding (``CLAUDE.md`` §7): Claude may implement this end to end. It is the demo surface, not
the science.

Design rule: the card shows the **exploit transcripts it actually found**, not just scores. A
red/amber/green grid nobody can audit is a worse artifact than three verbatim transcripts of the
model gaming the grader.

Every card states that ``assay_score``'s weighting is **fitted**, and on what
(``docs/desiderata.md`` §10).
"""

from __future__ import annotations

from pathlib import Path

from assay.battery import AssayReport


def render_html(report: AssayReport) -> str:
    """Render a self-contained HTML report card. Implement at Phase 3.6."""
    raise NotImplementedError("Phase 3.6 — CLI polish")


def write_report(report: AssayReport, out: Path) -> None:
    """Write the card to disk. Output is a derived artifact — ignored by git, regenerated."""
    raise NotImplementedError("Phase 3.6 — CLI polish")
