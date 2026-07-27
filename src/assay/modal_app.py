"""Modal wrapper for the training runs.

Backend policy (``CLAUDE.md`` §6): the exploratory grid targets the Prime Sprints free queue
(Llama-3.2-1B) where available; **confirmatory arms run here** (H100 $3.95/h, A100-80GB ~$3.20/h).
Tinker is the fallback that removes infra work.

House pattern (from ``../waterline``): ship the ``assay`` source into the image via
``add_local_python_source``; only third-party deps go through ``pip_install``.

Excluded from mypy until the ``modal`` extra is wired (see ``pyproject.toml``).
"""

from __future__ import annotations

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency
    modal = None  # type: ignore[assignment]


APP_NAME = "assay"
GPU = "H100"
TIMEOUT_S = 60 * 90


def build_app():  # type: ignore[no-untyped-def]
    """Construct the Modal app. Implement when the first confirmatory run is scheduled (Phase 2.2)."""
    raise NotImplementedError("Phase 2.2 — Modal wiring")
