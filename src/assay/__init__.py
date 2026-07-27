"""assay — a zero-GPU-hour diagnostic battery for RL environments.

Does an environment teach the skill, or teach the exploit?

Module map (see ``CLAUDE.md`` §3):

- ``config``   — pinned experimental configuration; the grader-variant factorial.
- ``env``      — ``bisect``: root-cause debugging under a query budget.
- ``grader``   — the proxy/true split: training grader (variant-configured) + held-out grader.
- ``screen``   — ``p_hack@64`` base-rate screen and the admission band.
- ``battery``  — the six inference-only probes A1..A6 and ``assay_score``.
- ``loop``     — GRPO wiring and per-step gap logging.
- ``measure``  — gap slope and the transfer-efficiency (eta) decomposition.
- ``report``   — the HTML report card.
- ``cli``      — ``assay run <env>``.

Ownership (``CLAUDE.md`` §7 — learning-first): the user writes ``loop`` (Phase 0.1),
``grader``'s pathology design (Phase 1.2), and ``battery``'s scoring logic (Phase 1.3).
Claude scaffolds ``config``, ``env`` plumbing, ``report``, ``cli`` and the Modal wrapper.
"""

__version__ = "0.1.0"
