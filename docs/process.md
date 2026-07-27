# Process — phase lifecycle

Inherited from the program house style (waterline / epibench / synthoracle / crit-thinking).

## Phase lifecycle

**PLAN → TEST → IMPLEMENT → VERIFY → RETRO → `/learn`.** All six. Only the current phase is planned
in detail; future phases stay headline-level (`stages.md`).

- **PLAN** — `phases/phase-X.Y-<slug>-plan.md`: purpose, config, gates (pass criteria), outputs,
  non-goals, change log. Cut the phase branch at plan-lock.
- **TEST** — code: failing tests first. Research: falsifiable hypothesis + evaluation criteria first.
  Tests and gates define "done."
- **IMPLEMENT** — minimal code, minimal blast radius. Root causes, not patches.
- **VERIFY** — never mark complete without proving it works: gates pass, numbers regenerate from a
  committed script, confound referees clean.
- **RETRO** — `phases/phase-X.Y-<slug>-retro.md`: hypothesis, what happened, **surprises
  (first-class)**, what to change. Each section anchors to its eventual paper section
  (→ Methods / Results / Discussion / Related Work / Limitations).
- **`/learn`** — appended to `../tasks/lessons.md` (process stream, paper-irrelevant). Mandatory
  `[DELETE]` section; empty deletions allowed but must be explicit.

## Phase numbering

`Stage.Phase`, where stages are **0 Crawl · 1 Walk · 2 Run · 3 Gallop**.
Phases: 0.1–0.5 · 1.1–1.6 · 2.1–2.5 · 3.1–3.6. Full map in `stages.md`.

## Gates (defaults, adapted per phase)

1. All tests pass (code) or all evaluation criteria met (research).
2. Lint + typecheck clean — `make check`.
3. Reproducibility: results regenerate from committed code + pinned parameters.
4. Retro written with paper-section anchors.
5. `/learn` written with an explicit `[DELETE]` section.

## Three-reviewer pass at critical boundaries

Spawn three **Opus** subagents in **clean-context** sessions, each given `conceptual.md` + the phase
artifacts and nothing else:

- **Method-rigor** — does the method work, conditional on the framing?
- **Framing-stress** — does the framing survive contact with this phase's evidence?
- **Prior-art-coverage** — would a domain reviewer raise an uncited reference class?

**Confidence filter:** each issue scored 0–100; the aggregated report includes only issues at **≥80**.
Lower-confidence flags go to an appendix for author judgment.

**Critical boundaries for this project:** Crawl→Walk (the R1 reachability gate) · Walk→Run (the
screen + kill-switch) · Run→Gallop (the H1 result).

## Branches

Phase branches off `main` (`phase-X.Y-<2-3-word-desc>`). Stage-0 scaffold work stays on `main`.
User merges manually. No force pushes. Small, focused commits. **No AI attribution in any git
artifact.**

**Prerequisite enforcement:** no commit on a phase branch without a written plan it references; no
merge to `main` without retro + `/learn` (+ three-reviewer pass at critical boundaries).

## Self-improvement loop

After ANY correction or surprise, append a dated bullet to `../tasks/lessons.md`. Review at session
start.

## Model tiering for subagents

**Opus** for reviewers · **Sonnet** for task agents and architects · **Haiku** for test runners and
mechanical checks.
