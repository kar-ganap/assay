"""``assay`` command line.

    assay run <env>       run the battery over an environment, emit a report card
    assay screen <env>    run only the base-rate screen (p_hack@k)

Gallop's exit gate is that someone else can ``pip install assay``, point it at their environment, and
get a report card. This module is that promise.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assay", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the diagnostic battery over an environment")
    run.add_argument("env", help="environment id or path")
    run.add_argument("--out", default="reports", help="report output directory")
    run.add_argument("--axes", default="A1,A2,A3", help="comma-separated axes to run")

    screen = sub.add_parser("screen", help="run only the base-rate screen")
    screen.add_argument("env", help="environment id or path")
    screen.add_argument("-k", type=int, default=64, help="rollouts per variant")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    raise NotImplementedError(f"Phase 3.6 — `assay {args.command}` not implemented yet")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
