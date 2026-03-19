"""Entry point for ``python -m ilp``.

Dispatches to subcommands:

* ``solve`` — solve a single SRG or DSRG instance
* ``sweep`` — batch-benchmark an ILP formulation on a CSV of parameter sets
* ``plot``  — visualise sweep results (wall time vs n)

Examples::

    python -m ilp solve srg 10 3 0 1 --fix-neighbors --lex exponential
    python -m ilp solve dsrg 6 2 1 0 1 --relaxed
    python -m ilp sweep --model srg_exact --params params.csv --timeout 300
    python -m ilp plot results.json --timeout 300
"""

from __future__ import annotations

import argparse
import sys

from ilp.cli import solve, sweep, plot


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m ilp",
        description=(
            "ILP tools for strongly regular graph (SRG) and directed "
            "strongly regular graph (DSRG) construction problems."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Available subcommands (use '<command> --help' for details).",
    )

    # Register each subcommand.
    solve.add_parser(subparsers)
    sweep.add_parser(subparsers)
    plot.add_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Each subcommand sets a ``func`` default via ``set_defaults``.
    args.func(args)


if __name__ == "__main__":
    main()
