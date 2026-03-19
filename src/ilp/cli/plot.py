"""``python -m ilp plot`` — visualise sweep results.

Reads one or more sweep result JSON files and produces a wall-time-vs-n
scatter plot with one series per ``(model, config)`` combination.

Examples::

    python -m ilp plot results.json
    python -m ilp plot results.json --timeout 300
    python -m ilp plot exact.json relaxed.json --timeout 600 --log-y
    python -m ilp plot results.json -o figure.png --title "SRG comparison"

Multiple result files are overlaid on the same axes so you can visually
compare formulations, symmetry-breaking strategies, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``plot`` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "plot",
        help="Visualise sweep results (wall time vs n).",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="One or more sweep result JSON files to plot.",
    )

    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Draw a horizontal timeout line at this value.  TimeLimit "
            "instances are rendered as triangles pinned to this line."
        ),
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Save figure to this file (e.g. figure.png, figure.pdf) "
            "instead of displaying interactively."
        ),
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        default=False,
        help="Use a logarithmic scale for the y-axis.",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[9.0, 6.0],
        metavar=("W", "H"),
        help="Figure size in inches (default: 9 6).",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Print a text summary table instead of (or in addition to) plotting.",
    )

    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Entry point called by the CLI dispatcher."""
    # Lazy import so that ``--help`` doesn't need matplotlib / numpy.
    from ilp.bench.viz import plot_walltime_vs_n, print_summary

    paths = [str(p) for p in args.result_files]

    if args.summary:
        print_summary(*paths)
        print()

    plot_walltime_vs_n(
        *paths,
        timeout=args.timeout,
        title=args.title,
        output=args.output,
        log_y=args.log_y,
        figsize=tuple(args.figsize),
    )
