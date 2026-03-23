"""``python -m ilp sweep`` — batch-benchmark ILP formulations.

Reads a CSV of graph parameters, pairs each row with a model + config,
and runs them sequentially via :mod:`ilp.bench.runner`.  Results are
saved incrementally to a JSON file (resume-friendly).

Examples::

    python -m ilp sweep --model srg_exact --params srg_params.csv -o results.json
    python -m ilp sweep --model dsrg_exact --params dsrg_params.csv --timeout 300
    python -m ilp sweep --model srg_exact --params srg_params.csv \\
        --fix-neighbors --lex exponential --timeout 600 --threads 8
    python -m ilp sweep --model srg_relaxed --params srg_params.csv \\
        --no-fix-neighbors -o relaxed_results.json

The params CSV must have columns matching the model's parameter names:

* **SRG models** (``srg_exact``, ``srg_relaxed``): ``n, k, lambda, mu``
* **DSRG models** (``dsrg_exact``, ``dsrg_relaxed``): ``n, k, t, lambda, mu``

Hydra compatibility
-------------------
Every CLI flag maps 1-to-1 to a flat config key, so wrapping with
``@hydra.main`` later requires only replacing ``argparse`` with a
dataclass config — the runner call stays identical.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from ilp.models import list_models


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``sweep`` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "sweep",
        help="Batch-benchmark an ILP formulation on a CSV of parameter sets.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required arguments ────────────────────────────────────────────────
    p.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Registered model name (e.g. 'srg_exact', 'dsrg_relaxed').",
    )
    p.add_argument(
        "--params",
        required=True,
        type=Path,
        metavar="CSV",
        help=(
            "Path to a CSV file of graph parameters.  Required columns "
            "depend on the model family (SRG: n,k,lambda,mu; "
            "DSRG: n,k,t,lambda,mu)."
        ),
    )

    # ── Model config flags ────────────────────────────────────────────────
    p.add_argument(
        "--fix-neighbors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pin (out-)neighbours of vertex 0 to {1,...,k} as a symmetry "
            "break (default: enabled).  Use --no-fix-neighbors to disable."
        ),
    )
    p.add_argument(
        "--fix-v1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also pin neighbours of vertex 1 (requires --fix-neighbors). "
            "Fixes λ common neighbours among N(v0) and the remaining "
            "k−1−λ from non-neighbours of v0.  Default: disabled."
        ),
    )
    p.add_argument(
        "--lex",
        choices=["none", "exponential", "lex_leader"],
        default="none",
        help=(
            "Lex-ordering strategy for symmetry breaking (SRG models only). "
            "Default: none."
        ),
    )

    # ── Solver options ────────────────────────────────────────────────────
    p.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Gurobi thread count (-1 = solver default / all cores).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Per-instance wall-clock time limit in seconds.",
    )
    p.add_argument(
        "--heuristics",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Fraction of solve time spent on MIP heuristics (0.0–1.0). "
            "Gurobi default is 0.05."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sweep_results.json"),
        metavar="JSON",
        help="Output JSON file for results (default: sweep_results.json).",
    )

    p.set_defaults(func=_run)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

# Column sets that we expect depending on graph family.
_SRG_COLUMNS = {"n", "k", "lambda", "mu"}
_DSRG_COLUMNS = {"n", "k", "t", "lambda", "mu"}


def _load_params_csv(
    path: Path,
    model_name: str,
) -> list[dict[str, int]]:
    """Load graph parameters from a CSV file.

    Validates that the required columns for the model family are present.
    All values are cast to ``int``.

    Returns:
        List of parameter dicts, one per CSV row.
    """
    if not path.exists():
        print(f"Error: params CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

    is_directed = model_name.startswith("dsrg")
    required = _DSRG_COLUMNS if is_directed else _SRG_COLUMNS

    rows: list[dict[str, int]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"Error: empty or malformed CSV: {path}", file=sys.stderr)
            sys.exit(1)

        available = set(reader.fieldnames)
        missing = required - available
        if missing:
            print(
                f"Error: CSV is missing required columns: {sorted(missing)}. "
                f"Available: {sorted(available)}",
                file=sys.stderr,
            )
            sys.exit(1)

        for line_no, raw_row in enumerate(reader, start=2):
            try:
                row = {col: int(raw_row[col]) for col in required}
            except (ValueError, KeyError) as exc:
                print(
                    f"Error: bad value on line {line_no} of {path}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            rows.append(row)

    if not rows:
        print(f"Warning: no data rows in {path}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> None:
    """Entry point called by the CLI dispatcher."""
    # Lazy import so the CLI --help is fast (no Gurobi import).
    from ilp.bench.runner import run_sweep

    model_name: str = args.model
    param_rows = _load_params_csv(args.params, model_name)

    # Build the config dict that will be passed to every instance.
    config: dict[str, Any] = {
        "fix_neighbors": args.fix_neighbors,
        "fix_v1": args.fix_v1,
    }
    # Only include lex_order for SRG models (DSRG ignores it).
    if model_name.startswith("srg"):
        config["lex_order"] = args.lex
    elif args.lex != "none":
        print(
            "Warning: --lex is only meaningful for SRG models. Ignoring.",
            file=sys.stderr,
        )

    # Lex ordering and fix-neighbors are mutually exclusive symmetry breaks.
    if args.lex != "none" and args.fix_neighbors:
        print(
            "Error: --lex and --fix-neighbors are mutually exclusive. "
            "Use --no-fix-neighbors with --lex.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Assemble the instance list: (model_name, params, config) tuples.
    instances = [(model_name, params, config) for params in param_rows]

    print(
        f"Sweep: {len(instances)} instance(s) with model={model_name}, "
        f"config={config}"
    )
    print(f"Output: {args.output}")
    if args.timeout is not None:
        print(f"Timeout: {args.timeout}s per instance")
    print()

    run_sweep(
        instances,
        threads=args.threads,
        time_limit=args.timeout,
        heuristics=args.heuristics,
        output_path=args.output,
        verbose=True,
    )
