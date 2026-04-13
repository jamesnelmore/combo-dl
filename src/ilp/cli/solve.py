"""``python -m ilp solve`` — solve a single SRG or DSRG instance.

Examples::

    python -m ilp solve srg 10 3 0 1
    python -m ilp solve srg 10 3 0 1 --fix-neighbors --lex exponential
    python -m ilp solve srg 10 3 0 1 --formulation quadratic
    python -m ilp solve srg 10 3 0 1 --formulation relaxed --time-limit 60
    python -m ilp solve dsrg 6 2 1 0 1 --formulation relaxed
    python -m ilp solve dsrg 8 3 2 1 1 --threads 4
"""

from __future__ import annotations

import argparse
import sys


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``solve`` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "solve",
        help="Solve a single SRG or DSRG instance.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Graph type (srg | dsrg) ───────────────────────────────────────────
    p.add_argument(
        "graph_type",
        choices=["srg", "dsrg"],
        help="Graph family: 'srg' (undirected) or 'dsrg' (directed).",
    )

    # ── Positional parameters ─────────────────────────────────────────────
    # SRG:  n k lambda mu
    # DSRG: n k t lambda mu
    p.add_argument(
        "params",
        nargs="+",
        type=int,
        help=(
            "Graph parameters as positional integers. "
            "SRG: n k lambda mu  |  DSRG: n k t lambda mu"
        ),
    )

    # ── Model options ─────────────────────────────────────────────────────
    p.add_argument(
        "--formulation",
        choices=["exact", "relaxed", "quadratic"],
        default="exact",
        help=(
            "ILP formulation to use. 'exact': all constraints hard. "
            "'relaxed': minimise violation count (big-M). "
            "'quadratic': minimise sum of squared residuals (MIQCQP, SRG only). "
            "Default: exact."
        ),
    )
    p.add_argument(
        "--fix-neighbors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pin neighbours of vertices 0 and 1 to high-index positions as a "
            "symmetry break (default: enabled).  Compatible with --lex."
        ),
    )
    p.add_argument(
        "--lex",
        choices=["none", "exponential", "lex_leader", "hybrid"],
        default="none",
        help=(
            "Lex-ordering strategy for symmetry breaking (undirected only). "
            "Default: none.  Can be combined with --fix-neighbors."
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
        "--time-limit",
        type=float,
        default=None,
        help="Wall-clock time limit in seconds.",
    )
    p.add_argument(
        "--heuristics",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Fraction of solve time spent on MIP heuristics (0.0–1.0). "
            "Gurobi default is 0.05. Useful for quadratic/relaxed "
            "formulations where the LP bound is weak."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress Gurobi console output.",
    )
    p.add_argument(
        "--lex-block-size",
        type=int,
        default=20,
        metavar="B",
        help=(
            "Block size for hybrid lex ordering (default: 20). "
            "Smaller values improve numerical stability at the cost of "
            "more auxiliary variables."
        ),
    )
    p.add_argument(
        "--gurobi-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set an arbitrary Gurobi parameter (repeatable). "
            "Example: --gurobi-param BarHomogeneous=1"
        ),
    )

    p.set_defaults(func=_run)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _parse_params(
    graph_type: str, raw: list[int],
) -> dict[str, int]:
    """Validate and unpack positional parameter list into a named dict."""
    if graph_type == "srg":
        if len(raw) != 4:
            print(
                f"Error: SRG requires exactly 4 parameters (n k lambda mu), "
                f"got {len(raw)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        n, k, lam, mu = raw
        return {"n": n, "k": k, "lambda": lam, "mu": mu}

    # dsrg
    if len(raw) != 5:
        print(
            f"Error: DSRG requires exactly 5 parameters (n k t lambda mu), "
            f"got {len(raw)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    n, k, t, lam, mu = raw
    return {"n": n, "k": k, "t": t, "lambda": lam, "mu": mu}


def _run(args: argparse.Namespace) -> None:
    """Entry point called by the CLI dispatcher."""
    params = _parse_params(args.graph_type, args.params)
    graph_type = args.graph_type
    formulation = args.formulation

    # Warn if lex ordering is requested for directed graphs (no-op).
    if graph_type == "dsrg" and args.lex != "none":
        print(
            "Warning: lex ordering is only supported for undirected (SRG) "
            "graphs. Ignoring --lex flag.",
            file=sys.stderr,
        )

    # Quadratic formulation is SRG-only.
    if graph_type == "dsrg" and formulation == "quadratic":
        print(
            "Error: --formulation quadratic is only available for SRG.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse --gurobi-param Key=Value pairs.
    gurobi_params: dict[str, str] = {}
    for kv in args.gurobi_param:
        if "=" not in kv:
            print(f"Error: --gurobi-param expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            sys.exit(1)
        key, val = kv.split("=", 1)
        gurobi_params[key] = val

    # ── Dispatch to the appropriate solver ────────────────────────────────
    if graph_type == "srg":
        from ilp.models.srg import solve_srg

        result = solve_srg(
            n=params["n"],
            k=params["k"],
            lambda_param=params["lambda"],
            mu=params["mu"],
            formulation=formulation,
            fix_neighbors=args.fix_neighbors,
            lex_order=args.lex,
            lex_block_size=args.lex_block_size,
            threads=args.threads,
            time_limit=args.time_limit,
            heuristics=args.heuristics,
            quiet=args.quiet,
            gurobi_params=gurobi_params,
        )
    else:
        from ilp.models.dsrg import solve_dsrg

        result = solve_dsrg(
            n=params["n"],
            k=params["k"],
            t=params["t"],
            lambda_param=params["lambda"],
            mu=params["mu"],
            relaxed=(formulation == "relaxed"),
            fix_neighbors=args.fix_neighbors,
            threads=args.threads,
            time_limit=args.time_limit,
            heuristics=args.heuristics,
            quiet=args.quiet,
            gurobi_params=gurobi_params,
        )

    # ── Print results ─────────────────────────────────────────────────────
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{graph_type.upper()} ({param_str})  [{formulation}]")
    print(f"  Status:    {result['status']}")
    print(f"  Wall time: {result['wall_seconds']:.3f}s")

    if result.get("obj_val") is not None:
        print(f"  Objective: {result['obj_val']}")

    adj = result.get("adjacency")
    if adj is not None:
        print("\n  Adjacency matrix:")
        for row in adj:
            print("  " + " ".join(str(x) for x in row))
    elif result["status"] == "Optimal":
        print("\n  (feasibility problem — no adjacency to display)")
