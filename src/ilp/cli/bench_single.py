"""``python -m ilp bench-single`` — run one benchmark instance by CSV index.

Designed for SLURM array jobs: each array task runs one parameter set from
a CSV file, identified by ``--index``.  Outputs a structured directory with
``result.json``, ``gurobi.log``, and (if a solution is found) ``adjacency.npy``.

Examples::

    python -m ilp bench-single \\
        --params srg_params_n50.csv --index 0 \\
        --model srg_exact --fix-neighbors --fix-v1 \\
        --threads 64 --timeout 14100 --heuristics 0.3 --seed 0 \\
        --output-dir bench_output/12345

    # In a SLURM array script:
    python -m ilp bench-single \\
        --params "$PARAMS_CSV" --index "$SLURM_ARRAY_TASK_ID" \\
        --model srg_exact --fix-neighbors --fix-v1 \\
        --threads "$SLURM_CPUS_PER_TASK" --timeout 14100 \\
        --output-dir "bench_output/$SLURM_ARRAY_JOB_ID"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ilp.models import list_models


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``bench-single`` subcommand on *subparsers*."""
    p = subparsers.add_parser(
        "bench-single",
        help="Run one benchmark instance by CSV row index (for SLURM array jobs).",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required arguments ────────────────────────────────────────────────
    p.add_argument(
        "--params",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path to a CSV file of graph parameters.",
    )
    p.add_argument(
        "--index",
        required=True,
        type=int,
        help="0-based row index into the params CSV.",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list_models(),
        help="Registered model name (e.g. 'srg_exact').",
    )

    # ── Model config flags ────────────────────────────────────────────────
    p.add_argument(
        "--fix-neighbors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin neighbours of vertex 0 to {1,...,k} (default: enabled).",
    )
    p.add_argument(
        "--fix-v1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also pin neighbours of vertex 1 (requires --fix-neighbors).",
    )
    p.add_argument(
        "--lex",
        choices=["none", "exponential", "lex_leader", "hybrid"],
        default="none",
        help="Lex-ordering strategy (SRG models only).  Default: none.",
    )

    p.add_argument(
        "--lex-block-size",
        type=int,
        default=20,
        metavar="B",
        help="Block size for hybrid lex ordering (default: 20).",
    )

    # ── Solver options ────────────────────────────────────────────────────
    p.add_argument(
        "--gurobi-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set an arbitrary Gurobi parameter (repeatable).",
    )
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
        help="Wall-clock time limit in seconds.",
    )
    p.add_argument(
        "--heuristics",
        type=float,
        default=None,
        metavar="FRAC",
        help="Fraction of solve time spent on MIP heuristics (0.0–1.0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Gurobi random seed for reproducibility.",
    )

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench_output"),
        metavar="DIR",
        help="Base output directory.  Instance output goes to DIR/{index:03d}/.",
    )

    p.set_defaults(func=_run)


# ---------------------------------------------------------------------------
# CSV loading (reuse the pattern from sweep.py)
# ---------------------------------------------------------------------------

_SRG_COLUMNS = {"n", "k", "lambda", "mu"}
_DSRG_COLUMNS = {"n", "k", "t", "lambda", "mu"}


def _load_row(path: Path, index: int, model_name: str) -> dict[str, int]:
    """Load a single row from a params CSV by 0-based index."""
    import csv

    is_directed = model_name.startswith("dsrg")
    required = _DSRG_COLUMNS if is_directed else _SRG_COLUMNS

    if not path.exists():
        print(f"Error: params CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

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

        for i, raw_row in enumerate(reader):
            if i == index:
                try:
                    return {col: int(raw_row[col]) for col in required}
                except (ValueError, KeyError) as exc:
                    print(
                        f"Error: bad value on row {index} of {path}: {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    print(
        f"Error: index {index} out of range for {path} "
        f"(file has fewer rows).",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> None:
    """Entry point called by the CLI dispatcher."""
    from ilp.bench.runner import run_instance

    model_name: str = args.model
    params = _load_row(args.params, args.index, model_name)

    # Build config dict.
    config: dict[str, Any] = {
        "fix_neighbors": args.fix_neighbors,
        "fix_v1": args.fix_v1,
    }
    if model_name.startswith("srg"):
        config["lex_order"] = args.lex
        config["lex_block_size"] = args.lex_block_size

    # Lex ordering and fix-neighbors are mutually exclusive symmetry breaks.
    if args.lex != "none" and args.fix_neighbors:
        print(
            "Error: --lex and --fix-neighbors are mutually exclusive. "
            "Use --no-fix-neighbors with --lex.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Output directory for this instance.
    instance_dir = args.output_dir / f"{args.index:03d}"
    instance_dir.mkdir(parents=True, exist_ok=True)

    log_file = instance_dir / "gurobi.log"

    label = "(" + ",".join(str(params[c]) for c in ["n", "k", "t", "lambda", "mu"] if c in params) + ")"
    print(f"bench-single: {model_name} {label} (index {args.index})")
    print(f"  config: {config}")
    print(f"  output: {instance_dir}")
    if args.timeout is not None:
        print(f"  timeout: {args.timeout}s")
    print(flush=True)

    # Parse --gurobi-param Key=Value pairs.
    gurobi_params: dict[str, str] = {}
    for kv in args.gurobi_param:
        if "=" not in kv:
            print(f"Error: --gurobi-param expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            sys.exit(1)
        gkey, gval = kv.split("=", 1)
        gurobi_params[gkey] = gval

    result, grb_model = run_instance(
        model_name,
        params,
        config,
        threads=args.threads,
        time_limit=args.timeout,
        heuristics=args.heuristics,
        log_file=log_file,
        seed=args.seed,
        return_model=True,
        gurobi_params=gurobi_params,
    )

    # Add bench metadata.
    result["index"] = args.index
    result["params_csv"] = str(args.params)

    # Save result JSON.
    result_path = instance_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    # Save adjacency matrix if a solution was found.
    if result["status"] in ("Optimal", "Suboptimal", "SolutionLimit"):
        try:
            _save_adjacency(grb_model, params, instance_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not save adjacency: {exc}")

    print(f"\n  Status: {result['status']}")
    print(f"  Wall time: {result['wall_seconds']:.2f}s")
    print(f"  Nodes: {result.get('node_count', '?')}")
    print(f"  Result saved to {result_path}")


def _save_adjacency(
    grb_model: Any,
    params: dict[str, int],
    instance_dir: Path,
) -> None:
    """Extract adjacency matrix from solved Gurobi model and save as .npy."""
    import numpy as np

    n = params["n"]
    adj = np.zeros((n, n), dtype=np.int8)

    for var in grb_model.getVars():
        name = var.VarName
        if not name.startswith("e_"):
            continue
        parts = name.split("_")
        if len(parts) != 3:
            continue
        i, j = int(parts[1]), int(parts[2])
        val = int(round(var.X))
        if val:
            adj[i, j] = 1
            adj[j, i] = 1  # undirected: symmetric

    np.save(instance_dir / "adjacency.npy", adj)
    print(f"  Adjacency saved to {instance_dir / 'adjacency.npy'}")
