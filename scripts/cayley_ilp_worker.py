#!/usr/bin/env python3
"""Worker script for a single Cayley ILP solve: one parameter set × one group.

Called by SLURM array jobs. Writes a JSON result file.

Usage:
    python scripts/cayley_ilp_worker.py \
        --n 48 --k 9 --t 5 --lambda 0 --mu 2 \
        --lib-id 3 --output-dir cayley_ilp_results \
        [--undirected] [--time-limit 300] [--no-aut-pruning]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from ilp.models.cayley_dsrg import load_cayley_data, build_cayley_dsrg_quad, _extract_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_param", type=int, required=True)
    parser.add_argument("--mu", type=int, required=True)
    parser.add_argument("--lib-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="cayley_ilp_results")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--no-aut-pruning", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    kind = "srg" if args.undirected else "dsrg"
    outfile = outdir / f"{kind}_{args.n}_{args.k}_{args.t}_{args.lambda_param}_{args.mu}_g{args.lib_id}.json"

    # Load group data
    t0 = time.perf_counter()
    group_data = load_cayley_data(args.n, args.lib_id)
    gap_time = time.perf_counter() - t0

    # Build model
    t0 = time.perf_counter()
    model, x_vars = build_cayley_dsrg_quad(
        args.n, args.k, args.t, args.lambda_param, args.mu, group_data,
        use_aut_pruning=not args.no_aut_pruning,
        undirected=args.undirected,
        quiet=False,
    )
    build_time = time.perf_counter() - t0

    # Configure solver
    model.setParam("MIPFocus", 1)
    if args.time_limit > 0:
        model.setParam("TimeLimit", args.time_limit)

    # Solve
    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0

    status = _extract_status(model)

    connection_set = None
    connection_set_orig = None
    if model.SolCount > 0:
        connection_set = [g for g in range(group_data.num_nonid) if round(x_vars[g].X) == 1]
        connection_set_orig = [group_data.nonid_order[g] for g in connection_set]

    result = {
        "n": args.n,
        "k": args.k,
        "t": args.t,
        "lambda": args.lambda_param,
        "mu": args.mu,
        "lib_id": args.lib_id,
        "group_name": group_data.name,
        "undirected": args.undirected,
        "use_aut_pruning": not args.no_aut_pruning,
        "status": status,
        "gap_seconds": round(gap_time, 4),
        "build_seconds": round(build_time, 4),
        "solve_seconds": round(solve_time, 4),
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        "num_gen_constrs": model.NumGenConstrs,
        "connection_set": connection_set,
        "connection_set_original_indices": connection_set_orig,
    }

    outfile.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
