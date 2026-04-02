#!/usr/bin/env python3
"""Aggregate Cayley ILP result JSON files into a summary CSV.

Usage:
    python scripts/cayley_ilp_aggregate.py cayley_ilp_results/
    python scripts/cayley_ilp_aggregate.py cayley_ilp_results/ --output summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="Directory containing result JSON files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV path (default: <result_dir>/summary.csv)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    json_files = sorted(result_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {result_dir}")
        sys.exit(1)

    results = []
    for f in json_files:
        try:
            results.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)

    print(f"Loaded {len(results)} results from {result_dir}")

    # Summary stats
    found = [r for r in results if r["status"] == "Optimal" and r.get("connection_set")]
    infeasible = [r for r in results if r["status"] == "Infeasible"]
    timeout = [r for r in results if r["status"] == "TimeLimit"]
    other = len(results) - len(found) - len(infeasible) - len(timeout)

    print(f"  {len(found)} found, {len(infeasible)} infeasible, "
          f"{len(timeout)} timeout, {other} other")

    # Group by parameter set
    by_params = {}
    for r in results:
        key = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
        by_params.setdefault(key, []).append(r)

    print(f"\n{'n':>4} {'k':>3} {'t':>3} {'λ':>3} {'μ':>3}  "
          f"{'groups':>6} {'found':>5} {'infeas':>6} {'timeout':>7} {'total_s':>8}")
    print("-" * 60)

    summary_rows = []
    for key in sorted(by_params):
        n, k, t, lam, mu = key
        group_results = by_params[key]
        n_found = sum(1 for r in group_results if r["status"] == "Optimal" and r.get("connection_set"))
        n_infeas = sum(1 for r in group_results if r["status"] == "Infeasible")
        n_timeout = sum(1 for r in group_results if r["status"] == "TimeLimit")
        total_time = sum(r.get("solve_seconds", 0) for r in group_results)

        print(f"{n:>4} {k:>3} {t:>3} {lam:>3} {mu:>3}  "
              f"{len(group_results):>6} {n_found:>5} {n_infeas:>6} {n_timeout:>7} {total_time:>8.1f}")

        # Detail for found solutions
        found_groups = [r for r in group_results if r["status"] == "Optimal" and r.get("connection_set")]
        for r in found_groups:
            orig = r.get("connection_set_original_indices", [])
            summary_rows.append({
                "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                "lib_id": r["lib_id"], "group_name": r["group_name"],
                "status": "FOUND",
                "solve_seconds": r.get("solve_seconds", 0),
                "connection_set": " ".join(str(x) for x in orig),
            })

        # Summary row for parameter set
        summary_rows.append({
            "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
            "lib_id": "", "group_name": "ALL",
            "status": f"{n_found}found/{n_infeas}infeas/{n_timeout}timeout",
            "solve_seconds": total_time,
            "connection_set": "",
        })

    # Write CSV
    outpath = Path(args.output) if args.output else result_dir / "summary.csv"
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "k", "t", "lambda", "mu",
            "lib_id", "group_name", "status", "solve_seconds", "connection_set",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to {outpath}")

    # Print found solutions
    if found:
        kind = "SRG" if results[0].get("undirected") else "DSRG"
        print(f"\n=== Found {kind} Cayley graphs ===")
        for r in found:
            orig = r.get("connection_set_original_indices", [])
            print(f"  {kind}({r['n']},{r['k']},{r['lambda']},{r['mu']}) "
                  f"on Group {r['lib_id']} ({r['group_name']}): "
                  f"S = {{{', '.join(str(x) for x in orig)}}}")


if __name__ == "__main__":
    main()
