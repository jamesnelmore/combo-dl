#!/usr/bin/env python3
"""Aggregate Cayley ILP result JSON files into a summary CSV.

Usage:
    python scripts/cayley_ilp_aggregate.py cayley_ilp_results/
    python scripts/cayley_ilp_aggregate.py cayley_ilp_results/ --output summary.csv
    watch -n 60 python scripts/cayley_ilp_aggregate.py cayley_ilp_results/
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="Directory containing result JSON files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write summary CSV to this path (default: no CSV)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full connection sets for all found graphs")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    json_files = sorted(result_dir.glob("*.json"))

    print(f"[{datetime.now().strftime('%H:%M:%S')}]  {result_dir}  —  ", end="")

    if not json_files:
        print("no results yet")
        return

    results = []
    for f in json_files:
        try:
            results.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)

    found    = [r for r in results if r["status"] == "Optimal" and r.get("connection_set")]
    infeas   = [r for r in results if r["status"] == "Infeasible"]
    timeout  = [r for r in results if r["status"] == "TimeLimit"]
    other    = len(results) - len(found) - len(infeas) - len(timeout)

    print(f"{len(results)} results  |  "
          f"found {len(found)}  infeas {len(infeas)}  "
          f"timeout {len(timeout)}  other {other}")

    unknown_params = {
        (42,12,9,4,3),(42,15,9,4,6),(42,18,14,7,8),(42,19,11,8,9),
        (44,12,8,1,4),(44,17,13,4,8),(44,20,16,10,8),
        (45,13,6,3,4),(45,16,12,3,7),(45,18,8,6,8),(45,18,16,7,7),(45,19,16,5,10),(45,20,12,11,7),
        (46,20,15,8,9),(46,21,12,9,10),
        (48,9,5,0,2),(48,19,14,5,9),(48,19,13,8,7),(48,20,10,6,10),(48,20,17,7,9),
        (48,21,18,9,9),(48,22,11,9,11),(48,22,17,11,9),
        (49,13,8,7,2),(49,19,16,9,6),(49,20,12,11,6),(49,22,18,7,12),
    }

    # Group by parameter set
    by_params = {}
    for r in results:
        key = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
        by_params.setdefault(key, []).append(r)

    print(f"\n{'n':>4} {'k':>3} {'t':>3} {'λ':>3} {'μ':>3}  "
          f"{'groups':>6} {'found':>5} {'infeas':>6} {'timeout':>7} {'total_s':>8}  {'':5}")
    print("-" * 67)

    summary_rows = []
    for key in sorted(by_params):
        n, k, t, lam, mu = key
        group_results = by_params[key]
        n_found   = sum(1 for r in group_results if r["status"] == "Optimal" and r.get("connection_set"))
        n_infeas  = sum(1 for r in group_results if r["status"] == "Infeasible")
        n_timeout = sum(1 for r in group_results if r["status"] == "TimeLimit")
        total_time = sum(r.get("solve_seconds", 0) for r in group_results)

        is_unknown = (n, k, t, lam, mu) in unknown_params
        new_tag = " *** NEW" if (is_unknown and n_found > 0) else ""
        print(f"{n:>4} {k:>3} {t:>3} {lam:>3} {mu:>3}  "
              f"{len(group_results):>6} {n_found:>5} {n_infeas:>6} {n_timeout:>7} {total_time:>8.1f}{new_tag}")

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

        summary_rows.append({
            "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
            "lib_id": "", "group_name": "ALL",
            "status": f"{n_found}found/{n_infeas}infeas/{n_timeout}timeout",
            "solve_seconds": total_time,
            "connection_set": "",
        })

    # Write CSV only when explicitly requested
    if args.output:
        outpath = Path(args.output)
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
        new_discoveries  = [r for r in found if (r["n"],r["k"],r["t"],r["lambda"],r["mu"]) in unknown_params]
        known_confirmed  = [r for r in found if (r["n"],r["k"],r["t"],r["lambda"],r["mu"]) not in unknown_params]

        if new_discoveries:
            print(f"\n=== *** NEW DISCOVERIES — unknown parameter sets *** ===")
            for r in new_discoveries:
                orig = r.get("connection_set_original_indices", [])
                s = f"  {kind}({r['n']},{r['k']},{r['t']},{r['lambda']},{r['mu']}) " \
                    f"on Group {r['lib_id']} ({r['group_name']})"
                if args.verbose:
                    s += f": S = {{{', '.join(str(x) for x in orig)}}}"
                print(s)

        if known_confirmed:
            if args.verbose:
                print(f"\n=== Found {kind} Cayley graphs (known parameter sets) ===")
                for r in known_confirmed:
                    orig = r.get("connection_set_original_indices", [])
                    print(f"  {kind}({r['n']},{r['k']},{r['t']},{r['lambda']},{r['mu']}) "
                          f"on Group {r['lib_id']} ({r['group_name']}): "
                          f"S = {{{', '.join(str(x) for x in orig)}}}")
            else:
                print(f"\n{len(known_confirmed)} known parameter set(s) confirmed (use --verbose to list)")


if __name__ == "__main__":
    main()
