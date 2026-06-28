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

sys.path.insert(0, "src")

try:
    from pynauty import Graph, certificate as nauty_certificate
    from ilp.models.cayley_dsrg import load_cayley_data
    _ISO_AVAILABLE = True
except ImportError:
    _ISO_AVAILABLE = False


def _build_adj_dict(connection_set: list, group_data) -> dict:
    """Build a pynauty adjacency dict from a connection set and group data.

    Vertices: 0 = identity, i+1 = non-identity element i.
    Edge u→v exists iff u⁻¹·v ∈ S.
    """
    S = set(connection_set)
    m = group_data.num_nonid
    adj = {}

    # From identity: u⁻¹·v = v, so edge to non-id j iff j ∈ S
    nbrs = sorted(j + 1 for j in S)
    if nbrs:
        adj[0] = nbrs

    for i in range(m):
        nbrs = []
        # To identity: u⁻¹·identity = u⁻¹ = inv_map[i]; edge iff inv_map[i] ∈ S
        if group_data.inv_map[i] in S:
            nbrs.append(0)
        # To non-id j: u⁻¹·v = products[i][j], -1 means identity (never in S)
        for j in range(m):
            p = group_data.products[i][j]
            if p != -1 and p in S:
                nbrs.append(j + 1)
        if nbrs:
            adj[i + 1] = nbrs

    return adj


def _iso_classes(found_results: list) -> list[list]:
    """Group found results by graph isomorphism class using nauty.

    Returns a list of equivalence classes (each a list of result dicts).
    Results where GAP/nauty fails are kept as singletons.
    """
    classes: dict[bytes | int, list] = {}
    _sentinel = 0

    for r in found_results:
        cert: bytes | int
        try:
            gd = load_cayley_data(r["n"], r["lib_id"])
            adj = _build_adj_dict(r["connection_set"], gd)
            directed = not r.get("undirected", False)
            g = Graph(number_of_vertices=r["n"], directed=directed, adjacency_dict=adj)
            cert = nauty_certificate(g)
        except Exception:
            cert = _sentinel  # unique key so it lands in its own class
            _sentinel -= 1

        classes.setdefault(cert, []).append(r)

    return list(classes.values())


def _load_open_params(csv_path: Path) -> set:
    """Return set of (n,k,t,lambda,mu) tuples with status 'open'."""
    if not csv_path.exists():
        return set()
    result = set()
    with open(csv_path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            row = dict(zip(header, parts))
            if row.get("Status", "").strip().lower() == "open":
                try:
                    result.add((int(row["n"]), int(row["k"]), int(row["t"]),
                                int(row["lambda"]), int(row["mu"])))
                except (KeyError, ValueError):
                    pass
    return result


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

    params_csv = Path("data/dsrg_parameters.csv")
    unknown_params = _load_open_params(params_csv)

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

        iso_note = ""
        if _ISO_AVAILABLE and n_found > 1:
            classes = _iso_classes(found_groups)
            n_distinct = len(classes)
            iso_note = f"  ({n_distinct} distinct)" if n_distinct < n_found else "  (all distinct)"

        print(f"{n:>4} {k:>3} {t:>3} {lam:>3} {mu:>3}  "
              f"{len(group_results):>6} {n_found:>5} {n_infeas:>6} {n_timeout:>7} {total_time:>8.1f}{new_tag}{iso_note}")

        found_groups = [r for r in group_results if r["status"] == "Optimal" and r.get("connection_set")]
        for r in found_groups:
            orig = r.get("connection_set_original_indices", [])
            summary_rows.append({
                "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                "lib_id": r["lib_id"], "group_name": r["group_name"],
                "status": "FOUND",
                "solve_seconds": r.get("solve_seconds", 0),
                "connection_set": " ".join(str(x + 1) for x in orig),
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

            by_param: dict = {}
            for r in new_discoveries:
                by_param.setdefault((r["n"], r["k"], r["t"], r["lambda"], r["mu"]), []).append(r)

            for param_key, group in sorted(by_param.items()):
                n, k, t, lam, mu = param_key
                if _ISO_AVAILABLE and len(group) > 1:
                    classes = _iso_classes(group)
                else:
                    classes = [[r] for r in group]

                n_graphs = len(classes)
                n_constructions = len(group)
                header = f"{kind}({n},{k},{t},{lam},{mu})"
                if n_graphs == 1 and n_constructions > 1:
                    header += f"  [{n_constructions} constructions, all isomorphic]"
                elif n_graphs > 1:
                    header += f"  [{n_graphs} non-isomorphic graphs]"
                print(f"\n  {header}")

                for cls in classes:
                    for r in cls:
                        orig = r.get("connection_set_original_indices", [])
                        s = f"      Group {r['lib_id']} ({r['group_name']})"
                        if args.verbose:
                            s += f": S = {{{', '.join(str(x + 1) for x in orig)}}}"
                        print(s)
                    print()  # blank line between iso classes

        if known_confirmed:
            if args.verbose:
                print(f"\n=== Found {kind} Cayley graphs (known parameter sets) ===")
                for r in known_confirmed:
                    orig = r.get("connection_set_original_indices", [])
                    print(f"  {kind}({r['n']},{r['k']},{r['t']},{r['lambda']},{r['mu']}) "
                          f"on Group {r['lib_id']} ({r['group_name']}): "
                          f"S = {{{', '.join(str(x + 1) for x in orig)}}}")
            else:
                print(f"\n{len(known_confirmed)} known parameter set(s) confirmed (use --verbose to list)")


if __name__ == "__main__":
    main()
