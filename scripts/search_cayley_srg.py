#!/usr/bin/env python3
"""Search for undirected SRGs as Cayley graphs over all groups of order n.

Usage:
    cd .claude/worktrees/agent-a6a4cffa
    uv run python3 /path/to/search_cayley_srg.py 69,20,7,5 85,30,11,10
"""

import sys
import time

sys.path.insert(0, "src")

from ilp.models.cayley_dsrg import search_all_groups


def run_one(n, k, lam, mu):
    t = k  # undirected: all edges reciprocal

    print(f"Searching for SRG({n},{k},{lam},{mu}) as Cayley graph")
    print(f"  Searching ALL groups (abelian + nonabelian)\n")

    t0 = time.perf_counter()
    results = search_all_groups(
        n, k, t, lam, mu,
        use_aut_pruning=True,
        undirected=True,
        include_abelian=True,
    )
    elapsed = time.perf_counter() - t0

    found = [r for r in results if r.get("connection_set") is not None]
    infeasible = [r for r in results if r["status"] == "Infeasible"]
    other = len(results) - len(found) - len(infeasible)

    print(f"\nSRG({n},{k},{lam},{mu}): {len(results)} groups, {elapsed:.1f}s")
    print(f"  {len(found)} found, {len(infeasible)} infeasible"
          + (f", {other} other" if other else ""))

    if found:
        print(f"\nSolutions:")
        for r in found:
            orig = r["connection_set_original_indices"]
            print(f"  Group {r['lib_id']} ({r['group_name']}): "
                  f"S = {{{', '.join(str(x) for x in orig)}}}")

    return found


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} n,k,lam,mu [n,k,lam,mu ...]")
        sys.exit(1)

    param_sets = []
    for arg in sys.argv[1:]:
        parts = arg.split(",")
        if len(parts) != 4:
            print(f"Bad parameter set '{arg}', expected n,k,lam,mu")
            sys.exit(1)
        param_sets.append(tuple(int(x) for x in parts))

    total_found = 0
    for i, (n, k, lam, mu) in enumerate(param_sets):
        if i > 0:
            print(f"\n{'='*60}\n")
        total_found += len(run_one(n, k, lam, mu))

    if len(param_sets) > 1:
        print(f"\n{'='*60}")
        print(f"Total: {total_found} Cayley SRGs found across {len(param_sets)} parameter sets")


if __name__ == "__main__":
    main()
