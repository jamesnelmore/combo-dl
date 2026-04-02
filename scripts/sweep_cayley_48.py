#!/usr/bin/env python3
"""Sweep all open DSRG parameter sets at n=48 across all nonabelian groups."""

import csv
import subprocess
import sys
import time

sys.path.insert(0, "src")

from ilp.models.cayley_dsrg import (
    GAP_BIN,
    _extract_status,
    build_cayley_dsrg,
    load_cayley_data,
)


def get_nonabelian_groups(n):
    gap_script = f"""LoadPackage("smallgrp");;
for i in [1..NumberSmallGroups({n})] do
  G := SmallGroup({n}, i);
  if not IsAbelian(G) then
    Print(i, " ", StructureDescription(G), "\\n");
  fi;
od;
Print("DONE\\n");
QUIT;
"""
    proc = subprocess.run(
        [GAP_BIN, "-q"],
        input=gap_script,
        capture_output=True,
        text=True,
        timeout=60,
    )
    groups = []
    for line in proc.stdout.strip().splitlines():
        if line.strip() == "DONE":
            break
        parts = line.strip().split(maxsplit=1)
        if parts:
            groups.append((int(parts[0]), parts[1] if len(parts) > 1 else "?"))
    return groups


def main():
    import pandas as pd

    df = pd.read_csv("dsrg_parameters.csv")
    open48 = df[(df.n == 48) & (df.Status == "open") & (df.k <= 24)]
    params = list(open48[["n", "k", "t", "lambda", "mu"]].itertuples(index=False))

    print(f"Loading nonabelian groups of order 48...")
    groups = get_nonabelian_groups(48)
    print(f"  {len(groups)} groups found")

    print(f"Loading group data from GAP...")
    t0 = time.perf_counter()
    group_data = {}
    for i, (lib_id, name) in enumerate(groups):
        group_data[lib_id] = (load_cayley_data(48, lib_id), name)
        print(f"  [{i+1}/{len(groups)}] Group {lib_id} ({name})", flush=True)
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    outfile = "cayley_dsrg_48_results.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "k", "t", "lambda", "mu",
            "lib_id", "group_name", "status", "wall_seconds",
            "num_vars", "connection_set",
        ])

    total_found = 0
    total_t0 = time.perf_counter()

    for pi, (n, k, t, lam, mu) in enumerate(params):
        label = f"DSRG(48,{k},{t},{lam},{mu})"
        print(f"[{pi+1}/{len(params)}] {label}", flush=True)
        found = 0
        infeasible = 0
        param_t0 = time.perf_counter()

        rows_to_write = []

        for lib_id, name in groups:
            gd, _ = group_data[lib_id]
            model, x_vars = build_cayley_dsrg(
                n, k, t, lam, mu, gd, use_aut_pruning=True, quiet=True,
            )
            model.setParam("MIPFocus", 1)
            model.setParam("TimeLimit", 30)

            st = time.perf_counter()
            model.optimize()
            solve_t = time.perf_counter() - st

            status = _extract_status(model)
            cs = None
            if status == "Optimal" and model.SolCount > 0:
                cs = [g for g in range(gd.num_nonid) if round(x_vars[g].X) == 1]
                found += 1
                cs_str = " ".join(str(gd.nonid_order[g]) for g in cs)
                print(f"  ** FOUND on Group {lib_id} ({name}) in {solve_t:.2f}s  S={{{cs_str}}}")
            elif status == "Infeasible":
                infeasible += 1
                cs_str = ""
            else:
                cs_str = ""
                print(f"  Group {lib_id} ({name}): {status} in {solve_t:.2f}s")

            rows_to_write.append([
                n, k, t, lam, mu,
                lib_id, name, status, f"{solve_t:.4f}",
                model.NumVars, cs_str,
            ])

        param_time = time.perf_counter() - param_t0
        total_found += found
        summary = f"  {found} found, {infeasible} infeasible"
        remaining = len(groups) - found - infeasible
        if remaining > 0:
            summary += f", {remaining} other"
        print(f"{summary} ({param_time:.1f}s)\n")

        # Append results for this parameter set
        with open(outfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

    total_time = time.perf_counter() - total_t0
    print(f"{'='*60}")
    print(f"Done. {total_found} Cayley DSRGs found across {len(params)} parameter sets")
    print(f"Total wall time: {total_time:.0f}s")
    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
