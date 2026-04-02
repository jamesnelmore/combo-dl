#!/usr/bin/env python3
"""Generate and optionally submit SLURM array jobs for Cayley ILP search.

Reads a parameter CSV + enumerates groups via GAP, then writes a task list
and a SLURM batch script. Each array task runs one (param_set, group) pair.

Usage:
    # DSRG search (directed, nonabelian groups only):
    python scripts/cayley_ilp_submit.py dsrg_parameters.csv --filter-n 48 --filter-status open

    # SRG search (undirected, all groups):
    python scripts/cayley_ilp_submit.py srg_open_cases.csv --undirected

    # Just generate, don't submit:
    python scripts/cayley_ilp_submit.py params.csv --dry-run

    # Submit:
    python scripts/cayley_ilp_submit.py params.csv --submit
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

GAP_BIN = "/opt/homebrew/bin/gap"


def get_groups(n, include_abelian=False):
    """Get list of (lib_id, name) for groups of order n."""
    filt = "" if include_abelian else "not IsAbelian(G) and "
    gap_script = f"""LoadPackage("smallgrp");;
for i in [1..NumberSmallGroups({n})] do
  G := SmallGroup({n}, i);
  if {filt}true then
    Print(i, " ", StructureDescription(G), "\\n");
  fi;
od;
Print("DONE\\n");
QUIT;
"""
    proc = subprocess.run(
        [GAP_BIN, "-q"], input=gap_script,
        capture_output=True, text=True, timeout=120,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("params_csv", help="CSV with parameter sets")
    parser.add_argument("--undirected", action="store_true",
                        help="SRG mode: enforce S=S^-1, include abelian groups")
    parser.add_argument("--filter-n", type=int, default=None,
                        help="Only include rows with this n")
    parser.add_argument("--filter-status", type=str, default=None,
                        help="Only include rows with this status (e.g. 'open')")
    parser.add_argument("--max-k-ratio", type=float, default=0.5,
                        help="Only include rows with k/n <= this (default 0.5, avoids complements)")
    parser.add_argument("--output-dir", type=str, default="cayley_ilp_results",
                        help="Directory for result JSON files")
    parser.add_argument("--time-limit", type=float, default=300,
                        help="Per-solve Gurobi time limit in seconds")
    parser.add_argument("--slurm-time", type=str, default="01:00:00",
                        help="SLURM wall time per task")
    parser.add_argument("--slurm-max-concurrent", type=int, default=50,
                        help="Max concurrent SLURM tasks")
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit to SLURM (default: dry run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would be submitted")
    args = parser.parse_args()

    # Read parameters
    import pandas as pd
    pf = Path(args.params_csv)
    if pf.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(pf)
    else:
        df = pd.read_csv(pf)

    df.columns = df.columns.str.strip()

    if args.filter_n is not None:
        df = df[df.n == args.filter_n]
    if args.filter_status is not None and "Status" in df.columns:
        df = df[df.Status == args.filter_status]
    df = df[df.k / df.n <= args.max_k_ratio]

    if args.undirected:
        # SRG CSV has n,k,lambda,mu — derive t=k
        if "t" not in df.columns:
            df = df.copy()
            df["t"] = df["k"]
        param_cols = ["n", "k", "t", "lambda", "mu"]
    else:
        param_cols = ["n", "k", "t", "lambda", "mu"]

    params = []
    for _, r in df.iterrows():
        params.append({
            "n": int(r["n"]), "k": int(r["k"]), "t": int(r.get("t", r["k"])),
            "lambda": int(r["lambda"]), "mu": int(r["mu"]),
        })

    if not params:
        print("No parameter sets match filters.")
        sys.exit(1)

    # Enumerate groups for each unique n
    print(f"Enumerating groups for {len(set(p['n'] for p in params))} unique orders...")
    group_cache = {}
    for p in params:
        n = p["n"]
        if n not in group_cache:
            groups = get_groups(n, include_abelian=args.undirected)
            group_cache[n] = groups
            print(f"  n={n}: {len(groups)} groups")

    # Build task list: (param_set, group)
    tasks = []
    for p in params:
        for lib_id, name in group_cache[p["n"]]:
            tasks.append({**p, "lib_id": lib_id, "group_name": name})

    print(f"\n{len(params)} parameter sets × groups = {len(tasks)} total tasks")

    # Write task list
    task_file = Path(f"cayley_ilp_tasks.json")
    task_file.write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"Task list written to {task_file}")

    # Write SLURM script
    kind = "srg" if args.undirected else "dsrg"
    undirected_flag = " --undirected" if args.undirected else ""
    slurm_script = Path(f"scripts/cayley_ilp_{kind}_array.sh")
    slurm_script.write_text(f"""#!/bin/bash
#SBATCH --job-name=cayley-{kind}
#SBATCH --time={args.slurm_time}
#SBATCH --output=slurm_logs/cayley_{kind}_%A_%a.out
#SBATCH --error=slurm_logs/cayley_{kind}_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-.}}"
mkdir -p slurm_logs {args.output_dir}

# Read task from JSON task list
TASK=$(python3 -c "
import json, sys
tasks = json.load(open('{task_file}'))
t = tasks[$SLURM_ARRAY_TASK_ID]
print(f'--n {{t[\"n\"]}} --k {{t[\"k\"]}} --t {{t[\"t\"]}} --lambda {{t[\"lambda\"]}} --mu {{t[\"mu\"]}} --lib-id {{t[\"lib_id\"]}}')
")

uv run python3 scripts/cayley_ilp_worker.py \\
    $TASK \\
    --output-dir {args.output_dir} \\
    --time-limit {args.time_limit}{undirected_flag}
""")

    print(f"SLURM script written to {slurm_script}")

    max_idx = len(tasks) - 1
    submit_cmd = f"sbatch --array=0-{max_idx}%{args.slurm_max_concurrent} {slurm_script}"
    print(f"\nSubmit command:\n  {submit_cmd}")

    if args.submit and not args.dry_run:
        subprocess.run(submit_cmd, shell=True, check=True)
    else:
        print("\n(Dry run — use --submit to actually submit)")


if __name__ == "__main__":
    main()
