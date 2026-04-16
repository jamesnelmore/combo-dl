#!/usr/bin/env python3
"""Sweep ILP over all n=48 parameter set × group pairs not yet resolved by exhaustive search.

Runs sequentially in an interactive job. Skips pairs already done by exhaustive search
or already solved by the ILP (output JSON exists).

Usage:
    python scripts/sweep_ilp_48.py [--time-limit 60] [--tasks cayley_48_tasks.csv]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tasks", default="cayley_48_tasks.csv")
    parser.add_argument("--output-dir", default="cayley_ilp_results")
    args = parser.parse_args()

    tasks = pd.read_csv(args.tasks)
    remaining = tasks[tasks["exhaustive_status"] != "done"].reset_index(drop=True)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(remaining)
    print(f"Tasks to run: {total}  (time limit: {args.time_limit}s each)")
    print(f"Estimated max wall time: {total * args.time_limit / 3600:.1f}h")
    print()

    t_start = time.perf_counter()
    for i, row in remaining.iterrows():
        n, k, t, lam, mu, lib_id = (
            int(row["n"]), int(row["k"]), int(row["t"]),
            int(row["lambda"]), int(row["mu"]), int(row["lib_id"]),
        )
        outfile = outdir / f"dsrg_{n}_{k}_{t}_{lam}_{mu}_g{lib_id}.json"

        if outfile.exists():
            result = json.loads(outfile.read_text())
            print(f"[{i+1}/{total}] SKIP ({result['status']})  {n}_{k}_{t}_{lam}_{mu} g{lib_id} ({row['group_name']})")
            continue

        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                sys.executable, "scripts/cayley_ilp_worker.py",
                "--n", str(n), "--k", str(k), "--t", str(t),
                "--lambda", str(lam), "--mu", str(mu),
                "--lib-id", str(lib_id),
                "--output-dir", args.output_dir,
                "--time-limit", str(args.time_limit),
                "--threads", str(args.threads),
            ],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            print(f"[{i+1}/{total}] ERROR ({elapsed:.1f}s)  {n}_{k}_{t}_{lam}_{mu} g{lib_id}: {proc.stderr.strip()[-200:]}")
            continue

        try:
            result = json.loads(proc.stdout.strip().split("\n")[-1])
            status = result["status"]
        except Exception:
            status = "parse_error"

        elapsed_total = time.perf_counter() - t_start
        rate = (i + 1) / elapsed_total * 3600
        print(f"[{i+1}/{total}] {status} ({elapsed:.1f}s)  {n}_{k}_{t}_{lam}_{mu} g{lib_id} ({row['group_name']})")

    print()
    print(f"Done in {(time.perf_counter() - t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
