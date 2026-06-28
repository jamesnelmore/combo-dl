#!/usr/bin/env python3
"""Generate a JSON task list + print submit command for the RRHC sweep.

Filters dsrg_parameters.csv to open, k < n/2 rows in the given n range, splits
each (param, n) into fixed-size group chunks, and writes rrhc_tasks.json.

Usage:
    python scripts/rrhc_submit.py --n-min 96 --n-max 100
    python scripts/rrhc_submit.py --n-min 96 --n-max 100 --batch-size 3 --submit

--batch-size packs multiple JSON tasks into each Slurm array element so the
total array size stays under Slurm's 999-task limit.  The submit command
printed (and run with --submit) passes BATCH_SIZE via --export=ALL,BATCH_SIZE=N.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

# NumberSmallGroups(n) for n values present in the DSRG CSV.
NSG = {
    24: 15, 25: 2, 27: 5, 28: 4, 30: 4, 32: 51, 33: 1, 34: 2, 35: 1, 36: 14,
    38: 2, 39: 2, 40: 14, 42: 6, 44: 4, 45: 2, 46: 2, 48: 52, 49: 2, 50: 5,
    51: 1, 52: 5, 54: 15, 55: 2, 56: 13, 57: 2, 58: 2, 60: 13, 62: 2, 63: 4,
    64: 256, 65: 1, 66: 4, 68: 5, 69: 1, 70: 4, 72: 50, 74: 2, 75: 3, 76: 4,
    77: 1, 78: 6, 80: 52, 81: 15, 82: 2, 84: 15, 85: 1, 86: 2, 87: 1, 88: 12,
    90: 10, 91: 1, 92: 4, 93: 2, 94: 2, 95: 1, 96: 231, 98: 5, 99: 2, 100: 16,
    102: 4, 104: 14, 105: 2, 106: 2, 108: 45, 110: 6,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dsrg_parameters.csv")
    ap.add_argument("--n-min", type=int, required=True)
    ap.add_argument("--n-max", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=4,
                    help="Groups per array task (default 4)")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--task-file", default="rrhc_tasks.json")
    ap.add_argument("--results-dir", default="rrhc_results")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="JSON tasks per Slurm array element (use to stay under "
                         "Slurm's 999-task array limit; default 1)")
    ap.add_argument("--submit", action="store_true",
                    help="Run sbatch after writing the task file")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    target = [
        r for r in rows
        if r["Status"] == "open"
        and 2 * int(r["k"]) < int(r["n"])
        and args.n_min <= int(r["n"]) <= args.n_max
    ]
    target.sort(key=lambda r: (int(r["n"]), int(r["k"]), int(r["t"]),
                                int(r["lambda"]), int(r["mu"])))

    missing_nsg = {int(r["n"]) for r in target if int(r["n"]) not in NSG}
    if missing_nsg:
        raise SystemExit(f"NSG missing for n values: {sorted(missing_nsg)}")

    tasks = []
    for r in target:
        n = int(r["n"])
        total_groups = NSG[n]
        k, t, lam, mu = int(r["k"]), int(r["t"]), int(r["lambda"]), int(r["mu"])
        start = 1
        while start <= total_groups:
            end = min(start + args.chunk_size - 1, total_groups)
            tasks.append({"n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                          "start": start, "end": end})
            start = end + 1

    Path(args.task_file).write_text(json.dumps(tasks, indent=1) + "\n")
    print(f"Wrote {len(tasks)} tasks to {args.task_file}")
    print(f"Covers {len(target)} param sets over n in [{args.n_min}, {args.n_max}]")

    n_array = (len(tasks) + args.batch_size - 1) // args.batch_size
    if n_array > 999 and args.batch_size == 1:
        min_batch = (len(tasks) + 998) // 999
        raise SystemExit(
            f"Array size {n_array} exceeds Slurm's 999-task limit. "
            f"Re-run with --batch-size {min_batch} or higher."
        )

    if args.batch_size > 1:
        submit_cmd = (f"sbatch --array=0-{n_array-1} "
                      f"--export=ALL,BATCH_SIZE={args.batch_size} "
                      f"scripts/rrhc_array.sh")
        print(f"  ({len(tasks)} JSON tasks → {n_array} array elements, "
              f"batch_size={args.batch_size})")
    else:
        submit_cmd = f"sbatch --array=0-{n_array-1} scripts/rrhc_array.sh"

    print(f"\nSubmit:\n  {submit_cmd}\n")
    if args.submit:
        subprocess.run(submit_cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
