#!/usr/bin/env python3
"""Run the rrhc search over every *open* parameter set in a dsrg_parameters.csv.

Reads a CSV shaped like data/dsrg_parameters.csv (columns n,k,t,lambda,mu and a
status column) and sweeps the nonabelian groups of order n for each row with
`rrhc dsrg ... --all-groups -r <restarts>`. By default every row is swept; pass
--status to restrict to one status class (e.g. --status open). The per-group
verdicts are written, in the searches.csv column layout, to a *new* file (never
appended to the real searches table).

For a slurm job array, sweep one row per task: `--count` prints how many rows
match (to size `--array=1-N`), and `--row N` sweeps just the Nth of them, e.g.
`--row "$SLURM_ARRAY_TASK_ID"` writing to a per-row `rrhc_sweep.N.csv`.

Usage: python3 rrhc_sweep.py params.csv [--status open] [--row N | --start N]
                                        [-r restarts] [-o out.csv] [--seed S]
"""

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

# rrhc prints one "id=<lib_id>: FOUND ..." or "id=<lib_id>: NONE ..." per group.
RESULT_RE = re.compile(r"^id=(\d+):\s*(FOUND|NONE)")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RRHC = REPO / "rrhc" / "target" / "release" / "rrhc"
FIELDS = ["lib_id", "n", "k", "t", "lambda", "mu", "method", "outcome", "num_dpds"]


def sweep(rrhc, n, k, t, lam, mu, restarts, seed):
    """Yield (lib_id, found) for every nonabelian group of order n."""
    cmd = [str(rrhc), "dsrg", *map(str, (n, k, t, lam, mu)), "--all-groups", "-r", str(restarts)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        m = RESULT_RE.match(line)
        if m:
            yield int(m.group(1)), m.group(2) == "FOUND"


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("params", help="input CSV with columns (n, k, t, lambda, mu, status)")
    ap.add_argument(
        "-r",
        "--restarts",
        type=int,
        default=200,
        help="hill-climbing restarts per group (default 1000)",
    )
    ap.add_argument("--status", help="only sweep rows with this status (default: all rows)")
    ap.add_argument(
        "--start",
        type=int,
        default=1,
        metavar="N",
        help="resume from the Nth swept row (1-based, default 1)",
    )
    ap.add_argument(
        "--row",
        type=int,
        metavar="N",
        help="sweep only the Nth matching row (1-based); for a "
        "slurm array pass $SLURM_ARRAY_TASK_ID",
    )
    ap.add_argument(
        "--count", action="store_true", help="print how many rows match (to size --array) and exit"
    )
    ap.add_argument(
        "-o",
        "--out",
        help="output CSV path (default rrhc_sweep.csv, or rrhc_sweep.N.csv with --row)",
    )
    ap.add_argument("--seed", type=int, help="seed for reproducible runs")
    ap.add_argument("--rrhc", default=DEFAULT_RRHC, help="path to the rrhc binary")
    args = ap.parse_args()

    with Path(args.params).open(newline="", encoding="utf8") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if args.status is None
            or (r.get("status") or r.get("Status", "")).lower() == args.status.lower()
        ]

    if args.count:
        print(len(rows))
        return

    # One row (slurm array task) or a resumable range to the end.
    if args.row is not None:
        if not 1 <= args.row <= len(rows):
            sys.exit(f"--row {args.row} out of range 1..{len(rows)}")
        selected = [(args.row, rows[args.row - 1])]
    else:
        selected = list(enumerate(rows[args.start - 1 :], args.start))

    out = args.out or (f"rrhc_sweep.{args.row}.csv" if args.row else "rrhc_sweep.csv")
    with Path(out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in selected:
            n, k, t, lam, mu = (int(r[c]) for c in ("n", "k", "t", "lambda", "mu"))
            print(f"[{i}/{len(rows)}] ({n},{k},{t},{lam},{mu})", file=sys.stderr)
            for lib_id, found in sweep(args.rrhc, n, k, t, lam, mu, args.restarts, args.seed):
                w.writerow({
                    "lib_id": lib_id,
                    "n": n,
                    "k": k,
                    "t": t,
                    "lambda": lam,
                    "mu": mu,
                    "method": "rrhc",
                    "outcome": "found" if found else "heuristic_none",
                    "num_dpds": 1 if found else 0,
                })
            f.flush()


if __name__ == "__main__":
    _main()
