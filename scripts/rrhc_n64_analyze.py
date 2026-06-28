#!/usr/bin/env python3
"""Aggregate RRHC sweep results and print a progress + findings table.

Reads one or more result directories (one line per (param, group) check) and
reports overall progress and a per-(n, param) findings table.

Usage:
    python scripts/rrhc_n64_analyze.py
    python scripts/rrhc_n64_analyze.py --results-dir rrhc_n64_results rrhc_test_results
    python scripts/rrhc_n64_analyze.py --groups-found
"""

import argparse
from collections import defaultdict
from pathlib import Path

# NumberSmallGroups(n) for n values that appear in the DSRG CSV. Used to
# compute the "done / total" group count per (n, param). Extend as needed.
NSG = {
    6: 2, 8: 5, 10: 2, 12: 5, 14: 2, 15: 1, 16: 14, 18: 5, 20: 5, 21: 2,
    22: 2, 24: 15, 25: 2, 26: 2, 27: 5, 28: 4, 30: 4, 32: 51, 33: 1, 34: 2,
    35: 1, 36: 14, 38: 2, 39: 2, 40: 14, 42: 6, 44: 4, 45: 2, 46: 2, 48: 52,
    49: 2, 50: 5, 51: 1, 52: 5, 54: 15, 55: 2, 56: 13, 57: 2, 58: 2, 60: 13,
    62: 2, 63: 4, 64: 256, 65: 1, 66: 4, 68: 5, 69: 1, 70: 4, 72: 50, 74: 2,
    75: 3, 76: 4, 77: 1, 78: 6, 80: 52, 81: 15, 82: 2, 84: 15, 85: 1, 86: 2,
    87: 1, 88: 12, 90: 10, 91: 1, 92: 4, 93: 2, 94: 2, 95: 1, 96: 231, 98: 5,
    99: 2, 100: 16, 102: 4, 104: 14, 105: 2, 106: 2, 108: 45, 110: 6,
}


def parse_line(line):
    # GAP wraps long FOUND difference-set lists across multiple indented lines;
    # only the first line of a record is left-aligned and starts with the int n.
    if not line or line[0].isspace():
        return None
    parts = line.split(None, 7)
    if len(parts) < 7:
        return None
    try:
        n, lib_id, k, t, lam, mu = (int(parts[i]) for i in range(6))
    except ValueError:
        return None
    status = parts[6]
    if status not in ("FOUND", "NONE"):
        return None
    return {"n": n, "lib_id": lib_id, "k": k, "t": t, "lambda": lam, "mu": mu, "status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        nargs="+",
        default=["rrhc_n64_results", "rrhc_test_results"],
        help="One or more result directories to scan",
    )
    ap.add_argument(
        "--groups-found",
        action="store_true",
        help="List the lib_ids of groups with FOUND results per param set",
    )
    args = ap.parse_args()

    dirs = [Path(d) for d in args.results_dir if Path(d).exists()]
    if not dirs:
        raise SystemExit(f"No existing dirs among: {args.results_dir}")

    # (n, k, t, lambda, mu) -> {"found": set(lib_id), "none": set(lib_id)}
    by_param = defaultdict(lambda: {"found": set(), "none": set()})

    n_files = 0
    for d in dirs:
        for fpath in sorted(d.glob("*.txt")):
            n_files += 1
            for line in fpath.read_text().splitlines():
                r = parse_line(line)
                if r is None:
                    continue
                key = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
                bucket = "found" if r["status"] == "FOUND" else "none"
                by_param[key][bucket].add(r["lib_id"])

    params_sorted = sorted(by_param.keys())
    total_checked = sum(len(v["found"]) + len(v["none"]) for v in by_param.values())
    # n=64 sweep: 16 params × 256 groups.
    # n in [65, 95] sweep: 3150 (group, param) pairs (from rrhc_tasks.json).
    total_expected = 16 * 256 + 3150

    pct = 100 * total_checked / total_expected if total_expected else 0.0
    print(
        f"(group, param) pairs checked: {total_checked} / {total_expected} "
        f"({pct:.1f}%)  [scanned {n_files} files in {len(dirs)} dir(s)]"
    )
    print()

    # Findings table
    print(
        f"{'n':>3} {'k':>3} {'t':>3} {'lam':>4} {'mu':>3}  "
        f"{'found':>5} {'none':>5} {'done':>5} {'total':>5}"
    )
    print("-" * 52)
    total_found = 0
    for key in params_sorted:
        n, k, t, lam, mu = key
        v = by_param[key]
        nf, nn = len(v["found"]), len(v["none"])
        total_found += nf
        total_n = NSG.get(n, "?")
        print(
            f"{n:>3} {k:>3} {t:>3} {lam:>4} {mu:>3}  "
            f"{nf:>5} {nn:>5} {nf + nn:>5} {str(total_n):>5}"
        )
    print("-" * 52)
    print(f"Total FOUND: {total_found}")
    print(
        f"(group, param) pairs checked: {total_checked} / {total_expected} "
        f"({pct:.1f}%)"
    )

    if args.groups_found:
        print()
        print("Group IDs with FOUND results:")
        for key in params_sorted:
            v = by_param[key]
            if not v["found"]:
                continue
            ids = sorted(v["found"])
            n, k, t, lam, mu = key
            print(f"  n={n} k={k} t={t} lambda={lam} mu={mu}: {ids}")


if __name__ == "__main__":
    main()
