#!/usr/bin/env python3
"""Print each FOUND result as GAP-loadable code: group + connection set.

For every FOUND record in the result files, emits:
    # n=64 lib_id=3 k=22 t=18 lambda=10 mu=6
    G := SmallGroup(64, 3);;
    AssignGeneratorVariables(G);;
    D := [ f3*f4, ... ];;

Paste into GAP to reconstruct the group and the difference set.

Usage:
    python scripts/rrhc_print_found.py
    python scripts/rrhc_print_found.py --results-dir rrhc_n64_results rrhc_results
    python scripts/rrhc_print_found.py --filter n=64 k=22
"""

import argparse
import re
from pathlib import Path

HEADER_RE = re.compile(
    r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(FOUND|NONE)\b"
)


def iter_records(text):
    """Yield (n, lib_id, k, t, lambda, mu, status, body_after_status) records.

    GAP writes FOUND lines as:
        n lib_id k t lambda mu FOUND [ f1*f2, f3, ...
          continuation,
          more ]
    Continuation lines start with whitespace; record start lines do not.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        n, lib_id, k, t, lam, mu, status = m.groups()
        rest = lines[i][m.end():].lstrip()
        i += 1
        while i < len(lines) and lines[i] and lines[i][0].isspace():
            rest += " " + lines[i].strip()
            i += 1
        yield {
            "n": int(n), "lib_id": int(lib_id), "k": int(k), "t": int(t),
            "lambda": int(lam), "mu": int(mu), "status": status, "body": rest,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", nargs="+",
                    default=["rrhc_n64_results", "rrhc_results", "rrhc_test_results"])
    ap.add_argument("--filter", nargs="*", default=[],
                    help="key=value pairs to filter on, e.g. n=64 k=22")
    args = ap.parse_args()

    filters = {}
    for f in args.filter:
        k, v = f.split("=", 1)
        filters[k] = int(v)

    dirs = [Path(d) for d in args.results_dir if Path(d).exists()]
    count = 0
    for d in dirs:
        for fpath in sorted(d.glob("*.txt")):
            for rec in iter_records(fpath.read_text()):
                if rec["status"] != "FOUND":
                    continue
                if any(rec.get(k) != v for k, v in filters.items()):
                    continue
                count += 1
                n, lib_id = rec["n"], rec["lib_id"]
                k, t, lam, mu = rec["k"], rec["t"], rec["lambda"], rec["mu"]
                print(f"# n={n} lib_id={lib_id} k={k} t={t} lambda={lam} mu={mu}")
                print(f"G := SmallGroup({n}, {lib_id});;")
                print(f"AssignGeneratorVariables(G);;")
                print(f"D := {rec['body']};;")
                print()
    if count == 0:
        print("(no FOUND records matched)")


if __name__ == "__main__":
    main()
