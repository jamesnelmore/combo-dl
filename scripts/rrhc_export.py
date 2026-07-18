#!/usr/bin/env python3
"""Export RRHC sweep results to a single text file for sharing.

Produces a self-contained document with:
  1. Header (date, scope)
  2. Per-(n, param) findings table (counts of FOUND / NONE / total)
  3. Totals
  4. Every FOUND case as a GAP-loadable snippet (group + connection set)

Usage:
    python scripts/rrhc_export.py
    python scripts/rrhc_export.py --output rrhc_results.txt
    python scripts/rrhc_export.py --results-dir rrhc_n64_results rrhc_results
"""

import argparse
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path

NSG = {
    24: 15, 25: 2, 27: 5, 28: 4, 30: 4, 32: 51, 33: 1, 34: 2, 35: 1, 36: 14,
    38: 2, 39: 2, 40: 14, 42: 6, 44: 4, 45: 2, 46: 2, 48: 52, 49: 2, 50: 5,
    51: 1, 52: 5, 54: 15, 55: 2, 56: 13, 57: 2, 58: 2, 60: 13, 62: 2, 63: 4,
    64: 256, 65: 1, 66: 4, 68: 5, 69: 1, 70: 4, 72: 50, 74: 2, 75: 3, 76: 4,
    77: 1, 78: 6, 80: 52, 81: 15, 82: 2, 84: 15, 85: 1, 86: 2, 87: 1, 88: 12,
    90: 10, 91: 1, 92: 4, 93: 2, 94: 2, 95: 1, 96: 231, 98: 5, 99: 2, 100: 16,
    102: 4, 104: 14, 105: 2, 106: 2, 108: 45, 110: 6,
}

HEADER_RE = re.compile(
    r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(FOUND|NONE)\b"
)


def iter_records(text):
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
    ap.add_argument("--output", default="rrhc_export.txt")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.results_dir if Path(d).exists()]
    if not dirs:
        raise SystemExit(f"No existing dirs among: {args.results_dir}")

    # (n, k, t, lambda, mu) -> {"found": [records], "none_ids": set}
    by_param = defaultdict(lambda: {"found": [], "none_ids": set()})
    n_files = 0
    for d in dirs:
        for fpath in sorted(d.glob("*.txt")):
            n_files += 1
            for rec in iter_records(fpath.read_text()):
                key = (rec["n"], rec["k"], rec["t"], rec["lambda"], rec["mu"])
                if rec["status"] == "FOUND":
                    by_param[key]["found"].append(rec)
                else:
                    by_param[key]["none_ids"].add(rec["lib_id"])

    keys = sorted(by_param)
    lines = []
    L = lines.append

    L("Random-restart hill-climbing DSRG search — results export")
    L(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    L(f"Result directories: {', '.join(str(d) for d in dirs)}")
    L(f"Files scanned: {n_files}")
    L("")
    L("Method: random-restart steepest-descent hill climbing on the group ring")
    L("        difference-set equation (src/random_restart.g, RRHCkill).")
    L("        Each (group, parameter set) is given up to 300 random restarts.")
    L("        A 'FOUND' result is a connection set D with k elements such that")
    L("        e2(D^2 - t*1 - lambda*D - mu*(G - D - 1)) = 0 in Z[G].")
    L("")

    # Summary table
    L("=" * 72)
    L("Findings summary")
    L("=" * 72)
    L(f"  {'n':>3} {'k':>3} {'t':>3} {'lam':>4} {'mu':>3}  "
      f"{'found':>5} {'none':>5} {'done':>5} {'total':>5}")
    L("  " + "-" * 50)
    total_found = 0
    total_pairs = 0
    total_expected = 0
    for key in keys:
        n, k, t, lam, mu = key
        v = by_param[key]
        nf = len(set(r["lib_id"] for r in v["found"]))
        nn = len(v["none_ids"])
        total_found += nf
        total_pairs += nf + nn
        total_n = NSG.get(n)
        if total_n is not None:
            total_expected += total_n
        tot_str = str(total_n) if total_n is not None else "?"
        L(f"  {n:>3} {k:>3} {t:>3} {lam:>4} {mu:>3}  "
          f"{nf:>5} {nn:>5} {nf + nn:>5} {tot_str:>5}")
    L("  " + "-" * 50)
    L(f"  Parameter sets with at least one FOUND: "
      f"{sum(1 for k in keys if by_param[k]['found'])}")
    L(f"  Total FOUND connection sets:           {total_found}")
    L(f"  Total (group, param) pairs checked:    {total_pairs}")
    if total_expected:
        L(f"  Coverage of seen params:               "
          f"{total_pairs}/{total_expected} "
          f"({100*total_pairs/total_expected:.1f}%)")
    L("")

    # Connection sets
    L("=" * 72)
    L("Connection sets (GAP-loadable)")
    L("=" * 72)
    L("")
    L("Each block reconstructs the group via SmallGroup(n, lib_id) and exposes")
    L("its generators as f1, f2, ... via AssignGeneratorVariables(G). The list")
    L("D is the connection set of the Cayley DSRG with parameters (n, k, t,")
    L("lambda, mu).")
    L("")

    any_found = False
    for key in keys:
        v = by_param[key]
        if not v["found"]:
            continue
        any_found = True
        n, k, t, lam, mu = key
        L("-" * 72)
        L(f"# Parameters: n={n}, k={k}, t={t}, lambda={lam}, mu={mu}")
        L(f"# Solutions found in {len(v['found'])} group(s) of order {n}")
        L("-" * 72)
        # Deduplicate by lib_id, keep first solution per group
        seen = set()
        for rec in v["found"]:
            if rec["lib_id"] in seen:
                continue
            seen.add(rec["lib_id"])
            L("")
            L(f"# n={n} lib_id={rec['lib_id']}  "
              f"(k={k}, t={t}, lambda={lam}, mu={mu})")
            L(f"G := SmallGroup({n}, {rec['lib_id']});;")
            L(f"AssignGeneratorVariables(G);;")
            L(f"D := {rec['body']};;")
        L("")
    if not any_found:
        L("(no FOUND results yet)")

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}  ({len(lines)} lines)")
    print(f"  Parameter sets seen:       {len(keys)}")
    print(f"  Total FOUND:               {total_found}")
    print(f"  (group, param) pairs:      {total_pairs}")


if __name__ == "__main__":
    main()
