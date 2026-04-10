#!/usr/bin/env python3
"""Generate CSV and scatter plot of DSRG parameter sets with t-valid statistics.

Adds two columns to the filtered parameter list (n<48, k<=n/2):
  num_cayley  — total labeled Cayley graphs = num_groups(n) * C(n-1, k)
  t_valid     — total (group, subset) pairs where |S ∩ S^{-1}| = t

Then plots % t-valid vs n, coloring known/open differently.
"""

import csv
import math
from math import comb
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS_CSV = Path(__file__).parent.parent / "dsrg_parameters.csv"
OUT_CSV = Path(__file__).parent.parent / "dsrg_params_tvalid.csv"
OUT_PLOT = Path(__file__).parent.parent / "plots" / "dsrg_tvalid_scatter.png"
OUT_PLOT_ABS = Path(__file__).parent.parent / "plots" / "dsrg_tvalid_abs_scatter.png"
OUT_PLOT_SIZE = Path(__file__).parent.parent / "plots" / "dsrg_tvalid_vs_size.png"

NEWLY_REALIZED = {(48, 19, 14, 5, 9), (48, 22, 17, 11, 9)}


# ── Group involution counts via GAP ──────────────────────────────────────────


def get_group_involutions(n_values: set[int]) -> dict[int, list[int]]:
    """Return {n: [inv_count_per_group, ...]} for all groups of each order."""
    lines = ['LoadPackage("smallgrp");;']
    for n in sorted(n_values):
        lines.append(f"""
n := {n};;
groups := AllSmallGroups(n);;
Print("N {n} ", Size(groups), "\\n");
for G in groups do
    inv := Size(Filtered(AsList(G), g -> Order(g) = 2));;
    Print("INV ", inv, "\\n");
od;
""")
    lines.append('Print("DONE\\n");;\nQUIT;\n')

    proc = subprocess.run(
        ["gap", "-q"],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode not in (0, 1):  # GAP often exits 1 on QUIT
        print(f"GAP warning (exit {proc.returncode}):", proc.stderr[:200], file=sys.stderr)

    result: dict[int, list[int]] = {}
    current_n = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("N "):
            current_n = int(line.split()[1])
            result[current_n] = []
        elif line.startswith("INV ") and current_n is not None:
            result[current_n].append(int(line.split()[1]))
    return result


# ── Combinatorial t-valid count ───────────────────────────────────────────────


def count_t_valid(inv: int, p: int, k: int, t: int) -> int:
    """Count size-k subsets S of G\\{e} with |S ∩ S^{-1}| = t.

    inv : number of involutions (order-2 elements) in G
    p   : number of non-involution inverse pairs = (n-1-inv)//2
    k   : connection-set size
    t   : target |S ∩ S^{-1}|

    Each subset is characterized by:
      i involutions chosen (each contributes 1 to |S ∩ S^{-1}|)
      j complete inverse pairs (each contributes 2)
      (k - i - 2j) one-directional elements (each contributes 0)

    Constraint: i + 2j = t  =>  i = t - 2j.
    """
    one_dir = k - t  # elements from pairs where only one direction is in S
    if one_dir < 0:
        return 0
    total = 0
    for j in range(t // 2 + 1):
        i = t - 2 * j
        if i < 0 or i > inv:
            continue
        if j > p:
            continue
        remaining = p - j
        if one_dir > remaining:
            continue
        total += comb(inv, i) * comb(p, j) * comb(remaining, one_dir) * (2**one_dir)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # Read and filter parameters.
    params = []
    with open(PARAMS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n, k = int(row["n"]), int(row["k"])
            if True:
                params.append({
                    "n": n,
                    "k": k,
                    "t": int(row["t"]),
                    "lambda": int(row["lambda"]),
                    "mu": int(row["mu"]),
                    "status": row["Status"],
                })

    unique_n = {p["n"] for p in params}
    print(f"Fetching involution counts from GAP for n in {sorted(unique_n)} ...")
    inv_data = get_group_involutions(unique_n)
    print("Done.")

    # Compute statistics per parameter set.
    for p in params:
        n, k, t = p["n"], p["k"], p["t"]
        groups = inv_data.get(n, [])
        num_groups = len(groups)
        total_cayley = num_groups * comb(n - 1, k)

        t_valid = 0
        for inv in groups:
            pair_count = (n - 1 - inv) // 2
            t_valid += count_t_valid(inv, pair_count, k, t)

        p["num_cayley"] = total_cayley
        p["t_valid"] = t_valid
        p["pct_t_valid"] = (100.0 * t_valid / total_cayley) if total_cayley > 0 else 0.0
        p["log_cayley"] = math.log10(total_cayley) if total_cayley > 0 else 0.0

    # Write CSV.
    fieldnames = ["n", "k", "t", "lambda", "mu", "status", "num_cayley", "t_valid"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(params)
    print(f"CSV written to {OUT_CSV}")

    # Split into plot groups.
    def _key(p):
        return (p["n"], p["k"], p["t"], p["lambda"], p["mu"])

    newly = [p for p in params if _key(p) in NEWLY_REALIZED]
    impossibl = [
        p for p in params if p["status"] == "impossible" and _key(p) not in NEWLY_REALIZED
    ]
    known = [p for p in params if p["status"] == "known" and _key(p) not in NEWLY_REALIZED]
    open_ = [p for p in params if p["status"] == "open" and _key(p) not in NEWLY_REALIZED]

    def _scatter(ax, y_field, ylabel, title, yscale=None, x_field="n"):
        if impossibl:
            ax.scatter(
                [p[x_field] for p in impossibl],
                [p[y_field] for p in impossibl],
                c="lightgray",
                edgecolors="gray",
                linewidths=0.6,
                s=45,
                label="impossible",
                zorder=2,
            )
        ax.scatter(
            [p[x_field] for p in known],
            [p[y_field] for p in known],
            c="steelblue",
            alpha=0.75,
            s=55,
            label="known",
            zorder=3,
        )
        ax.scatter(
            [p[x_field] for p in open_],
            [p[y_field] for p in open_],
            c="crimson",
            alpha=0.9,
            s=90,
            marker="*",
            label="open",
            zorder=4,
        )
        ax.scatter(
            [p[x_field] for p in newly],
            [p[y_field] for p in newly],
            c="limegreen",
            edgecolors="darkgreen",
            linewidths=0.8,
            s=120,
            marker="D",
            label="newly realized",
            zorder=5,
        )
        if yscale:
            ax.set_yscale(yscale)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=11)
        if x_field == "n":
            ax.set_xlabel("n", fontsize=13)
            ax.set_xlim(4, 115)
        ax.grid(True, alpha=0.3, which="both")

    # % t-valid plot
    fig, ax = plt.subplots(figsize=(13, 7))
    _scatter(
        ax,
        "pct_t_valid",
        "% t-valid Cayley graphs",
        "Feasible DSRG parameter sets (n ≤ 110):\n"
        "fraction of (group, connection set) pairs satisfying the t-constraint",
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"Plot saved to {OUT_PLOT}")
    plt.close()

    # Absolute t-valid plot
    fig, ax = plt.subplots(figsize=(13, 7))
    _scatter(
        ax,
        "t_valid",
        r"$\log(\text{Count of } t\text{-valid Cayley graphs})$",
        "Feasible DSRG parameter sets (n ≤ 110):\n"
        "absolute number of (group, connection set) pairs satisfying the t-constraint",
        yscale="log",
    )
    plt.tight_layout()
    plt.savefig(OUT_PLOT_ABS, dpi=150)
    print(f"Plot saved to {OUT_PLOT_ABS}")
    plt.close()

    # % t-valid vs log(total subsets) plot
    fig, ax = plt.subplots(figsize=(13, 7))
    _scatter(
        ax,
        "pct_t_valid",
        "% t-valid Cayley graphs",
        "Feasible DSRG parameter sets (n ≤ 110):\n% t-valid vs search space size",
        x_field="log_cayley",
    )
    ax.set_xlabel("log₁₀(total Cayley graphs)", fontsize=13)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUT_PLOT_SIZE, dpi=150)
    print(f"Plot saved to {OUT_PLOT_SIZE}")
    plt.close()


if __name__ == "__main__":
    main()
