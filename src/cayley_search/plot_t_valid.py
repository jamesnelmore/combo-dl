#!/usr/bin/env python3
"""Plot t-valid subset counts vs n for DSRG parameter sets.

Reads a parameter CSV/spreadsheet, loads group tables from GAP to get
element structure, and plots t-valid counts as a scatter with dot size
proportional to k/n and color mapped to t/k.

Usage:
    python plot_t_valid.py params.csv [--output plot.png]
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate import (
    _classify_elements,
    _count_t_valid_subsets,
    load_group_tables,
)


def main() -> None:
    import sys

    params_file = None
    output_file = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output":
            output_file = sys.argv[i + 1]
            i += 2
        elif params_file is None:
            params_file = sys.argv[i]
            i += 1
        else:
            i += 1

    if params_file is None:
        print("Usage: plot_t_valid.py params.csv [--output plot.png]")
        sys.exit(1)

    pf = Path(params_file)
    if pf.suffix in (".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"):
        df = pd.read_excel(pf)
    else:
        df = pd.read_csv(pf)

    # Cache group tables per n
    group_cache: dict[int, list] = {}

    rows: list[dict] = []
    for _, row in df.iterrows():
        n = int(row["n"])
        k = int(row["k"])
        t = int(row["t"])

        if n not in group_cache:
            print(f"Loading groups for n={n}...")
            group_cache[n] = load_group_tables(n, device="cpu")

        groups = group_cache[n]
        if not groups:
            continue

        for group in groups:
            involutions, pairs = _classify_elements(group)
            count = _count_t_valid_subsets(len(involutions), len(pairs), k, t)
            if count > 0:
                rows.append({
                    "n": n,
                    "k": k,
                    "t": t,
                    "k_over_n": k / n,
                    "t_over_k": t / k,
                    "t_valid_count": count,
                    "total_subsets": comb(n - 1, k),
                    "group": group.name,
                })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("No t-valid subsets found for any parameter set.")
        sys.exit(0)

    import numpy as np

    fig, ax = plt.subplots(figsize=(14, 7))

    # -- Jitter x to reduce overlap --
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.3, 0.3, size=len(plot_df))
    x = plot_df["n"].values + jitter

    # -- Scatter --
    sizes = 30 + 250 * plot_df["k_over_n"]
    scatter = ax.scatter(
        x,
        plot_df["t_valid_count"],
        s=sizes,
        c=plot_df["t_over_k"],
        cmap="plasma",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_xlabel("n (group order)", fontsize=13)
    ax.set_ylabel("t-valid subsets", fontsize=13)
    ax.set_title("T-valid subset counts by group order", fontsize=15)

    # -- Colorbar --
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("t / k", fontsize=12)

    # -- Size legend --
    legend_handles = []
    for ratio, label in [(0.1, "k/n = 0.1"), (0.25, "k/n = 0.25"), (0.45, "k/n = 0.45")]:
        h = ax.scatter([], [], s=30 + 250 * ratio, c="gray", alpha=0.7,
                       edgecolors="white", linewidths=0.4, label=label)
        legend_handles.append(h)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              title="Dot size", title_fontsize=10)

    # -- Styling --
    ax.grid(True, alpha=0.2, which="both")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    ax.set_xlim(plot_df["n"].min() - 1, plot_df["n"].max() + 1)

    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
