"""Plot ILP benchmark results: solve time by formulation and parameter set."""

import glob
import json

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


def load(job: str, base: str = "bench_output") -> dict:
    results = {}
    for f in sorted(glob.glob(f"{base}/{job}/*/result.json")):
        with open(f) as fp:
            r = json.load(fp)
        results[r["index"]] = r
    return results


def main() -> None:
    jobs = {
        "feasible only": load("naive_ilp"),
        "feasible + lex-leader": load("srg-fix-lex"),
        "feasible + hybrid lex": load("srg-fix-hybrid"),
        "relaxed": load("srg-relaxed-known-no-lex"),
        "relaxed + hybrid lex": load("srg-relaxed-known"),
        "relaxed + lex-leader": load("srg-relaxed-known-leader"),
    }

    timeout = 28500

    # x-axis: all parameter sets sorted lexicographically by (n, k, λ, μ)
    ref = jobs["feasible only"]
    all_indices = sorted(
        ref.keys(), key=lambda i: (ref[i]["n"], ref[i]["k"], ref[i]["lambda"], ref[i]["mu"])
    )
    x_labels = [
        f"({ref[i]['n']},{ref[i]['k']},{ref[i]['lambda']},{ref[i]['mu']})" for i in all_indices
    ]

    colors = {
        "feasible only": "#2196F3",
        "feasible + lex-leader": "#1565C0",
        "feasible + hybrid lex": "#00BCD4",
        "relaxed": "#4CAF50",
        "relaxed + hybrid lex": "#FF9800",
        "relaxed + lex-leader": "#E91E63",
    }
    markers = {
        "feasible only": "o",
        "feasible + lex-leader": "^",
        "feasible + hybrid lex": "s",
        "relaxed": "o",
        "relaxed + hybrid lex": "s",
        "relaxed + lex-leader": "^",
    }

    fig, ax = plt.subplots(figsize=(16, 6))

    x_pos = np.arange(len(all_indices))
    offsets = np.linspace(-0.2, 0.2, len(jobs))
    rng = np.random.default_rng(42)

    for offset, (name, data) in zip(offsets, jobs.items()):
        solved_x, solved_y = [], []
        timeout_x, timeout_y = [], []
        for xi, idx in enumerate(all_indices):
            r = data.get(idx)
            if r is None:
                continue
            status = r["status"]
            if status == "Optimal":
                solved_x.append(xi + offset)
                solved_y.append(r["wall_seconds"])
            elif status == "Infeasible":
                continue
            else:
                timeout_x.append(xi + offset)
                timeout_y.append(timeout * rng.uniform(0.97, 1.03))

        ax.scatter(
            solved_x,
            solved_y,
            c=colors[name],
            marker=markers[name],
            s=40,
            alpha=0.85,
            label=name,
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            timeout_x,
            timeout_y,
            c=colors[name],
            marker=markers[name],
            s=40,
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )

    ax.axhline(timeout, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="8h timeout")
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("Solve time (s, log scale)")
    ax.set_xlabel("SRG parameters (n, k, λ, μ)")
    ax.set_title("ILP benchmark: solve time by formulation and parameter set")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mlines.Line2D([], [], color=colors[n], marker=markers[n], linestyle="None",
                      markersize=7, label=n)
        for n in colors
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    out = "plots/ilp_benchmark_scatter.png"
    plt.savefig(out, dpi=200)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
