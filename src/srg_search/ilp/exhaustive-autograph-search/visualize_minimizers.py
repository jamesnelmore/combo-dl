"""
Visualize the tree that minimises λ₁ + μ for each n.

Produces a single figure with one subplot per n (from 3 to n_max).
Each subplot shows the tree drawn with a spring layout, annotated with
n, the value of λ₁ + μ, λ₁, μ, and the degree sequence.

Usage
-----
    python visualize_minimizers.py [--nmax 18] [--out minimizers.png]
"""

import argparse
import math
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ── helpers (same as search.py) ──────────────────────────────────────────────

def largest_eigenvalue(G: nx.Graph) -> float:
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    return float(np.linalg.eigvalsh(A)[-1])


def matching_number(G: nx.Graph) -> int:
    return len(nx.max_weight_matching(G))


# ── find minimisers ─────────────────────────────────────────────────────────

def find_minimizers(n_max: int):
    """Return a list of dicts, one per n in [3 .. n_max]."""
    results = []
    for n in range(3, n_max + 1):
        best_val = float("inf")
        best_tree = None
        best_lam = None
        best_mu = None
        count = 0

        for T in nx.nonisomorphic_trees(n):
            count += 1
            lam1 = largest_eigenvalue(T)
            mu = matching_number(T)
            val = lam1 + mu
            if val < best_val:
                best_val = val
                best_tree = T.copy()
                best_lam = lam1
                best_mu = mu

        threshold = math.sqrt(n - 1) + 1
        deg_seq = sorted((d for _, d in best_tree.degree()), reverse=True)

        results.append(dict(
            n=n,
            tree=best_tree,
            val=best_val,
            lam1=best_lam,
            mu=best_mu,
            threshold=threshold,
            margin=best_val - threshold,
            deg_seq=deg_seq,
            num_trees=count,
        ))
        print(
            f"n={n:2d}  trees={count:>7d}  "
            f"min(λ₁+μ)={best_val:.4f}  λ₁={best_lam:.4f}  μ={best_mu}  "
            f"deg={deg_seq}"
        )

    return results


# ── degree-based colour palette ─────────────────────────────────────────────

def _node_colors(G):
    """Colour nodes by degree: leaves light, hubs dark."""
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    if degrees.max() == degrees.min():
        return ["#4A90D9"] * len(degrees)
    normed = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    cmap = plt.cm.YlOrRd  # light-yellow (leaf) → red (hub)
    return [cmap(0.15 + 0.75 * v) for v in normed]


# ── layout helper ───────────────────────────────────────────────────────────

def _layout(T):
    """Pick a nice deterministic layout for the tree."""
    # Use a rooted layout centred on a node of maximum degree.
    hub = max(T.nodes, key=lambda v: T.degree(v))
    # BFS layers from hub → hierarchical feel
    try:
        pos = nx.nx_agraph.graphviz_layout(T, prog="dot", root=hub)
    except Exception:
        # graphviz may not be installed; fall back
        pos = nx.spring_layout(T, seed=42, k=1.8 / math.sqrt(max(T.number_of_nodes(), 1)))
    return pos


# ── main plotting routine ──────────────────────────────────────────────────

def plot_minimizers(results, out_path: str | None = None):
    k = len(results)
    # Grid dimensions: prefer roughly 4 columns
    ncols = min(4, k)
    nrows = math.ceil(k / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5.5 * nrows),
        constrained_layout=True,
    )
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(
        "Trees minimising  λ₁ + μ  (by number of vertices n)",
        fontsize=16, fontweight="bold", y=1.01,
    )

    for idx, rec in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        T = rec["tree"]
        n = rec["n"]
        pos = _layout(T)

        node_size = max(350 - 8 * n, 80)
        font_size = max(10 - 0.3 * n, 5)
        edge_width = max(2.5 - 0.06 * n, 0.8)

        nx.draw_networkx_edges(
            T, pos, ax=ax,
            width=edge_width, edge_color="#888888", alpha=0.7,
        )
        nx.draw_networkx_nodes(
            T, pos, ax=ax,
            node_size=node_size,
            node_color=_node_colors(T),
            edgecolors="#333333", linewidths=0.8,
        )
        nx.draw_networkx_labels(
            T, pos, ax=ax,
            font_size=font_size, font_color="black", font_weight="bold",
        )

        # Build annotation text
        deg_str = ", ".join(str(d) for d in rec["deg_seq"])
        title = f"n = {n}"
        info = (
            f"λ₁ + μ = {rec['val']:.4f}\n"
            f"λ₁ = {rec['lam1']:.4f},  μ = {rec['mu']}\n"
            f"√(n−1)+1 = {rec['threshold']:.4f}\n"
            f"margin = {rec['margin']:+.4f}\n"
            f"deg = [{deg_str}]"
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.text(
            0.02, 0.02, info,
            transform=ax.transAxes,
            fontsize=7.5, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
        )
        ax.set_axis_off()

    # Turn off any unused axes
    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_axis_off()

    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"\nSaved figure to {out_path}")
    else:
        plt.show()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the tree minimising λ₁+μ for each n."
    )
    parser.add_argument(
        "--nmax", type=int, default=18,
        help="Largest n to enumerate (default: 18)",
    )
    parser.add_argument(
        "--out", type=str, default="minimizers.png",
        help="Output image path (omit or set to '' to show interactively)",
    )
    args = parser.parse_args()

    if args.nmax < 3:
        print("n_max must be >= 3", file=sys.stderr)
        sys.exit(1)

    results = find_minimizers(args.nmax)
    out = args.out if args.out else None
    plot_minimizers(results, out_path=out)


if __name__ == "__main__":
    main()
