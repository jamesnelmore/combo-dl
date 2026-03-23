# %%
# Load all DSRG adjacency matrices from the HPC results directory
# and parse parameters from filenames.

import re
from pathlib import Path

import numpy as np

results_dir = Path("../../hpc_cayley")

# Parse filename pattern: dsrg_{n}_{k}_{t}_{lambda}_{mu}_g{lib_id}.npz
pattern = re.compile(r"dsrg_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_g(\d+)\.npz")

all_graphs = []
for f in sorted(results_dir.glob("dsrg_*.npz")):
    m = pattern.match(f.name)
    if not m:
        continue
    n, k, t, lam, mu, gid = (int(x) for x in m.groups())
    adj = np.load(f)["adjacency"]
    all_graphs.append({
        "file": f.name,
        "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
        "group_lib_id": gid,
        "adj": adj,
        "count": adj.shape[0],
    })

total = sum(g["count"] for g in all_graphs)
print(f"Loaded {len(all_graphs)} files, {total} total adjacency matrices")

# %%
# Verify every adjacency matrix satisfies the DSRG equation:
#   A² = t·I + λ·A + μ·(J - I - A)
# which rearranges to:
#   A² = (t - μ)·I + (λ - μ)·A + μ·J

failures = 0
checked = 0

for g in all_graphs:
    n, k, t, lam, mu = g["n"], g["k"], g["t"], g["lambda"], g["mu"]
    I = np.eye(n, dtype=int)
    J = np.ones((n, n), dtype=int)

    for i in range(g["count"]):
        A = g["adj"][i].astype(int)
        A2 = A @ A
        expected = (t - mu) * I + (lam - mu) * A + mu * J

        if not np.array_equal(A2, expected):
            print(f"FAIL: {g['file']} index {i}")
            failures += 1
        checked += 1

print(f"Checked {checked} matrices: {failures} failures")

# %%
# Also verify basic regularity:
# - No self-loops (diagonal of A is 0)
# - Every row sums to k (out-degree)
# - Every column sums to k (in-degree)

reg_failures = 0

for g in all_graphs:
    k = g["k"]
    for i in range(g["count"]):
        A = g["adj"][i]
        diag_ok = np.all(np.diag(A) == 0)
        row_ok = np.all(A.sum(axis=1) == k)
        col_ok = np.all(A.sum(axis=0) == k)
        if not (diag_ok and row_ok and col_ok):
            print(f"REGULARITY FAIL: {g['file']} index {i}")
            reg_failures += 1

print(f"Regularity check: {reg_failures} failures out of {total}")

# %%
# Print a representative from DSRG(34, 16, 8, 7, 8)

import matplotlib.pyplot as plt
import networkx as nx

target = [g for g in all_graphs if g["n"] == 34 and g["k"] == 16 and g["t"] == 8]
if target:
    g = target[0]
    A = g["adj"][0]
    print(f"DSRG({g['n']}, {g['k']}, {g['t']}, {g['lambda']}, {g['mu']})")
    print(f"From: {g['file']} (index 0 of {g['count']})")

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    print(f"Vertices: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Graph drawing
    ax = axes[0]
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120, node_color="steelblue")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5,
                           arrows=True, arrowsize=5, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white")
    ax.set_title(f"DSRG({g['n']}, {g['k']}, {g['t']}, {g['lambda']}, {g['mu']})", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    # Adjacency matrix heatmap
    ax = axes[1]
    ax.imshow(A, cmap="Greys", interpolation="nearest")
    ax.set_title("Adjacency matrix", fontsize=13)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.tight_layout()
    plt.show()
