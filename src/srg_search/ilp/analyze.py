# %% Imports and data loading
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_JSON = "sweep_results.json"

df = pd.read_json(RESULTS_JSON)
df = df.rename(columns={"lambda": "lam"})  # lambda is a Python keyword
df

# %% Summary statistics
print(f"Total problems: {len(df)}")
print(f"Optimal:    {(df.status == 'Optimal').sum()}")
print(f"Infeasible: {(df.status == 'Infeasible').sum()}")
print()
print(df.groupby("status")["wall_seconds"].describe().round(3))

# %% Solve time by problem — horizontal bar chart
fig, ax = plt.subplots(figsize=(9, 6))

labels = [f"({r.n},{r.k},{r.t},{r.lam},{r.mu})" for r in df.itertuples()]
colors = ["steelblue" if s == "Optimal" else "tomato" for s in df.status]

bars = ax.barh(labels, df.wall_seconds, color=colors)

ax.set_xlabel("Wall time (seconds)")
ax.set_title("DSRG ILP solve times  (n, k, t, λ, μ)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x/60:.1f} min" if x >= 60 else f"{x:.1f} s"
))

# legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue", label="Optimal"),
    Patch(color="tomato",    label="Infeasible"),
])

plt.tight_layout()
plt.show()

# %% Solve time vs n — scatter, coloured by status
fig, ax = plt.subplots(figsize=(7, 5))

for status, grp in df.groupby("status"):
    color = "steelblue" if status == "Optimal" else "tomato"
    ax.scatter(grp.n, grp.wall_seconds, label=status, color=color, s=60, zorder=3)

ax.set_xlabel("n  (number of vertices)")
ax.set_ylabel("Wall time (seconds)")
ax.set_title("Solve time vs graph size")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Solve time vs k/n ratio — useful proxy for problem density
df["k_over_n"] = df.k / df.n

fig, ax = plt.subplots(figsize=(7, 5))

for status, grp in df.groupby("status"):
    color = "steelblue" if status == "Optimal" else "tomato"
    sc = ax.scatter(grp.k_over_n, grp.wall_seconds, label=status,
                    color=color, s=grp.n * 4, alpha=0.7, zorder=3)

ax.set_xlabel("k / n  (edge density)")
ax.set_ylabel("Wall time (seconds)")
ax.set_title("Solve time vs edge density  (marker size ∝ n)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Log-scale solve time distribution
fig, ax = plt.subplots(figsize=(7, 4))

times = df.wall_seconds
log_bins = np.logspace(np.log10(times.min()), np.log10(times.max()), 15)

for status, grp in df.groupby("status"):
    color = "steelblue" if status == "Optimal" else "tomato"
    ax.hist(grp.wall_seconds, bins=log_bins, alpha=0.7, color=color, label=status)

ax.set_xscale("log")
ax.set_xlabel("Wall time (seconds, log scale)")
ax.set_ylabel("Count")
ax.set_title("Distribution of solve times")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% Parameter correlations with solve time (Optimal only)
opt = df[df.status == "Optimal"].copy()

params = ["n", "k", "t", "lam", "mu", "k_over_n"]
correlations = opt[params].corrwith(opt.wall_seconds).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["steelblue" if v >= 0 else "tomato" for v in correlations]
correlations.plot.bar(ax=ax, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Pearson r  with  wall_seconds")
ax.set_title("Parameter correlation with solve time  (Optimal only)")
ax.set_ylim(-1, 1)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print(correlations.to_string())

# %% Sorted results table
df.sort_values("wall_seconds", ascending=False)[
    ["n", "k", "t", "lam", "mu", "status", "wall_seconds"]
].reset_index(drop=True)
