# %% [markdown]
# # Cayley DSRG Results Analysis
#
# Analyze the HPC Cayley graph search results: parameter coverage,
# hit rates, and DSRG counts per parameter set.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "../cayley_data/hpc_cayley"
df = pd.read_csv(f"{results_dir}/results.csv")

# %% [markdown]
# ## Summary statistics
#
# Each row in results.csv is one (parameter set, group) pair.
# Aggregate to the parameter-set level first.

# %%
# Deduplicate to one row per parameter set using the 'row' column
params = df.groupby("row").agg(
    n=("n", "first"),
    k=("k", "first"),
    t=("t", "first"),
    lam=("lambda", "first"),
    mu=("mu", "first"),
    num_groups=("num_groups", "first"),
    total_dsrgs=("total_dsrgs", "first"),
    num_groups_with_hits=("num_dsrgs", lambda x: (x > 0).sum()),
).reset_index()

total_param_sets = len(params)
param_sets_with_hits = (params["total_dsrgs"] > 0).sum()
param_sets_no_hits = total_param_sets - param_sets_with_hits

print(f"Parameter sets searched:    {total_param_sets}")
print(f"  with at least one DSRG:   {param_sets_with_hits} ({100*param_sets_with_hits/total_param_sets:.1f}%)")
print(f"  with no DSRGs:            {param_sets_no_hits}")
print(f"Range of n:                 {params['n'].min()} – {params['n'].max()}")
print(f"Total DSRGs found (before dedup): {params['total_dsrgs'].sum():,}")

# %%
# Full summary table
print(params[["row", "n", "k", "t", "lam", "mu", "num_groups", "total_dsrgs", "num_groups_with_hits"]]
      .to_string(index=False))

# %% [markdown]
# ## DSRG counts per parameter set

# %%
fig, ax = plt.subplots(figsize=(14, 5))

colors = np.where(params["total_dsrgs"] > 0, "steelblue", "salmon")
ax.bar(range(total_param_sets), params["total_dsrgs"], color=colors, edgecolor="none")
ax.set_xlabel("Parameter set index")
ax.set_ylabel("Total DSRGs found")
ax.set_title("DSRGs found per parameter set (n, k, t, λ, μ)")
ax.set_yscale("symlog", linthresh=1)

# Label the top-5 bars
top5 = params.nlargest(5, "total_dsrgs")
for _, r in top5.iterrows():
    ax.annotate(
        f"({int(r['n'])},{int(r['k'])},{int(r['t'])})\n{int(r['total_dsrgs']):,}",
        xy=(r["row"], r["total_dsrgs"]),
        fontsize=7, ha="center", va="bottom",
    )

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Hit rate by group order n

# %%
by_n = params.groupby("n").agg(
    param_sets=("row", "count"),
    hits=("total_dsrgs", lambda x: (x > 0).sum()),
    total_dsrgs=("total_dsrgs", "sum"),
).reset_index()
by_n["hit_rate"] = by_n["hits"] / by_n["param_sets"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(by_n["n"], by_n["hit_rate"], color="steelblue", width=0.8)
ax.set_xlabel("n (group order)")
ax.set_ylabel("Fraction of param sets with DSRGs")
ax.set_title("Hit rate by group order")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

ax = axes[1]
ax.bar(by_n["n"], by_n["total_dsrgs"], color="darkorange", width=0.8)
ax.set_xlabel("n (group order)")
ax.set_ylabel("Total DSRGs (all param sets)")
ax.set_title("Total DSRGs by group order")
ax.set_yscale("symlog", linthresh=1)

fig.tight_layout()
plt.show()

print(by_n.to_string(index=False))
