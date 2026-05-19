#!/usr/bin/env python3
"""Aggregate Cayley DSRG results from per-parameter-set directories.

Walks cayley_data/*/ directories (produced by run_single.py), builds a
combined results.csv matching the old format, then runs nauty dedup to
produce catalog.csv and provenance.csv.

Usage:
    python -m cayley_search.aggregate cayley_data/ [--output-dir DIR]
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pynauty import Graph, certificate


def _adj_to_pynauty(adj: np.ndarray) -> Graph:
    n = adj.shape[0]
    d: dict[int, list[int]] = {}
    for i in range(n):
        nbrs = np.nonzero(adj[i])[0].tolist()
        if nbrs:
            d[i] = nbrs
    return Graph(number_of_vertices=n, directed=True, adjacency_dict=d)


# Pattern for task directories: {n}_{k}_{t}_{lambda}_{mu}
_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")
# Pattern for npz files: dsrg_{n}_{k}_{t}_{lambda}_{mu}_g{lib_id}.npz
_NPZ_PATTERN = re.compile(r"^dsrg_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_g(\d+)\.npz$")


def aggregate(base_dir: Path, output_dir: Path | None = None) -> None:
    out = output_dir or base_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: build results.csv ────────────────────────────────────────
    results_rows: list[dict] = []
    row_idx = 0

    task_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and _DIR_PATTERN.match(d.name)
    )

    for task_dir in task_dirs:
        m = _DIR_PATTERN.match(task_dir.name)
        n, k, t, lam, mu = (int(x) for x in m.groups())

        progress_csv = task_dir / "progress.csv"
        if not progress_csv.exists():
            print(f"Warning: {task_dir.name} has no progress.csv, skipping")
            continue

        progress = pd.read_csv(progress_csv)
        num_groups = len(progress)

        # Total DSRGs for this parameter set
        total_dsrgs = int(progress["num_dsrgs"].sum())

        npz_files = sorted(task_dir.glob("dsrg_*.npz"))

        if not npz_files:
            # No hits — single row with empty group info
            results_rows.append({
                "row": row_idx,
                "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                "num_groups": num_groups,
                "group_lib_id": "",
                "group_name": "",
                "num_dsrgs": 0,
                "total_dsrgs": 0,
                "file": "",
            })
        else:
            for npz_path in npz_files:
                fm = _NPZ_PATTERN.match(npz_path.name)
                if not fm:
                    continue
                lib_id = int(fm.group(6))

                # Find group name from progress.csv
                prog_row = progress[progress["group_lib_id"] == lib_id]
                group_name = prog_row.iloc[0]["group_name"] if len(prog_row) > 0 else "?"

                adj_all = np.load(npz_path)["adjacency"]
                num_dsrgs = adj_all.shape[0]

                results_rows.append({
                    "row": row_idx,
                    "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                    "num_groups": num_groups,
                    "group_lib_id": lib_id,
                    "group_name": group_name,
                    "num_dsrgs": num_dsrgs,
                    "total_dsrgs": total_dsrgs,
                    "file": npz_path.name,
                })

        row_idx += 1

    results_df = pd.DataFrame(results_rows)
    results_path = out / "results.csv"
    results_df.to_csv(results_path, index=False)

    param_sets = results_df["row"].nunique()
    hits = results_df.groupby("row")["total_dsrgs"].first()
    param_sets_with_hits = (hits > 0).sum()
    print(f"Results: {param_sets} parameter sets, {param_sets_with_hits} with DSRGs")
    print(f"Written to {results_path}")

    # ── Phase 2: nauty dedup ──────────────────────────────────────────────
    print("\nRunning dedup...")

    seen: dict[bytes, int] = {}
    next_id = 0
    provenance: list[dict] = []

    for _, row in results_df.iterrows():
        if not row["file"] or pd.isna(row["file"]):
            continue

        # Resolve npz path from the task directory
        dir_name = f"{int(row['n'])}_{int(row['k'])}_{int(row['t'])}_{int(row['lambda'])}_{int(row['mu'])}"
        npz_path = base_dir / dir_name / row["file"]
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping")
            continue

        adj_all = np.load(npz_path)["adjacency"]

        for i in range(adj_all.shape[0]):
            g = _adj_to_pynauty(adj_all[i])
            cert = certificate(g)

            if cert not in seen:
                seen[cert] = next_id
                next_id += 1

            provenance.append({
                "graph_id": seen[cert],
                "n": int(row["n"]),
                "k": int(row["k"]),
                "t": int(row["t"]),
                "lambda": int(row["lambda"]),
                "mu": int(row["mu"]),
                "group_lib_id": row["group_lib_id"],
                "group_name": row["group_name"],
                "subset_index": i,
                "params_row": row["row"],
            })

    prov_df = pd.DataFrame(provenance)

    # Build catalog
    catalog_rows = []
    for graph_id in range(next_id):
        entries = prov_df[prov_df["graph_id"] == graph_id]
        first = entries.iloc[0]

        groups = entries[["group_lib_id", "group_name"]].drop_duplicates()
        group_strs = [
            f'{int(r["group_lib_id"])}:{r["group_name"]}'
            for _, r in groups.iterrows()
        ]

        catalog_rows.append({
            "graph_id": graph_id,
            "n": int(first["n"]),
            "k": int(first["k"]),
            "t": int(first["t"]),
            "lambda": int(first["lambda"]),
            "mu": int(first["mu"]),
            "num_constructions": len(entries),
            "num_groups": len(groups),
            "groups": "; ".join(group_strs),
        })

    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = out / "catalog.csv"
    prov_path = out / "provenance.csv"
    catalog_df.to_csv(catalog_path, index=False)
    prov_df.to_csv(prov_path, index=False)

    print(f"Total adjacency matrices processed: {len(prov_df)}")
    print(f"Unique graphs (up to isomorphism): {next_id}")
    print(f"Catalog written to {catalog_path}")
    print(f"Provenance written to {prov_path}")


if __name__ == "__main__":
    import sys

    base_dir = None
    output_dir = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output-dir":
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif base_dir is None:
            base_dir = Path(sys.argv[i])
            i += 1
        else:
            i += 1

    if base_dir is None:
        print("Usage: python -m cayley_search.aggregate cayley_data/ [--output-dir DIR]")
        sys.exit(1)

    aggregate(base_dir, Path(output_dir) if output_dir else None)
