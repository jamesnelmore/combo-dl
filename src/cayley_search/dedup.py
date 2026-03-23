#!/usr/bin/env python3
"""Deduplicate Cayley DSRG adjacency matrices via canonical labeling.

Reads results.csv and the .npz files produced by generate.py, computes
nauty canonical certificates for each directed graph, and groups
isomorphic graphs together. Outputs a catalog CSV with one row per
unique graph and a provenance CSV recording all constructions.

Usage:
    python dedup.py results_dir/ [--output-dir DIR]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pynauty import Graph, certificate


def _adj_to_pynauty(adj: np.ndarray) -> Graph:
    """Convert an (n, n) adjacency matrix to a pynauty directed Graph."""
    n = adj.shape[0]
    adjacency_dict: dict[int, list[int]] = {}
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0].tolist()
        if neighbors:
            adjacency_dict[i] = neighbors
    return Graph(number_of_vertices=n, directed=True, adjacency_dict=adjacency_dict)


def dedup(results_dir: Path, output_dir: Path | None = None) -> None:
    out = output_dir or results_dir
    out.mkdir(parents=True, exist_ok=True)

    results_csv = results_dir / "results.csv"
    df = pd.read_csv(results_csv)

    # cert_bytes -> canonical id
    seen: dict[bytes, int] = {}
    next_id = 0

    # One entry per (graph_index, construction)
    provenance: list[dict] = []

    for _, row in df.iterrows():
        if not row["file"] or pd.isna(row["file"]):
            continue

        npz_path = results_dir / row["file"]
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping")
            continue

        adj_all = np.load(npz_path)["adjacency"]
        n_graphs = adj_all.shape[0]
        n = row["n"]

        for i in range(n_graphs):
            adj = adj_all[i]
            g = _adj_to_pynauty(adj)
            cert = certificate(g)

            if cert not in seen:
                seen[cert] = next_id
                next_id += 1

            provenance.append({
                "graph_id": seen[cert],
                "n": n,
                "k": row["k"],
                "t": row["t"],
                "lambda": row["lambda"],
                "mu": row["mu"],
                "group_lib_id": row["group_lib_id"],
                "group_name": row["group_name"],
                "subset_index": i,
                "params_row": row["row"],
            })

    prov_df = pd.DataFrame(provenance)

    # Build catalog: one row per unique graph
    catalog_rows = []
    for graph_id in range(next_id):
        entries = prov_df[prov_df["graph_id"] == graph_id]
        first = entries.iloc[0]
        n = int(first["n"])
        k = int(first["k"])
        t = int(first["t"])
        lam = int(first["lambda"])
        mu = int(first["mu"])

        # All groups that produced this graph
        groups = entries[["group_lib_id", "group_name"]].drop_duplicates()
        group_strs = [
            f'{int(r["group_lib_id"])}:{r["group_name"]}'
            for _, r in groups.iterrows()
        ]

        catalog_rows.append({
            "graph_id": graph_id,
            "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
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

    if len(sys.argv) < 2:
        print("Usage: dedup.py results_dir/ [--output-dir DIR]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output-dir":
            output_dir = Path(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    dedup(results_dir, output_dir)
