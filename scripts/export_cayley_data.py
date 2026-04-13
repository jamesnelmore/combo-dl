#!/usr/bin/env python3
"""Export all Cayley DSRG search results into three normalized CSVs.

Consolidates data from:
  - old_cayley_results/  (exhaustive GPU search, 4 subdirs with results.csv)
  - cayley_data/          (exhaustive GPU search, progress.csv per param set)
  - cayley_ilp_results/   (ILP search, one JSON per group-param solve)

Outputs to cayley_export/:
  parameters.csv     — one row per (n,k,t,λ,μ)
  graphs.csv         — one row per unique DSRG up to isomorphism
  cayley_searches.csv — per (group, graph) for found; per (group, params) otherwise

Usage:
    python scripts/export_cayley_data.py [--output-dir cayley_export]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pynauty import Graph, autgrp, canon_label, certificate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cayley_search.generate import GroupTable, load_group_tables

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _adj_to_pynauty(adj: np.ndarray) -> Graph:
    n = adj.shape[0]
    adjacency_dict: dict[int, list[int]] = {}
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0].tolist()
        if neighbors:
            adjacency_dict[i] = neighbors
    return Graph(number_of_vertices=n, directed=True, adjacency_dict=adjacency_dict)


def encode_digraph6(adj: np.ndarray) -> str:
    """Encode an adjacency matrix in digraph6 format."""
    n = adj.shape[0]
    result = ["&"]
    if n < 63:
        result.append(chr(n + 63))
    elif n < 258048:
        result.append(chr(126))
        result.append(chr((n >> 12) + 63))
        result.append(chr(((n >> 6) & 63) + 63))
        result.append(chr((n & 63) + 63))
    else:
        raise ValueError(f"n={n} too large for digraph6")
    # Column-major bit packing
    bits = []
    for j in range(n):
        for i in range(n):
            bits.append(int(adj[i, j]))
    while len(bits) % 6 != 0:
        bits.append(0)
    for k in range(0, len(bits), 6):
        val = 0
        for b in range(6):
            val = (val << 1) | bits[k + b]
        result.append(chr(val + 63))
    return "".join(result)


def canonical_adjacency(adj: np.ndarray, perm: list[int]) -> np.ndarray:
    """Apply canonical labeling permutation to an adjacency matrix."""
    inv_perm = np.argsort(perm)
    return adj[np.ix_(inv_perm, inv_perm)]


def extract_connection_set(adj: np.ndarray, identity_idx: int) -> list[int]:
    """Extract the connection set S from a Cayley graph adjacency matrix."""
    return sorted(np.nonzero(adj[identity_idx])[0].tolist())


def build_adj_from_connset(
    connset_0idx: list[int], inv: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Reconstruct adjacency matrix from a 0-indexed connection set."""
    n = table.shape[0]
    left_mult = table[inv]  # left_mult[i,j] = table[inv[i], j]
    mask = np.zeros(n, dtype=np.uint8)
    mask[connset_0idx] = 1
    return mask[left_mult]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Params = tuple[int, int, int, int, int]  # (n, k, t, lambda, mu)


@dataclass
class NpzTask:
    """An NPZ file to process from exhaustive search."""
    params: Params
    group_lib_id: int
    group_name: str
    npz_path: Path


@dataclass
class IlpResult:
    """A single ILP solve result."""
    params: Params
    group_lib_id: int
    group_name: str
    status: str  # "found", "infeasible", "inconclusive"
    connset_0idx: list[int] | None  # 0-indexed, only for found


@dataclass
class InfeasibleGroup:
    """A group proven to have no Cayley DSRG for given params (exhaustive)."""
    params: Params
    group_lib_id: int
    group_name: str


@dataclass
class GraphRecord:
    params: Params
    digraph6: str
    aut_group_order: int


# ---------------------------------------------------------------------------
# Phase 1: Build inventory
# ---------------------------------------------------------------------------

def _load_old_results_inventory(
    old_dir: Path,
) -> tuple[list[NpzTask], dict[Params, int], set[Params], set[Params]]:
    """Parse old_cayley_results/ results.csv files.

    Returns:
        npz_tasks: NPZ files to process
        num_groups_map: params -> num_nonabelian_groups (from results.csv)
        all_searched_params: param sets where all groups were searched
        old_all_abelian: param sets where num_groups == 0
    """
    subdirs = ["cayley_results", "cayley_results2", "cayley_results40_60", "cayley_continued"]
    npz_tasks: list[NpzTask] = []
    num_groups_map: dict[Params, int] = {}
    all_searched_params: set[Params] = set()
    old_all_abelian: set[Params] = set()
    found_groups: dict[Params, set[int]] = defaultdict(set)

    for sub in subdirs:
        csv_path = old_dir / sub / "results.csv"
        if not csv_path.exists():
            continue
        base = csv_path.parent
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                p: Params = (
                    int(row["n"]), int(row["k"]), int(row["t"]),
                    int(row["lambda"]), int(row["mu"]),
                )
                ng = row.get("num_groups", "").strip()
                if ng:
                    num_groups_map[p] = int(ng)
                    if int(ng) == 0:
                        old_all_abelian.add(p)

                gid_str = str(row.get("group_lib_id", "")).strip()
                file_str = str(row.get("file", "")).strip()

                if not gid_str or not file_str:
                    # All groups searched, 0 DSRGs
                    all_searched_params.add(p)
                    continue

                gid = int(gid_str)
                npz_path = base / file_str
                if npz_path.exists():
                    found_groups[p].add(gid)
                    npz_tasks.append(NpzTask(
                        params=p,
                        group_lib_id=gid,
                        group_name=row.get("group_name", "").strip(),
                        npz_path=npz_path,
                    ))
                    all_searched_params.add(p)

    return npz_tasks, num_groups_map, all_searched_params, old_all_abelian


def _load_cayley_data_inventory(
    cd_dir: Path,
) -> tuple[list[NpzTask], list[InfeasibleGroup], set[Params], dict[Params, set[int]], set[Params]]:
    """Parse cayley_data/ progress.csv files.

    Returns:
        npz_tasks: NPZ files to process
        infeasible: groups proven to have no DSRG
        all_abelian: param sets with no nonabelian groups
        done_groups: params -> set of group_lib_ids that completed exhaustive search
        all_cd_params: all param sets that have a directory (including incomplete)
    """
    npz_tasks: list[NpzTask] = []
    infeasible: list[InfeasibleGroup] = []
    all_abelian: set[Params] = set()
    done_groups: dict[Params, set[int]] = defaultdict(set)
    all_cd_params: set[Params] = set()

    for d in sorted(cd_dir.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        if len(parts) != 5:
            continue
        p: Params = tuple(int(x) for x in parts)  # type: ignore[assignment]
        all_cd_params.add(p)
        prog = d / "progress.csv"
        if not prog.exists():
            continue

        with open(prog) as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "").strip()

                if status == "no_groups":
                    all_abelian.add(p)
                    continue

                gid_str = str(row.get("group_lib_id", "")).strip()
                if not gid_str or gid_str == "nan":
                    continue
                gid = int(float(gid_str))
                gname = row.get("group_name", "").strip()

                if status != "done":
                    continue  # skip running/queued

                done_groups[p].add(gid)
                num_dsrgs = int(row.get("num_dsrgs", 0))

                if num_dsrgs == 0:
                    infeasible.append(InfeasibleGroup(p, gid, gname))
                    continue

                # Look for NPZ file
                n, k, t, lam, mu = p
                npz_name = f"dsrg_{n}_{k}_{t}_{lam}_{mu}_g{gid}.npz"
                npz_path = d / npz_name
                if npz_path.exists():
                    npz_tasks.append(NpzTask(p, gid, gname, npz_path))

    return npz_tasks, infeasible, all_abelian, done_groups, all_cd_params


def _load_ilp_inventory(ilp_dir: Path) -> list[IlpResult]:
    """Parse cayley_ilp_results/ JSON files."""
    results: list[IlpResult] = []
    if not ilp_dir.exists():
        return results

    for f in sorted(ilp_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        p: Params = (data["n"], data["k"], data["t"], data["lambda"], data["mu"])
        gid = data["lib_id"]
        gname = data.get("group_name", "")
        raw_status = data.get("status", "")

        if raw_status == "Optimal" and data.get("connection_set_original_indices"):
            results.append(IlpResult(p, gid, gname, "found", data["connection_set_original_indices"]))
        elif raw_status == "Infeasible":
            results.append(IlpResult(p, gid, gname, "infeasible", None))
        elif raw_status == "TimeLimit":
            results.append(IlpResult(p, gid, gname, "inconclusive", None))

    return results


# ---------------------------------------------------------------------------
# Phase 2: Group table loading
# ---------------------------------------------------------------------------

def _load_all_group_tables(
    orders: set[int],
    large_order_groups: dict[int, set[int]],
) -> dict[tuple[int, int], GroupTable]:
    """Load group tables for all needed orders.

    For small orders (in `orders`), load all nonabelian groups via GAP.
    For large orders (in `large_order_groups`), load individually via the ILP
    module's load_cayley_data to avoid loading all groups of that order.
    """
    cache: dict[tuple[int, int], GroupTable] = {}

    for n in sorted(orders):
        print(f"  Loading group tables for n={n}...", flush=True)
        try:
            tables = load_group_tables(n, device="cpu")
            for gt in tables:
                cache[(n, gt.library_id)] = gt
        except Exception as e:
            print(f"    WARNING: GAP failed for n={n}: {e}")

    # Large orders: load individually
    for n, lib_ids in large_order_groups.items():
        if n in orders:
            continue  # already loaded
        for lid in lib_ids:
            print(f"  Loading group table for SmallGroup({n},{lid})...", flush=True)
            try:
                from ilp.models.cayley_dsrg import load_cayley_data
                cgd = load_cayley_data(n, lid)
                # Build a GroupTable-compatible object from CayleyGroupData
                # We only need identity_pos for connection set extraction from ILP
                # and inv/table for building adjacency from connection set
                # CayleyGroupData doesn't have full table in the same form,
                # so we use load_group_tables for just this one group
                tables = load_group_tables(n, device="cpu")
                for gt in tables:
                    if gt.library_id == lid:
                        cache[(n, lid)] = gt
                        break
            except Exception as e:
                print(f"    WARNING: Failed for SmallGroup({n},{lid}): {e}")

    return cache


# ---------------------------------------------------------------------------
# Phase 3 & 4: Canonical labeling, dedup, and connection set extraction
# ---------------------------------------------------------------------------

def _process_all(
    npz_tasks: list[NpzTask],
    ilp_found: list[IlpResult],
    group_cache: dict[tuple[int, int], GroupTable],
) -> tuple[
    dict[int, GraphRecord],       # graph_info
    list[dict],                   # search_rows (found only)
    dict[Params, int],            # raw_counts
]:
    """Process all NPZ files and ILP found results.

    Returns graph_info, found search_rows, and raw DSRG counts.
    """
    cert_to_id: dict[bytes, int] = {}
    graph_info: dict[int, GraphRecord] = {}
    next_id = 0
    search_rows: list[dict] = []
    seen_graph_group: set[tuple[int, int, int]] = set()  # (graph_id, n, group_lib_id)
    raw_counts: dict[Params, int] = defaultdict(int)

    CHUNK = 1000

    # --- Exhaustive search NPZ files ---
    total_npz = len(npz_tasks)
    for ti, task in enumerate(npz_tasks):
        n, k, t, lam, mu = task.params
        gt = group_cache.get((n, task.group_lib_id))
        if gt is None:
            print(f"  WARNING: No group table for SmallGroup({n},{task.group_lib_id}), skipping {task.npz_path.name}")
            continue

        identity_idx = gt.identity

        try:
            data = np.load(task.npz_path, mmap_mode="r")
            adj_all = data["adjacency"]
        except Exception as e:
            print(f"  WARNING: Failed to load {task.npz_path}: {e}")
            continue

        total_matrices = adj_all.shape[0]
        raw_counts[task.params] += total_matrices

        if (ti + 1) % 50 == 0 or ti == total_npz - 1:
            print(f"  NPZ [{ti+1}/{total_npz}] {task.npz_path.name} ({total_matrices} matrices)", flush=True)

        for start in range(0, total_matrices, CHUNK):
            end = min(start + CHUNK, total_matrices)
            batch = np.array(adj_all[start:end])

            for idx in range(batch.shape[0]):
                adj = batch[idx]
                g = _adj_to_pynauty(adj)
                cert = certificate(g)

                if cert not in cert_to_id:
                    perm = canon_label(g)
                    aut_info = autgrp(g)
                    aut_order = int(aut_info[1] * (10 ** aut_info[2]))
                    canon_adj = canonical_adjacency(adj, perm)
                    d6 = encode_digraph6(canon_adj)

                    gid = next_id
                    next_id += 1
                    cert_to_id[cert] = gid
                    graph_info[gid] = GraphRecord(
                        params=task.params,
                        digraph6=d6,
                        aut_group_order=aut_order,
                    )

                graph_id = cert_to_id[cert]
                key = (graph_id, n, task.group_lib_id)
                if key not in seen_graph_group:
                    seen_graph_group.add(key)
                    conn = extract_connection_set(adj, identity_idx)
                    conn_1idx = [x + 1 for x in conn]
                    search_rows.append({
                        "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                        "group_lib_id": task.group_lib_id,
                        "group_name": task.group_name,
                        "search_method": "exhaustive",
                        "status": "found",
                        "graph_id": graph_id,
                        "connection_set": ";".join(str(x) for x in conn_1idx),
                    })

    # --- ILP found results ---
    print(f"  Processing {len(ilp_found)} ILP found results...", flush=True)
    for ilp in ilp_found:
        n, k, t, lam, mu = ilp.params
        gt = group_cache.get((n, ilp.group_lib_id))
        if gt is None:
            print(f"  WARNING: No group table for SmallGroup({n},{ilp.group_lib_id}), skipping ILP result")
            continue

        inv_np = gt.inv.numpy().astype(int)
        table_np = gt.table.numpy().astype(int)
        adj = build_adj_from_connset(ilp.connset_0idx, inv_np, table_np)
        g = _adj_to_pynauty(adj)
        cert = certificate(g)

        if cert not in cert_to_id:
            perm = canon_label(g)
            aut_info = autgrp(g)
            aut_order = int(aut_info[1] * (10 ** aut_info[2]))
            canon_adj = canonical_adjacency(adj, perm)
            d6 = encode_digraph6(canon_adj)

            gid = next_id
            next_id += 1
            cert_to_id[cert] = gid
            graph_info[gid] = GraphRecord(
                params=ilp.params,
                digraph6=d6,
                aut_group_order=aut_order,
            )

        graph_id = cert_to_id[cert]
        key = (graph_id, n, ilp.group_lib_id)
        if key not in seen_graph_group:
            seen_graph_group.add(key)
            conn_1idx = [x + 1 for x in ilp.connset_0idx]
            search_rows.append({
                "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
                "group_lib_id": ilp.group_lib_id,
                "group_name": ilp.group_name,
                "search_method": "inexhaustive",
                "status": "found",
                "graph_id": graph_id,
                "connection_set": ";".join(str(x) for x in sorted(conn_1idx)),
            })

    return graph_info, search_rows, dict(raw_counts)


# ---------------------------------------------------------------------------
# Phase 5 & 6: Assemble CSVs
# ---------------------------------------------------------------------------

def _build_infeasible_rows_old(
    old_all_searched: set[Params],
    old_num_groups: dict[Params, int],
    found_search_rows: list[dict],
    group_cache: dict[tuple[int, int], GroupTable],
    overlap_params: set[Params],
) -> list[dict]:
    """Build infeasible rows for old_cayley_results groups that found nothing."""
    rows: list[dict] = []

    # Build set of (params, group_lib_id) that have found rows from exhaustive
    found_pg = set()
    for r in found_search_rows:
        if r["search_method"] == "exhaustive":
            p = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
            found_pg.add((p, r["group_lib_id"]))

    for p in old_all_searched:
        if p in overlap_params:
            continue  # handled by cayley_data
        n = p[0]
        num_groups = old_num_groups.get(p, 0)
        # Get all nonabelian groups for this order from cache
        order_groups = [(k, gt) for k, gt in group_cache.items() if k[0] == n]
        for (_, lid), gt in order_groups:
            if (p, lid) not in found_pg:
                rows.append({
                    "n": p[0], "k": p[1], "t": p[2], "lambda": p[3], "mu": p[4],
                    "group_lib_id": lid,
                    "group_name": gt.name,
                    "search_method": "exhaustive",
                    "status": "infeasible",
                    "graph_id": "",
                    "connection_set": "",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export Cayley DSRG data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cayley_export")
    args = parser.parse_args()

    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Build inventory ──────────────────────────────────────────
    print("Phase 1: Building inventory...", flush=True)

    old_npz, old_num_groups, old_all_searched, old_all_abelian = _load_old_results_inventory(
        ROOT / "old_cayley_results"
    )
    print(f"  old_cayley_results: {len(old_npz)} NPZ tasks, {len(old_all_searched)} param sets, {len(old_all_abelian)} all-abelian")

    cd_npz, cd_infeasible, cd_all_abelian, cd_done_groups, all_cd_params = _load_cayley_data_inventory(
        ROOT / "cayley_data"
    )
    print(f"  cayley_data: {len(cd_npz)} NPZ tasks, {len(cd_infeasible)} infeasible groups, {len(cd_all_abelian)} all-abelian")

    ilp_results = _load_ilp_inventory(ROOT / "cayley_ilp_results")
    ilp_found = [r for r in ilp_results if r.status == "found"]
    ilp_infeasible = [r for r in ilp_results if r.status == "infeasible"]
    ilp_inconclusive = [r for r in ilp_results if r.status == "inconclusive"]
    print(f"  ILP: {len(ilp_found)} found, {len(ilp_infeasible)} infeasible, {len(ilp_inconclusive)} inconclusive")

    # Resolve overlaps: cayley_data supersedes old_cayley_results
    old_params = set(t.params for t in old_npz)
    cd_params = set(t.params for t in cd_npz) | set(ig.params for ig in cd_infeasible) | cd_all_abelian
    overlap_params = old_params & cd_params
    if overlap_params:
        print(f"  Overlap: {len(overlap_params)} param sets, using cayley_data for these")
        old_npz = [t for t in old_npz if t.params not in overlap_params]

    all_npz = old_npz + cd_npz
    print(f"  Total NPZ tasks after dedup: {len(all_npz)}")

    # ── Phase 2: Load group tables ────────────────────────────────────────
    print("\nPhase 2: Loading group tables...", flush=True)

    # Collect orders needed
    normal_orders: set[int] = set()
    large_order_groups: dict[int, set[int]] = defaultdict(set)

    for task in all_npz:
        normal_orders.add(task.params[0])
    for ig in cd_infeasible:
        normal_orders.add(ig.params[0])
    for p in old_all_searched:
        if p not in overlap_params:
            normal_orders.add(p[0])

    # ILP results: check if order is already covered
    for r in ilp_results:
        n = r.params[0]
        if n > 100:  # large order, load individually
            large_order_groups[n].add(r.group_lib_id)
        else:
            normal_orders.add(n)

    group_cache = _load_all_group_tables(normal_orders, large_order_groups)
    print(f"  Loaded {len(group_cache)} group tables across {len(normal_orders)} orders")

    # ── Phase 3 & 4: Process NPZ + ILP ───────────────────────────────────
    print("\nPhase 3: Canonical labeling and dedup...", flush=True)

    graph_info, found_rows, raw_counts = _process_all(all_npz, ilp_found, group_cache)
    print(f"  Unique graphs: {len(graph_info)}")
    print(f"  Found search rows: {len(found_rows)}")

    # ── Phase 5: Build non-found rows ────────────────────────────────────
    print("\nPhase 5: Building infeasible/inconclusive rows...", flush=True)

    all_search_rows = list(found_rows)

    # cayley_data infeasible (explicit from progress.csv)
    for ig in cd_infeasible:
        all_search_rows.append({
            "n": ig.params[0], "k": ig.params[1], "t": ig.params[2],
            "lambda": ig.params[3], "mu": ig.params[4],
            "group_lib_id": ig.group_lib_id,
            "group_name": ig.group_name,
            "search_method": "exhaustive",
            "status": "infeasible",
            "graph_id": "",
            "connection_set": "",
        })

    # old_cayley_results infeasible (inferred from num_groups - found groups)
    old_infeasible_rows = _build_infeasible_rows_old(
        old_all_searched, old_num_groups, found_rows, group_cache, overlap_params,
    )
    all_search_rows.extend(old_infeasible_rows)
    print(f"  Infeasible from cayley_data: {len(cd_infeasible)}")
    print(f"  Infeasible from old results: {len(old_infeasible_rows)}")

    # ILP infeasible
    for r in ilp_infeasible:
        all_search_rows.append({
            "n": r.params[0], "k": r.params[1], "t": r.params[2],
            "lambda": r.params[3], "mu": r.params[4],
            "group_lib_id": r.group_lib_id,
            "group_name": r.group_name,
            "search_method": "inexhaustive",
            "status": "infeasible",
            "graph_id": "",
            "connection_set": "",
        })

    # ILP inconclusive
    for r in ilp_inconclusive:
        all_search_rows.append({
            "n": r.params[0], "k": r.params[1], "t": r.params[2],
            "lambda": r.params[3], "mu": r.params[4],
            "group_lib_id": r.group_lib_id,
            "group_name": r.group_name,
            "search_method": "inexhaustive",
            "status": "inconclusive",
            "graph_id": "",
            "connection_set": "",
        })

    print(f"  ILP infeasible: {len(ilp_infeasible)}")
    print(f"  ILP inconclusive: {len(ilp_inconclusive)}")
    print(f"  Total search rows: {len(all_search_rows)}")

    # ── Phase 6: Assemble CSVs ───────────────────────────────────────────
    print("\nPhase 6: Writing CSVs...", flush=True)

    # --- cayley_searches.csv ---
    all_search_rows.sort(key=lambda r: (r["n"], r["k"], r["t"], r["lambda"], r["mu"], r["group_lib_id"]))
    searches_path = outdir / "cayley_searches.csv"
    with open(searches_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "k", "t", "lambda", "mu",
            "group_lib_id", "group_name",
            "search_method", "status", "graph_id", "connection_set",
        ])
        writer.writeheader()
        writer.writerows(all_search_rows)
    print(f"  {searches_path}: {len(all_search_rows)} rows")

    # --- graphs.csv ---
    # Count constructions and groups per graph_id from search rows
    graph_constructions: dict[int, int] = defaultdict(int)
    graph_groups: dict[int, set[int]] = defaultdict(set)
    for r in all_search_rows:
        if r["status"] == "found" and r["graph_id"] != "":
            gid = r["graph_id"]
            graph_constructions[gid] += 1
            graph_groups[gid].add(r["group_lib_id"])

    graph_rows = []
    for gid in sorted(graph_info):
        rec = graph_info[gid]
        n, k, t, lam, mu = rec.params
        graph_rows.append({
            "graph_id": gid,
            "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
            "digraph6": rec.digraph6,
            "aut_group_order": rec.aut_group_order,
            "num_constructions": graph_constructions.get(gid, 0),
            "num_groups": len(graph_groups.get(gid, set())),
        })

    graphs_path = outdir / "graphs.csv"
    with open(graphs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "graph_id", "n", "k", "t", "lambda", "mu",
            "digraph6", "aut_group_order", "num_constructions", "num_groups",
        ])
        writer.writeheader()
        writer.writerows(graph_rows)
    print(f"  {graphs_path}: {len(graph_rows)} rows")

    # --- parameters.csv ---
    # Collect all param sets from every source
    all_params: set[Params] = set()
    for r in all_search_rows:
        all_params.add((r["n"], r["k"], r["t"], r["lambda"], r["mu"]))
    for p in cd_all_abelian:
        all_params.add(p)
    for p in old_all_abelian:
        all_params.add(p)
    for p in all_cd_params:
        all_params.add(p)

    # Count unique graphs per param set
    unique_per_params: dict[Params, set[int]] = defaultdict(set)
    for r in all_search_rows:
        if r["status"] == "found" and r["graph_id"] != "":
            p = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
            unique_per_params[p].add(r["graph_id"])

    # Count groups checked exhaustively per param set
    exh_groups: dict[Params, set[int]] = defaultdict(set)
    for r in all_search_rows:
        if r["search_method"] == "exhaustive":
            p = (r["n"], r["k"], r["t"], r["lambda"], r["mu"])
            exh_groups[p].add(r["group_lib_id"])

    # Determine num_nonabelian_groups per order
    groups_per_order: dict[int, int] = {}
    for (n, _), _ in group_cache.items():
        groups_per_order[n] = groups_per_order.get(n, 0) + 1
    # Also use old_num_groups for orders not in cache
    for p, ng in old_num_groups.items():
        n = p[0]
        if n not in groups_per_order:
            groups_per_order[n] = ng

    param_rows = []
    for p in sorted(all_params):
        n, k, t, lam, mu = p
        num_na = groups_per_order.get(n, 0)
        gce = len(exh_groups.get(p, set()))

        if p in cd_all_abelian or p in old_all_abelian:
            status = "all_abelian"
            num_na = 0
        elif gce >= num_na and num_na > 0:
            status = "exhaustive"
        else:
            status = "partial"

        param_rows.append({
            "n": n, "k": k, "t": t, "lambda": lam, "mu": mu,
            "num_nonabelian_groups": num_na,
            "groups_checked_exhaustively": gce,
            "search_status": status,
            "num_dsrgs_raw": raw_counts.get(p, 0),
            "num_dsrgs_unique": len(unique_per_params.get(p, set())),
        })

    params_path = outdir / "parameters.csv"
    with open(params_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "k", "t", "lambda", "mu",
            "num_nonabelian_groups", "groups_checked_exhaustively",
            "search_status", "num_dsrgs_raw", "num_dsrgs_unique",
        ])
        writer.writeheader()
        writer.writerows(param_rows)
    print(f"  {params_path}: {len(param_rows)} rows")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Parameter sets: {len(param_rows)}")
    exhaustive_count = sum(1 for r in param_rows if r["search_status"] == "exhaustive")
    partial_count = sum(1 for r in param_rows if r["search_status"] == "partial")
    abelian_count = sum(1 for r in param_rows if r["search_status"] == "all_abelian")
    print(f"  exhaustive: {exhaustive_count}, partial: {partial_count}, all_abelian: {abelian_count}")
    print(f"Unique graphs: {len(graph_rows)}")
    total_raw = sum(r["num_dsrgs_raw"] for r in param_rows)
    print(f"Total raw adjacency matrices: {total_raw}")
    print(f"Search rows: {len(all_search_rows)}")
    print(f"\nOutput written to {outdir}/")


if __name__ == "__main__":
    main()
