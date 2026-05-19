#!/usr/bin/env python3
"""Run the Cayley DSRG search for a single parameter set.

Designed for SLURM array jobs: reads one row from a params CSV by index,
creates an output directory, writes a progress CSV upfront, and updates
it as each group completes.

Usage:
    python -m cayley_search.run_single \
        --params larger_params.csv \
        --index 0 \
        --output-dir cayley_data \
        [--batch-size 100000]
"""

from __future__ import annotations

import csv
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .generate import (
    _classify_elements,
    _count_batches,
    _count_t_valid_subsets,
    _t_valid_batches,
    build_adjacency,
    check_dsrg,
    load_group_tables,
)


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def run_single(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    output_dir: Path,
    batch_size: int = 100_000,
    device: torch.device | str = "cpu",
    noninteractive: bool = False,
) -> None:
    """Search for DSRGs in a single parameter set, saving results incrementally."""

    task_dir = output_dir / f"{n}_{k}_{t}_{lambda_}_{mu}"
    task_dir.mkdir(parents=True, exist_ok=True)
    progress_csv = task_dir / "progress.csv"

    total_subsets = comb(n - 1, k)
    print(f"DSRG({n}, {k}, {t}, {lambda_}, {mu})")
    print(f"Total {k}-subsets of {n-1} non-identity elements: {total_subsets:,}")
    print(f"Output: {task_dir}")

    # Check feasibility
    lhs = k * (k - lambda_) - t
    rhs = (n - k - 1) * mu
    if lhs != rhs:
        print(f"INFEASIBLE: k(k-λ)-t={lhs} != (n-k-1)μ={rhs}")
        _write_progress(progress_csv, [{
            "group_lib_id": "",
            "group_name": "",
            "status": "infeasible",
            "t_valid_count": 0,
            "num_dsrgs": 0,
            "elapsed_s": 0,
        }])
        return

    # Load groups
    include_abelian = t == k
    groups = load_group_tables(n, device=device, include_abelian=include_abelian)
    print(f"Groups of order {n}: {len(groups)}")

    if not groups:
        print("No groups to search")
        _write_progress(progress_csv, [{
            "group_lib_id": "",
            "group_name": "",
            "status": "no_groups",
            "t_valid_count": 0,
            "num_dsrgs": 0,
            "elapsed_s": 0,
        }])
        return

    # Load existing progress if resuming, treating any "running" row as not done
    previous_done: dict[int, dict] = {}
    if progress_csv.exists():
        with open(progress_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "done":
                    try:
                        previous_done[int(row["group_lib_id"])] = row
                    except ValueError:
                        pass

    # Build progress rows, preserving results for already-completed groups
    progress_rows: list[dict] = []
    for group in groups:
        if group.library_id in previous_done:
            progress_rows.append(previous_done[group.library_id])
        else:
            involutions, pairs = _classify_elements(group)
            tv = _count_t_valid_subsets(len(involutions), len(pairs), k, t)
            progress_rows.append({
                "group_lib_id": group.library_id,
                "group_name": group.name,
                "status": "queued",
                "t_valid_count": tv,
                "num_dsrgs": 0,
                "elapsed_s": 0,
            })
    _write_progress(progress_csv, progress_rows)

    if previous_done:
        print(f"Resuming: {len(previous_done)} group(s) already done, skipping.")

    # Search each group
    for gi, group in enumerate(groups):
        row = progress_rows[gi]
        tv = int(row["t_valid_count"])

        if row["status"] == "done":
            print(f"\n  {group.name} (lib_id={group.library_id}): already done, skipping.")
            continue

        print(f"\n  {group.name} (lib_id={group.library_id}): {tv:,} t-valid subsets")

        if tv == 0:
            row["status"] = "done"
            row["elapsed_s"] = 0
            _write_progress(progress_csv, progress_rows)
            print(f"    No t-valid subsets — skipped")
            continue

        row["status"] = "running"
        _write_progress(progress_csv, progress_rows)

        involutions, pairs = _classify_elements(group)
        found_subsets: list[torch.Tensor] = []
        t0 = time.perf_counter()
        checked = 0
        total_batches = _count_batches(len(involutions), len(pairs), k, t, batch_size)
        batch_iter = _t_valid_batches(involutions, pairs, k, t, batch_size, device)

        if noninteractive:
            bi = -1
            for bi, batch in enumerate(batch_iter):
                checked += batch.shape[0]
                dsrg = check_dsrg(batch, group, t, lambda_, mu)
                if dsrg.shape[0] > 0:
                    found_subsets.append(dsrg)
                if (bi + 1) % 200 == 0:
                    found_so_far = sum(s.shape[0] for s in found_subsets)
                    elapsed = time.perf_counter() - t0
                    bps = (bi + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"    batch {bi+1}/{total_batches}"
                        f"  generated={checked}  found={found_so_far}"
                        f"  elapsed={_fmt_elapsed(elapsed)}  {bps:.1f} batch/s"
                    )
            if bi >= 0 and (bi + 1) % 200 != 0:
                found_so_far = sum(s.shape[0] for s in found_subsets)
                elapsed = time.perf_counter() - t0
                bps = (bi + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"    batch {bi+1}/{total_batches}"
                    f"  generated={checked}  found={found_so_far}"
                    f"  elapsed={_fmt_elapsed(elapsed)}  {bps:.1f} batch/s"
                )
        else:
            from tqdm import tqdm

            pbar = tqdm(
                batch_iter,
                total=total_batches,
                desc=f"    {group.name}",
                unit="batch",
            )
            for batch in pbar:
                checked += batch.shape[0]
                dsrg = check_dsrg(batch, group, t, lambda_, mu)
                if dsrg.shape[0] > 0:
                    found_subsets.append(dsrg)
                pbar.set_postfix(
                    generated=checked,
                    found=sum(s.shape[0] for s in found_subsets),
                )

        elapsed = time.perf_counter() - t0

        if found_subsets:
            all_found = torch.cat(found_subsets, dim=0)
            count = all_found.shape[0]
            adj = build_adjacency(all_found, group).cpu().numpy().astype(np.uint8)
            npz_file = task_dir / f"dsrg_{n}_{k}_{t}_{lambda_}_{mu}_g{group.library_id}.npz"
            np.savez_compressed(npz_file, adjacency=adj)
            print(f"    {count} DSRGs found, saved to {npz_file.name}")
        else:
            count = 0
            print(f"    No DSRGs")

        row["status"] = "done"
        row["num_dsrgs"] = count
        row["elapsed_s"] = round(elapsed, 1)
        _write_progress(progress_csv, progress_rows)

    total_found = sum(r["num_dsrgs"] for r in progress_rows)
    print(f"\nDone. {total_found} total DSRGs across {len(groups)} groups.")


def _write_progress(path: Path, rows: list[dict]) -> None:
    """Write (or overwrite) the progress CSV."""
    fieldnames = ["group_lib_id", "group_name", "status", "t_valid_count", "num_dsrgs", "elapsed_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Cayley DSRG search for a single parameter set.",
        usage="%(prog)s n k t lambda mu [options]\n       %(prog)s --params FILE --index N [options]",
    )
    parser.add_argument("positional", nargs="*", type=int, metavar="N", help="n k t lambda mu")
    parser.add_argument("--params", type=Path, default=None, help="CSV/Excel with n,k,t,lambda,mu columns")
    parser.add_argument("--index", type=int, default=None, help="Row index (0-based) in the params file")
    parser.add_argument("--output-dir", type=Path, default=Path("cayley_data"), help="Base output directory")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Subsets per GPU batch")
    parser.add_argument("--noninteractive", action="store_true", help="Print periodic logs instead of tqdm progress bars")
    args = parser.parse_args()

    if args.params is not None:
        if args.index is None:
            parser.error("--index is required when using --params")

        pf = args.params
        if pf.suffix in (".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"):
            params_df = pd.read_excel(pf)
        else:
            params_df = pd.read_csv(pf)

        params_df.columns = params_df.columns.str.strip()
        for col in params_df.columns:
            if params_df[col].dtype == object:
                params_df[col] = params_df[col].str.strip()

        params_df = params_df.dropna(subset=["n"]).reset_index(drop=True)

        if args.index >= len(params_df):
            print(f"Error: index {args.index} out of range (file has {len(params_df)} rows)")
            sys.exit(1)

        row = params_df.iloc[args.index]
        n = int(row["n"])
        k = int(row["k"])
        t = int(row["t"])
        lambda_ = int(row["lambda"])
        mu = int(row["mu"])
    elif len(args.positional) == 5:
        n, k, t, lambda_, mu = args.positional
    else:
        parser.error("provide n k t lambda mu, or use --params FILE --index N")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    run_single(
        n, k, t, lambda_, mu,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        noninteractive=args.noninteractive,
    )
