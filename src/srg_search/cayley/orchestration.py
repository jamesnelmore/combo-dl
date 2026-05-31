"""Cayley DSRG search workflows: file I/O, progress tracking, top-level runs.

Two entry points:
  - `run_single`: search all groups of order n for one parameter set, with
    progress.csv checkpointing for SLURM array job resume.
  - `run_one_group`: search a single (group, parameter) pair, writing a
    standalone npz.

Both delegate the per-group GPU work to `subset_enumeration.search_one_group`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .subset_enumeration import (
    BatchUpdate,
    DSRGParams,
    GroupTable,
    analyze_group,
    build_adjacency,
    load_group_tables,
    search_one_group,
)


_PROGRESS_FIELDS = [
    "group_lib_id",
    "group_name",
    "status",
    "t_valid_count",
    "num_dsrgs",
    "elapsed_s",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_progress(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PROGRESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_adjacency(
    found, group: GroupTable, params: DSRGParams, task_dir: Path
) -> Path:
    adj = build_adjacency(found, group).cpu().numpy().astype(np.uint8)
    npz = task_dir / (
        f"dsrg_{params.n}_{params.k}_{params.t}_{params.lambda_}_{params.mu}"
        f"_g{group.library_id}.npz"
    )
    np.savez_compressed(npz, adjacency=adj)
    return npz


def _make_progress_callback(
    group: GroupTable, noninteractive: bool, total_batches: int
):
    """Return an `on_batch` callback wired to tqdm or periodic printing."""
    if noninteractive:
        return _periodic_print_callback(group, total_batches)
    pbar = tqdm(total=total_batches, desc=f"    {group.name}", unit="batch")

    def cb(update: BatchUpdate) -> None:
        pbar.update(1)
        pbar.set_postfix(
            generated=update.subsets_checked, found=update.found_so_far
        )

    cb.close = pbar.close  # type: ignore[attr-defined]
    return cb


def _periodic_print_callback(group: GroupTable, total_batches: int):
    import time

    _ = group  # progress bar shows nothing per-group in noninteractive mode
    t0 = time.perf_counter()

    def cb(update: BatchUpdate) -> None:
        if (update.batch_idx + 1) % 200 != 0:
            return
        elapsed = time.perf_counter() - t0
        bps = (update.batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        print(
            f"    batch {update.batch_idx + 1}/{total_batches}"
            f"  generated={update.subsets_checked}  found={update.found_so_far}"
            f"  elapsed={_fmt_elapsed(elapsed)}  {bps:.1f} batch/s"
        )

    cb.close = lambda: None  # type: ignore[attr-defined]
    return cb


# ---------------------------------------------------------------------------
# Workflow: search all groups for one parameter set
# ---------------------------------------------------------------------------


def run_single(
    params: DSRGParams,
    output_dir: Path,
    *,
    batch_size: int = 100_000,
    device: str | None = None,
    noninteractive: bool = False,
) -> None:
    """Search every group of order n for the given parameter set.

    Writes `<output_dir>/<n>_<k>_<t>_<lambda>_<mu>/progress.csv` incrementally
    and one `dsrg_*.npz` per group that yields hits. Re-running resumes:
    rows with status="done" are kept and their groups are skipped.
    """
    device = device or _pick_device()
    task_dir = output_dir / f"{params.n}_{params.k}_{params.t}_{params.lambda_}_{params.mu}"
    task_dir.mkdir(parents=True, exist_ok=True)
    progress_csv = task_dir / "progress.csv"

    print(params)
    print(f"Output: {task_dir}")
    print(f"Device: {device}")

    if not params.is_feasible():
        lhs = params.k * (params.k - params.lambda_) - params.t
        rhs = (params.n - params.k - 1) * params.mu
        print(f"INFEASIBLE: k(k-λ)-t={lhs} != (n-k-1)μ={rhs}")
        _write_progress(progress_csv, [_infeasible_row()])
        return

    include_abelian = params.t == params.k
    groups = load_group_tables(params.n, device=device, include_abelian=include_abelian)
    print(f"Groups of order {params.n}: {len(groups)}")

    if not groups:
        print("No groups to search")
        _write_progress(progress_csv, [_no_groups_row()])
        return

    previous_done = _load_done_rows(progress_csv)

    progress_rows: list[dict] = []
    structures = []
    for group in groups:
        if group.library_id in previous_done:
            progress_rows.append(previous_done[group.library_id])
            structures.append(None)
        else:
            structure = analyze_group(group, params, batch_size)
            progress_rows.append({
                "group_lib_id": group.library_id,
                "group_name": group.name,
                "status": "queued",
                "t_valid_count": structure.t_valid_count,
                "num_dsrgs": 0,
                "elapsed_s": 0,
            })
            structures.append(structure)
    _write_progress(progress_csv, progress_rows)

    if previous_done:
        print(f"Resuming: {len(previous_done)} group(s) already done, skipping.")

    for gi, group in enumerate(groups):
        row = progress_rows[gi]
        if row["status"] == "done":
            print(f"\n  {group.name} (lib_id={group.library_id}): already done.")
            continue

        structure = structures[gi]
        assert structure is not None
        tv = structure.t_valid_count
        print(f"\n  {group.name} (lib_id={group.library_id}): {tv:,} t-valid subsets")

        if tv == 0:
            row["status"] = "done"
            row["elapsed_s"] = 0
            _write_progress(progress_csv, progress_rows)
            print("    No t-valid subsets — skipped")
            continue

        row["status"] = "running"
        _write_progress(progress_csv, progress_rows)

        cb = _make_progress_callback(group, noninteractive, structure.total_batches)
        result = search_one_group(
            params,
            group,
            structure,
            batch_size=batch_size,
            device=device,
            on_batch=cb,
        )
        cb.close()  # type: ignore[attr-defined]

        if result.found is not None:
            npz_path = _save_adjacency(result.found, group, params, task_dir)
            print(f"    {result.count} DSRGs found, saved to {npz_path.name}")
        else:
            print("    No DSRGs")

        row["status"] = "done"
        row["num_dsrgs"] = result.count
        row["elapsed_s"] = round(result.elapsed_s, 1)
        _write_progress(progress_csv, progress_rows)

    total_found = sum(int(r["num_dsrgs"]) for r in progress_rows)
    print(f"\nDone. {total_found} total DSRGs across {len(groups)} groups.")


# ---------------------------------------------------------------------------
# Workflow: search a single (group, params) pair
# ---------------------------------------------------------------------------


def run_one_group(
    params: DSRGParams,
    group_id: int,
    output_dir: Path,
    *,
    batch_size: int = 100_000,
    device: str | None = None,
    noninteractive: bool = False,
) -> None:
    """Search a single group (by GAP SmallGroups library_id) for one parameter set.

    Writes one `dsrg_*.npz` under `<output_dir>/<n>_<k>_<t>_<lambda>_<mu>/`
    if hits are found. Intended as a standalone entry point — does not touch
    `progress.csv`.
    """
    device = device or _pick_device()
    task_dir = output_dir / f"{params.n}_{params.k}_{params.t}_{params.lambda_}_{params.mu}"
    task_dir.mkdir(parents=True, exist_ok=True)

    print(params)
    print(f"Group library_id: {group_id}")
    print(f"Output: {task_dir}")
    print(f"Device: {device}")

    if not params.is_feasible():
        print("INFEASIBLE — skipping search.")
        return

    include_abelian = params.t == params.k
    groups = load_group_tables(params.n, device=device, include_abelian=include_abelian)
    group = next((g for g in groups if g.library_id == group_id), None)
    if group is None:
        available = ", ".join(str(g.library_id) for g in groups)
        raise ValueError(
            f"group library_id {group_id} not found for order {params.n}. "
            f"Available: {available}"
        )

    structure = analyze_group(group, params, batch_size)
    print(f"{group.name}: {structure.t_valid_count:,} t-valid subsets")

    if structure.t_valid_count == 0:
        print("No t-valid subsets — nothing to search.")
        return

    cb = _make_progress_callback(group, noninteractive, structure.total_batches)
    result = search_one_group(
        params, group, structure,
        batch_size=batch_size, device=device, on_batch=cb,
    )
    cb.close()  # type: ignore[attr-defined]

    if result.found is not None:
        npz_path = _save_adjacency(result.found, group, params, task_dir)
        print(f"{result.count} DSRGs found, saved to {npz_path.name}")
    else:
        print("No DSRGs found.")


# ---------------------------------------------------------------------------
# Internal: resume / placeholder rows
# ---------------------------------------------------------------------------


def _load_done_rows(progress_csv: Path) -> dict[int, dict]:
    if not progress_csv.exists():
        return {}
    done: dict[int, dict] = {}
    with open(progress_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "done":
                continue
            try:
                done[int(row["group_lib_id"])] = row
            except ValueError:
                pass
    return done


def _infeasible_row() -> dict:
    return {
        "group_lib_id": "",
        "group_name": "",
        "status": "infeasible",
        "t_valid_count": 0,
        "num_dsrgs": 0,
        "elapsed_s": 0,
    }


def _no_groups_row() -> dict:
    return {
        "group_lib_id": "",
        "group_name": "",
        "status": "no_groups",
        "t_valid_count": 0,
        "num_dsrgs": 0,
        "elapsed_s": 0,
    }
