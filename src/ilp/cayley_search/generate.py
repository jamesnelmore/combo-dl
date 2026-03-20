#!/usr/bin/env python3
"""Cayley graph DSRG search via GPU-accelerated enumeration.

Loads group multiplication tables from GAP, enumerates connection sets on GPU,
and checks the DSRG condition using batched matrix operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import subprocess

import numpy as np

import torch
from tqdm import tqdm


@dataclass
class GroupTable:
    """Multiplication table and inverse map for a single group."""

    library_id: int
    name: str
    order: int
    identity: int  # 0-indexed position of the identity element
    inv: torch.Tensor  # (n,) int64 — inv[i] = index of elements[i]^{-1}
    table: torch.Tensor  # (n, n) int64 — table[i, j] = index of elements[i] * elements[j]


def load_group_tables(n: int, device: torch.device | str = "cpu") -> list[GroupTable]:
    """Run GAP to get multiplication tables for all nonabelian groups of order *n*.

    Returns a list of GroupTable, one per nonabelian group, with tensors on *device*.
    """
    script = Path(__file__).with_name("group_tables.g")
    gap_input = f"n := {n};;\n" + script.read_text()

    proc = subprocess.run(
        ["gap", "-q"],
        input=gap_input,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GAP failed (exit {proc.returncode}):\n{proc.stderr}")

    return _parse_gap_output(proc.stdout, n, device)


def _parse_gap_output(output: str, n: int, device: torch.device | str) -> list[GroupTable]:
    """Parse GAP output, handling line-wrapping for large n."""
    groups: list[GroupTable] = []
    # Split into tokens aware of marker lines.  GAP wraps long rows across
    # multiple lines, so we collect all numbers between markers.
    lines = output.splitlines()
    i = 0

    def _collect_ints(start: int, stop_marker: str) -> tuple[list[int], int]:
        """Collect all integers from lines[start..] until stop_marker is seen."""
        vals: list[int] = []
        j = start
        while j < len(lines):
            stripped = lines[j].strip()
            if stripped == stop_marker:
                return vals, j
            for tok in stripped.split():
                vals.append(int(tok))
            j += 1
        raise ValueError(f"Expected {stop_marker!r} but hit end of output")

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("GROUP_START"):
            parts = line.split(maxsplit=2)
            lib_id = int(parts[1])
            name = parts[2] if len(parts) > 2 else "?"

            # IDENTITY — always a single short line
            i += 1
            identity = int(lines[i].split()[1])

            # INV — may wrap across multiple lines, ends at TABLE_START
            i += 1
            # First line starts with "INV", strip that token
            first_inv_line = lines[i].strip().split()[1:]  # skip "INV"
            inv_vals = [int(x) for x in first_inv_line]
            i += 1
            # Keep reading until TABLE_START
            while i < len(lines) and lines[i].strip() != "TABLE_START":
                inv_vals.extend(int(x) for x in lines[i].split())
                i += 1
            inv = torch.tensor(inv_vals, dtype=torch.int32, device=device)

            # TABLE_START — i now points here
            assert lines[i].strip() == "TABLE_START"
            i += 1

            # Collect all ints until TABLE_END
            table_vals, i = _collect_ints(i, "TABLE_END")
            table = torch.tensor(table_vals, dtype=torch.int32, device=device).reshape(n, n)

            # GROUP_END
            i += 1
            assert lines[i].strip() == "GROUP_END"

            groups.append(
                GroupTable(
                    library_id=lib_id,
                    name=name,
                    order=n,
                    identity=identity,
                    inv=inv,
                    table=table,
                )
            )

        elif line == "ALL_DONE":
            break

        i += 1

    return groups


def _subset_batches(non_identity: list[int], k: int, full_batch_size: int, device: torch.device | str):
    """Yield batches of k-subsets as (batch, k) int64 tensors."""
    batch: list[tuple[int, ...]] = []
    for combo in combinations(non_identity, k):
        batch.append(combo)
        if len(batch) == full_batch_size:
            yield torch.tensor(batch, dtype=torch.int32, device=device)
            batch = []
    if batch:
        yield torch.tensor(batch, dtype=torch.int32, device=device)


def t_filter(subsets: torch.Tensor, inv: torch.Tensor, n: int, t: int) -> torch.Tensor:
    """Return the subset of rows where |S ∩ S⁻¹| == t.

    Args:
        subsets: (batch, k) element indices.
        inv: (n,) inverse map.
        n: group order.
        t: required number of self-inverse elements in S.

    Returns:
        (num_valid, k) tensor of valid subsets.
    """
    full_batch_size, k = subsets.shape
    # Build boolean membership mask: (batch, n)
    mask = torch.zeros(full_batch_size, n, dtype=torch.bool, device=subsets.device)
    mask.scatter_(1, subsets, True)
    # For each element in each subset, check if its inverse is also in the subset
    inv_of_subsets = inv[subsets]  # (batch, k)
    t_counts = mask.gather(1, inv_of_subsets).sum(dim=1)  # (batch,)
    return subsets[t_counts == t]


def check_dsrg(
    subsets: torch.Tensor,
    group: GroupTable,
    t: int,
    lambda_: int,
    mu: int,
) -> torch.Tensor:
    """Check which subsets produce a DSRG, using the identity-row shortcut.

    Returns:
        (num_valid, k) tensor of subsets that satisfy the DSRG equation.
    """
    if subsets.shape[0] == 0:
        return subsets

    n = group.order
    e = group.identity
    device = subsets.device
    full_batch_size = subsets.shape[0]

    # Precompute left_mult[g, h] = table[inv[g], h] — the element g⁻¹h.
    left_mult = group.table[group.inv]  # (n, n)

    # Build membership mask: (batch, n)
    mask = torch.zeros(full_batch_size, n, dtype=torch.bool, device=device)
    mask.scatter_(1, subsets, True)

    # Full adjacency: A[b, g, h] = mask[b, left_mult[g, h]]
    # mask[:, left_mult] indexes dim 1 of mask with left_mult, giving (batch, n, n)
    A = mask[:, left_mult]  # (batch, n, n) bool

    # Identity row of A² via bmm: mask is A[e, :], so A²[e, :] = A[e, :] @ A
    A_float = A.float()
    a_sq_e = torch.bmm(mask.unsqueeze(1).float(), A_float).squeeze(1)  # (batch, n)

    # Build expected values: mu everywhere, lambda where mask is True, t at identity
    expected = torch.where(mask, lambda_, mu).float()  # (batch, n)
    expected[:, e] = t

    # A row is DSRG iff a_sq_e matches expected at every position
    is_dsrg = (a_sq_e == expected).all(dim=1)
    return subsets[is_dsrg]


def build_adjacency(subsets: torch.Tensor, group: GroupTable) -> torch.Tensor:
    """Build adjacency matrices from connection sets.

    Args:
        subsets: (num_sets, k) element indices.
        group: GroupTable for the group.

    Returns:
        (num_sets, n, n) bool adjacency matrices.
    """
    n = group.order
    left_mult = group.table[group.inv]  # (n, n)
    mask = torch.zeros(subsets.shape[0], n, dtype=torch.bool, device=subsets.device)
    mask.scatter_(1, subsets, True)
    return mask[:, left_mult]  # (num_sets, n, n)


@torch.no_grad()
def search_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    full_batch_size: int = 10_000_000,
    t_reduced_size: int = 100_000,
    device: torch.device | str = "cpu",
) -> list[tuple[GroupTable, torch.Tensor]]:
    """Search for DSRGs among Cayley graphs of nonabelian groups of order n.

    Args:
        full_batch_size: Number of subsets to generate and t-filter at once (cheap).
        t_reduced_size: Number of t-valid subsets to accumulate before running
            the DSRG check (expensive — builds batch×n×n adjacency matrices).

    Returns:
        List of (group, subsets) pairs where subsets is a (num_found, k) tensor
        of connection sets that yield a DSRG.
    """
    from math import comb

    total_subsets = comb(n - 1, k)

    print(f"DSRG({n}, {k}, {t}, {lambda_}, {mu})")
    print(f"Total {k}-subsets of {n - 1} non-identity elements: {total_subsets:,}")

    groups = load_group_tables(n, device=device)
    print(f"Nonabelian groups of order {n}: {len(groups)}")

    results: list[tuple[GroupTable, torch.Tensor]] = []

    for group in groups:
        non_id = [i for i in range(n) if i != group.identity]

        found_subsets: list[torch.Tensor] = []
        pending: list[torch.Tensor] = []  # t-valid subsets awaiting DSRG check
        pending_count = 0
        total_checked = 0
        total_t_valid = 0

        def _flush_pending() -> None:
            nonlocal pending, pending_count
            if not pending:
                return
            check_batch = torch.cat(pending, dim=0)
            pending = []
            pending_count = 0
            dsrg = check_dsrg(check_batch, group, t, lambda_, mu)
            if dsrg.shape[0] > 0:
                found_subsets.append(dsrg)

        pbar = tqdm(
            _subset_batches(non_id, k, full_batch_size, device),
            total=(total_subsets + full_batch_size - 1) // full_batch_size,
            desc=f"  {group.name}",
            unit="batch",
        )
        for batch in pbar:
            total_checked += batch.shape[0]

            # T-filter (cheap)
            valid = t_filter(batch, group.inv, n, t)
            total_t_valid += valid.shape[0]

            if valid.shape[0] > 0:
                pending.append(valid)
                pending_count += valid.shape[0]

            # Flush to DSRG check when we've accumulated enough
            if pending_count >= t_reduced_size:
                _flush_pending()

            pbar.set_postfix(
                t_valid=total_t_valid,
                found=sum(s.shape[0] for s in found_subsets),
            )

        # Flush remaining
        _flush_pending()

        if found_subsets:
            all_found = torch.cat(found_subsets, dim=0)
            results.append((group, all_found))
            print(f"  → {all_found.shape[0]} DSRG(s) found in {group.name}")
        else:
            print(f"  → No DSRGs in {group.name}")

    return results


def _run_single(n, k, t, lambda_, mu, full_batch_size, t_reduced_size, device):
    results = search_dsrg(
        n, k, t, lambda_, mu,
        full_batch_size=full_batch_size,
        t_reduced_size=t_reduced_size,
        device=device,
    )
    for group, subsets in results:
        count = subsets.shape[0]
        print(f"\n{group.name} (lib_id={group.library_id}): {count} DSRG connection set(s)")
        adj = build_adjacency(subsets, group).cpu().numpy().astype(np.uint8)
        filename = f"dsrg_{n}_{k}_{t}_{lambda_}_{mu}_g{group.library_id}.npz"
        np.savez_compressed(filename, adjacency=adj)
        print(f"  Saved {count} adjacency matrices ({adj.shape}) to {filename}")


if __name__ == "__main__":
    import csv
    import sys

    full_batch_size = 1_000_000
    t_reduced_size = 100_000
    csv_file = None
    positional: list[str] = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--full-batch-size":
            full_batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--t-reduced-size":
            t_reduced_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--csv":
            csv_file = sys.argv[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1

    if csv_file is None and len(positional) < 5:
        print("Usage: generate.py n k t lambda mu [--full-batch-size N] [--t-reduced-size N]")
        print("       generate.py --csv params.csv [--full-batch-size N] [--t-reduced-size N]")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    if csv_file is not None:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = int(row["n"])
                k = int(row["k"])
                t = int(row["t"])
                lambda_ = int(row["lambda"])
                mu = int(row["mu"])
                _run_single(n, k, t, lambda_, mu, full_batch_size, t_reduced_size, device)
    else:
        n, k, t, lambda_, mu = (
            int(positional[0]),
            int(positional[1]),
            int(positional[2]),
            int(positional[3]),
            int(positional[4]),
        )
        _run_single(n, k, t, lambda_, mu, full_batch_size, t_reduced_size, device)
