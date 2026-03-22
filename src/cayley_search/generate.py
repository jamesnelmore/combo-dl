#!/usr/bin/env python3
"""Cayley graph DSRG search via GPU-accelerated enumeration.

Loads group multiplication tables from GAP, enumerates connection sets on GPU,
and checks the DSRG condition using batched matrix operations.

Key optimization: instead of enumerating all C(n-1, k) subsets and filtering
for t-validity, we decompose non-identity elements into involutions and
non-self-inverse pairs, then directly construct only subsets with |S ∩ S⁻¹| = t.
This reduces the search space by orders of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
import subprocess

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Group loading (unchanged)
# ---------------------------------------------------------------------------


@dataclass
class GroupTable:
    """Multiplication table and inverse map for a single group."""

    library_id: int
    name: str
    order: int
    identity: int  # 0-indexed position of the identity element
    inv: torch.Tensor  # (n,) int32 — inv[i] = index of elements[i]^{-1}
    table: torch.Tensor  # (n, n) int32 — table[i, j] = index of elements[i] * elements[j]


def load_group_tables(n: int, device: torch.device | str = "cpu") -> list[GroupTable]:
    """Run GAP to get multiplication tables for all nonabelian groups of order *n*."""
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
    lines = output.splitlines()
    i = 0

    def _collect_ints(start: int, stop_marker: str) -> tuple[list[int], int]:
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

            i += 1
            identity = int(lines[i].split()[1])

            i += 1
            first_inv_line = lines[i].strip().split()[1:]
            inv_vals = [int(x) for x in first_inv_line]
            i += 1
            while i < len(lines) and lines[i].strip() != "TABLE_START":
                inv_vals.extend(int(x) for x in lines[i].split())
                i += 1
            inv = torch.tensor(inv_vals, dtype=torch.int32, device=device)

            assert lines[i].strip() == "TABLE_START"
            i += 1

            table_vals, i = _collect_ints(i, "TABLE_END")
            table = torch.tensor(table_vals, dtype=torch.int32, device=device).reshape(n, n)

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


# ---------------------------------------------------------------------------
# Element structure analysis
# ---------------------------------------------------------------------------


def _classify_elements(group: GroupTable) -> tuple[list[int], list[tuple[int, int]]]:
    """Partition non-identity elements into involutions and paired elements.

    Returns:
        involutions: list of element indices where inv[x] == x and x != identity
        pairs: list of (x, x_inv) tuples where x < x_inv and inv[x] != x
               Each unordered pair appears exactly once.
    """
    n = group.order
    inv_cpu = group.inv.cpu().tolist()
    e = group.identity

    involutions: list[int] = []
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()

    for x in range(n):
        if x == e or x in seen:
            continue
        x_inv = inv_cpu[x]
        if x_inv == x:
            involutions.append(x)
        else:
            pairs.append((min(x, x_inv), max(x, x_inv)))
            seen.add(x)
            seen.add(x_inv)

    return involutions, pairs


def _count_t_valid_subsets(
    num_involutions: int,
    num_pairs: int,
    k: int,
    t: int,
) -> int:
    """Count total t-valid subsets without enumerating them.

    A t-valid subset S of size k has |S ∩ S⁻¹| = t.
    Decomposition: pick a involutions, b complete pairs, c = k-t half-pairs.
    Constraint: a + 2b = t, 0 <= a <= num_involutions, 0 <= b <= num_pairs.
    For half-pairs: choose c pairs (not already fully included), pick 1 of 2
    from each → C(num_pairs - b, c) * 2^c.
    """
    c = k - t  # number of half-pair elements
    if c < 0:
        return 0

    total = 0
    for b in range(min(t // 2, num_pairs) + 1):
        a = t - 2 * b
        if a < 0 or a > num_involutions:
            continue
        remaining_pairs = num_pairs - b
        if c > remaining_pairs:
            continue
        total += comb(num_involutions, a) * comb(num_pairs, b) * comb(remaining_pairs, c) * (2**c)

    return total


# ---------------------------------------------------------------------------
# Direct t-valid subset generation (the key optimization)
# ---------------------------------------------------------------------------


def _t_valid_batches(
    involutions: list[int],
    pairs: list[tuple[int, int]],
    k: int,
    t: int,
    batch_size: int,
    device: torch.device | str,
):
    """Yield batches of t-valid k-subsets as (B, k) int32 tensors on device.

    Directly constructs subsets S with |S ∩ S⁻¹| = t by decomposing into:
      - a involutions chosen from the involution set
      - b complete pairs {x, x⁻¹}
      - c = k - t unpaired elements (one from each of c remaining pairs)

    The constraint is a + 2b = t, with a ≥ 0, b ≥ 0, c = k - t.
    """
    c = k - t  # number of half-pair elements needed
    if c < 0:
        return

    num_inv = len(involutions)
    num_pairs = len(pairs)
    pair_arr = pairs  # list of (x, x_inv) tuples

    batch: list[list[int]] = []

    for b in range(min(t // 2, num_pairs) + 1):
        a = t - 2 * b
        if a < 0 or a > num_inv:
            continue
        remaining_pairs_needed = c
        if remaining_pairs_needed > num_pairs - b:
            continue

        # Enumerate: choose a involutions, b full pairs, c half-pairs
        for inv_choice in combinations(range(num_inv), a):
            inv_elems = [involutions[i] for i in inv_choice]

            for pair_full_choice in combinations(range(num_pairs), b):
                pair_full_set = set(pair_full_choice)
                pair_full_elems: list[int] = []
                for pi in pair_full_choice:
                    pair_full_elems.extend(pair_arr[pi])

                # Remaining pair indices (not chosen as full pairs)
                remaining_pair_indices = [pi for pi in range(num_pairs) if pi not in pair_full_set]

                for half_pair_choice in combinations(remaining_pair_indices, c):
                    # For each half-pair, we pick element 0 or 1 → 2^c choices
                    for bits in range(1 << c):
                        half_elems: list[int] = []
                        for j, pi in enumerate(half_pair_choice):
                            side = (bits >> j) & 1
                            half_elems.append(pair_arr[pi][side])

                        subset = inv_elems + pair_full_elems + half_elems
                        batch.append(subset)

                        if len(batch) == batch_size:
                            yield torch.tensor(batch, dtype=torch.int32, device=device)
                            batch = []

    if batch:
        yield torch.tensor(batch, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# DSRG checking (GPU-accelerated)
# ---------------------------------------------------------------------------


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
    B = subsets.shape[0]

    # Precompute left_mult[g, h] = table[inv[g], h] — the element g⁻¹h.
    left_mult = group.table[group.inv]  # (n, n)

    # Build membership mask: (B, n)
    mask = torch.zeros(B, n, dtype=torch.bool, device=device)
    mask.scatter_(1, subsets.long(), True)

    # Full adjacency: A[b, g, h] = mask[b, left_mult[g, h]]
    # mask[:, left_mult] indexes dim 1 of mask with left_mult, giving (B, n, n)
    A = mask[:, left_mult.long()]  # (B, n, n) bool

    # Identity row of A² via bmm: mask is A[e, :], so A²[e, :] = A[e, :] @ A
    A_float = A.float()
    a_sq_e = torch.bmm(mask.unsqueeze(1).float(), A_float).squeeze(1)  # (B, n)

    # Build expected values: mu everywhere, lambda where mask is True, t at identity
    expected = torch.where(mask, lambda_, mu).float()  # (B, n)
    expected[:, e] = t

    # A row is DSRG iff a_sq_e matches expected at every position
    is_dsrg = (a_sq_e == expected).all(dim=1)
    return subsets[is_dsrg]


def build_adjacency(subsets: torch.Tensor, group: GroupTable) -> torch.Tensor:
    """Build adjacency matrices from connection sets.

    Returns:
        (num_sets, n, n) bool adjacency matrices.
    """
    n = group.order
    left_mult = group.table[group.inv]  # (n, n)
    mask = torch.zeros(subsets.shape[0], n, dtype=torch.bool, device=subsets.device)
    mask.scatter_(1, subsets.long(), True)
    return mask[:, left_mult.long()]  # (num_sets, n, n)


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


@torch.no_grad()
def search_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    gen_batch_size: int = 1_000_000,
    dsrg_batch_size: int = 100_000,
    device: torch.device | str = "cpu",
) -> tuple[int, list[tuple[GroupTable, torch.Tensor]]]:
    """Search for DSRGs among Cayley graphs of nonabelian groups of order n.

    Args:
        gen_batch_size: Number of t-valid subsets to generate per batch (cheap).
        dsrg_batch_size: Number of t-valid subsets to accumulate before running
            the DSRG check (expensive — builds batch×n×n adjacency matrices).

    Returns:
        (num_groups, results) where num_groups is the number of nonabelian groups
        checked and results is a list of (group, subsets) pairs.
    """
    total_subsets = comb(n - 1, k)

    print(f"DSRG({n}, {k}, {t}, {lambda_}, {mu})")
    print(f"Total {k}-subsets of {n - 1} non-identity elements: {total_subsets:,}")

    groups = load_group_tables(n, device=device)
    print(f"Nonabelian groups of order {n}: {len(groups)}")

    results: list[tuple[GroupTable, torch.Tensor]] = []

    for group in groups:
        involutions, pairs = _classify_elements(group)
        num_inv = len(involutions)
        num_pairs = len(pairs)

        t_valid_count = _count_t_valid_subsets(num_inv, num_pairs, k, t)
        t_valid_pct = 100.0 * t_valid_count / total_subsets if total_subsets > 0 else 0.0

        print(f"\n  {group.name} (lib_id={group.library_id}):")
        print(f"    Involutions: {num_inv}, Non-self-inverse pairs: {num_pairs}")
        print(f"    T-valid subsets: {t_valid_count:,} ({t_valid_pct:.4f}% of all subsets)")

        if t_valid_count == 0:
            print(f"    → No t-valid subsets exist — skipping DSRG check")
            continue

        found_subsets: list[torch.Tensor] = []
        pending: list[torch.Tensor] = []
        pending_count = 0
        total_generated = 0

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

        total_batches = (t_valid_count + gen_batch_size - 1) // gen_batch_size

        pbar = tqdm(
            _t_valid_batches(involutions, pairs, k, t, gen_batch_size, device),
            total=total_batches,
            desc=f"    {group.name}",
            unit="batch",
        )
        for batch in pbar:
            total_generated += batch.shape[0]

            pending.append(batch)
            pending_count += batch.shape[0]

            if pending_count >= dsrg_batch_size:
                _flush_pending()

            pbar.set_postfix(
                generated=total_generated,
                found=sum(s.shape[0] for s in found_subsets),
            )

        # Flush remaining
        _flush_pending()

        if found_subsets:
            all_found = torch.cat(found_subsets, dim=0)
            results.append((group, all_found))
            print(f"    → {all_found.shape[0]} DSRG(s) found in {group.name}")
        else:
            print(f"    → No DSRGs in {group.name}")

    return len(groups), results


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Result of searching a single parameter set."""

    num_groups: int
    hits: list[tuple[GroupTable, torch.Tensor]]


def _run_single(
    n, k, t, lambda_, mu, gen_batch_size, dsrg_batch_size, device,
    output_dir: Path | None = None,
) -> SearchResult:
    """Run search for a single parameter set and save results."""
    num_groups, hits = search_dsrg(
        n,
        k,
        t,
        lambda_,
        mu,
        gen_batch_size=gen_batch_size,
        dsrg_batch_size=dsrg_batch_size,
        device=device,
    )
    for group, subsets in hits:
        count = subsets.shape[0]
        print(f"\n{group.name} (lib_id={group.library_id}): {count} DSRG connection set(s)")
        adj = build_adjacency(subsets, group).cpu().numpy().astype(np.uint8)
        filename = f"dsrg_{n}_{k}_{t}_{lambda_}_{mu}_g{group.library_id}.npz"
        if output_dir is not None:
            filename = str(output_dir / filename)
        np.savez_compressed(filename, adjacency=adj)
        print(f"  Saved {count} adjacency matrices ({adj.shape}) to {filename}")
    return SearchResult(num_groups=num_groups, hits=hits)


if __name__ == "__main__":
    import sys

    import pandas as pd

    gen_batch_size = 1_000_000
    dsrg_batch_size = 100_000
    params_file = None
    output_dir = None
    positional: list[str] = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--gen-batch-size":
            gen_batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--dsrg-batch-size":
            dsrg_batch_size = int(sys.argv[i + 1])
            i += 2
        # Keep old flag names as aliases for backwards compatibility
        elif arg == "--full-batch-size":
            gen_batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--t-reduced-size":
            dsrg_batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg in ("--csv", "--params"):
            params_file = sys.argv[i + 1]
            i += 2
        elif arg == "--output-dir":
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1

    if params_file is None and len(positional) < 5:
        print("Usage: generate.py n k t lambda mu [--gen-batch-size N] [--dsrg-batch-size N]")
        print("       generate.py --params params.csv|.xlsx [--output-dir DIR] [--gen-batch-size N] [--dsrg-batch-size N]")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    out_path = Path(output_dir) if output_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    if params_file is not None:
        pf = Path(params_file)
        if pf.suffix in (".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"):
            params_df = pd.read_excel(pf)
        else:
            params_df = pd.read_csv(pf)

        required = {"n", "k", "t", "lambda", "mu"}
        missing = required - set(params_df.columns)
        if missing:
            print(f"Error: missing required columns: {missing}")
            print(f"Found columns: {list(params_df.columns)}")
            sys.exit(1)

        results_log: list[dict] = []
        results_csv = (out_path / "results.csv") if out_path else Path("results.csv")
        fieldnames = [
            "row", "n", "k", "t", "lambda", "mu",
            "num_groups", "group_lib_id", "group_name",
            "num_dsrgs", "total_dsrgs", "file",
        ]

        for row_idx, row in params_df.iterrows():
            n = int(row["n"])
            k = int(row["k"])
            t = int(row["t"])
            lambda_ = int(row["lambda"])
            mu = int(row["mu"])

            print(f"\n{'='*60}")
            print(f"Row {row_idx} : DSRG({n}, {k}, {t}, {lambda_}, {mu})")
            print(f"{'='*60}")

            result = _run_single(
                n, k, t, lambda_, mu,
                gen_batch_size, dsrg_batch_size, device,
                output_dir=out_path,
            )

            total_dsrgs = sum(s.shape[0] for _, s in result.hits)

            if result.hits:
                for group, subsets in result.hits:
                    results_log.append({
                        "row": row_idx,
                        "n": n, "k": k, "t": t,
                        "lambda": lambda_, "mu": mu,
                        "num_groups": result.num_groups,
                        "group_lib_id": group.library_id,
                        "group_name": group.name,
                        "num_dsrgs": subsets.shape[0],
                        "total_dsrgs": total_dsrgs,
                        "file": f"dsrg_{n}_{k}_{t}_{lambda_}_{mu}_g{group.library_id}.npz",
                    })
            else:
                results_log.append({
                    "row": row_idx,
                    "n": n, "k": k, "t": t,
                    "lambda": lambda_, "mu": mu,
                    "num_groups": result.num_groups,
                    "group_lib_id": "", "group_name": "",
                    "num_dsrgs": 0, "total_dsrgs": 0, "file": "",
                })

            pd.DataFrame(results_log).to_csv(results_csv, index=False)

        hits = sum(1 for r in results_log if r["total_dsrgs"])
        print(f"\n{'='*60}")
        print(f"Done. {hits} parameter set(s) with results out of {len(params_df)} checked.")
        print(f"Results summary written to {results_csv}")
    else:
        n, k, t, lambda_, mu = (
            int(positional[0]),
            int(positional[1]),
            int(positional[2]),
            int(positional[3]),
            int(positional[4]),
        )
        _run_single(n, k, t, lambda_, mu, gen_batch_size, dsrg_batch_size, device, output_dir=out_path)
