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


def load_group_tables(
    n: int,
    device: torch.device | str = "cpu",
    include_abelian: bool = False,
) -> list[GroupTable]:
    """Run GAP to get multiplication tables for groups of order *n*.

    When *include_abelian* is False (default), only nonabelian groups are
    loaded (directed DSRG search).  When True, all groups are loaded
    (undirected / t=k search where S = S⁻¹).
    """
    script = Path(__file__).with_name("group_tables.g")
    abelian_flag = "true" if include_abelian else "false"
    gap_input = f"n := {n};;\ninclude_abelian := {abelian_flag};;\n" + script.read_text()

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


def _unrank_combinations(indices: torch.Tensor, n: int, c: int) -> torch.Tensor:
    """Decode flat indices to combinations using the combinatorial number system.

    Given a tensor of indices in [0, C(n, c)), returns the corresponding
    c-combinations of {0, ..., n-1} in sorted order. Fully vectorized.

    Args:
        indices: (B,) int64 tensor of combination indices.
        n: Size of the set to choose from.
        c: Number of elements to choose.

    Returns:
        (B, c) int64 tensor where each row is a sorted combination.
    """
    B = indices.shape[0]
    device = indices.device
    result = torch.empty(B, c, dtype=torch.int64, device=device)
    remaining = indices.clone()

    # Precompute C(m, j) for all m in [0, n) and j in [1, c].
    # binom_table[m, j-1] = C(m, j)
    binom_table = torch.zeros(n, c, dtype=torch.int64, device=device)
    for m in range(n):
        for j in range(1, c + 1):
            binom_table[m, j - 1] = comb(m, j)

    for j in range(c, 0, -1):
        # Find largest m where C(m, j) <= remaining, with m < n
        # binom_col[m] = C(m, j) for m = 0..n-1
        binom_col = binom_table[:, j - 1].contiguous()  # (n,)

        # For each index in remaining, find the largest valid m.
        # binom_col is non-decreasing, so we can use searchsorted.
        # searchsorted(a, v, side='right') gives first index where a > v,
        # so the largest m with C(m, j) <= remaining is searchsorted(..., 'right') - 1.
        m = torch.searchsorted(binom_col, remaining, side="right") - 1
        m = m.clamp(min=0)

        result[:, c - j] = m
        remaining -= binom_col[m]

    return result


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

    The inner loops (half-pair selection × 2^c bit patterns) are fully
    generated on the target device using the combinatorial number system,
    avoiding Python-level iteration over combinations.
    """
    c = k - t  # number of half-pair elements needed
    if c < 0:
        return

    num_inv = len(involutions)
    num_pairs = len(pairs)
    pair_tensor = torch.tensor(pairs, dtype=torch.int32, device=device)  # (num_pairs, 2)
    inv_tensor = torch.tensor(involutions, dtype=torch.int32, device=device)

    # Precompute bit patterns for half-pair side selection: (2^c, c)
    if c > 0:
        two_to_c = 1 << c
        bits_range = torch.arange(two_to_c, device=device, dtype=torch.int32)
        shifts = torch.arange(c, device=device, dtype=torch.int32)
        bit_patterns = (bits_range.unsqueeze(1) >> shifts.unsqueeze(0)) & 1  # (2^c, c)
    else:
        two_to_c = 1
        bit_patterns = None

    # How many half-pair combos to generate per GPU chunk.
    # Each expands to 2^c subsets.
    hp_chunk_size = max(1, batch_size // two_to_c) if c > 0 else batch_size

    for b in range(min(t // 2, num_pairs) + 1):
        a = t - 2 * b
        if a < 0 or a > num_inv:
            continue
        if c > num_pairs - b:
            continue

        for inv_choice in combinations(range(num_inv), a):
            inv_elems = inv_tensor[list(inv_choice)] if a > 0 else torch.empty(0, dtype=torch.int32, device=device)

            for pair_full_choice in combinations(range(num_pairs), b):
                pair_full_set = set(pair_full_choice)
                if b > 0:
                    pair_full_elems = pair_tensor[list(pair_full_choice)].reshape(-1)
                else:
                    pair_full_elems = torch.empty(0, dtype=torch.int32, device=device)

                prefix = torch.cat([inv_elems, pair_full_elems])  # (t,)

                if c == 0:
                    yield prefix.unsqueeze(0)
                    continue

                remaining_pair_indices = [pi for pi in range(num_pairs) if pi not in pair_full_set]
                remaining_pairs = pair_tensor[remaining_pair_indices]  # (R, 2)
                R = len(remaining_pair_indices)
                total_hp_combos = comb(R, c)

                # Generate half-pair combos on GPU in chunks via unranking
                for chunk_start in range(0, total_hp_combos, hp_chunk_size):
                    chunk_end = min(chunk_start + hp_chunk_size, total_hp_combos)
                    chunk_len = chunk_end - chunk_start

                    # Unrank indices to combinations on device
                    flat_indices = torch.arange(
                        chunk_start, chunk_end, dtype=torch.int64, device=device,
                    )
                    hp_combos = _unrank_combinations(flat_indices, R, c)  # (chunk_len, c)

                    # Gather pairs: (chunk_len, c, 2)
                    selected_pairs = remaining_pairs[hp_combos]

                    # Expand bit patterns: (chunk_len, 2^c, c)
                    expanded = selected_pairs.unsqueeze(1).expand(chunk_len, two_to_c, c, 2)
                    bp = bit_patterns.unsqueeze(0).unsqueeze(-1).expand(chunk_len, two_to_c, c, 1).long()
                    half_elems = expanded.gather(-1, bp).squeeze(-1)  # (chunk_len, 2^c, c)

                    # Reshape to (chunk_len * 2^c, c)
                    total = chunk_len * two_to_c
                    half_elems = half_elems.reshape(total, c)

                    # Prepend prefix
                    prefix_block = prefix.unsqueeze(0).expand(total, -1)
                    subsets = torch.cat([prefix_block, half_elems], dim=1)

                    # Yield in batch_size slices
                    for start in range(0, total, batch_size):
                        yield subsets[start : start + batch_size]


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
    batch_size: int = 100_000,
    device: torch.device | str = "cpu",
) -> tuple[int, list[tuple[GroupTable, torch.Tensor]]]:
    """Search for DSRGs among Cayley graphs of nonabelian groups of order n.

    Args:
        batch_size: Number of t-valid subsets per DSRG check batch.
            Controls GPU memory usage (builds batch×n×n adjacency matrices).

    Returns:
        (num_groups, results) where num_groups is the number of nonabelian groups
        checked and results is a list of (group, subsets) pairs.
    """
    total_subsets = comb(n - 1, k)

    print(f"DSRG({n}, {k}, {t}, {lambda_}, {mu})")
    print(f"Total {k}-subsets of {n - 1} non-identity elements: {total_subsets:,}")

    groups = load_group_tables(n, device=device, include_abelian=(t == k))
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
        total_generated = 0

        total_batches = (t_valid_count + batch_size - 1) // batch_size

        pbar = tqdm(
            _t_valid_batches(involutions, pairs, k, t, batch_size, device),
            total=total_batches,
            desc=f"    {group.name}",
            unit="batch",
        )
        for batch in pbar:
            total_generated += batch.shape[0]
            dsrg = check_dsrg(batch, group, t, lambda_, mu)
            if dsrg.shape[0] > 0:
                found_subsets.append(dsrg)

            pbar.set_postfix(
                generated=total_generated,
                found=sum(s.shape[0] for s in found_subsets),
            )

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
    n, k, t, lambda_, mu, batch_size, device,
    output_dir: Path | None = None,
) -> SearchResult:
    """Run search for a single parameter set and save results."""
    num_groups, hits = search_dsrg(
        n,
        k,
        t,
        lambda_,
        mu,
        batch_size=batch_size,
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

    batch_size = 100_000
    params_file = None
    output_dir = None
    positional: list[str] = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--batch-size":
            batch_size = int(sys.argv[i + 1])
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
        print("Usage: generate.py n k t lambda mu [--batch-size N]")
        print("       generate.py --params params.csv|.xlsx [--output-dir DIR] [--batch-size N]")
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
                batch_size, device, output_dir=out_path,
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
        _run_single(n, k, t, lambda_, mu, batch_size, device, output_dir=out_path)
