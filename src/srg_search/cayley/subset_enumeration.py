"""Cayley-graph DSRG search: pure-compute core.

Loads group multiplication tables from GAP, enumerates t-valid connection sets
on GPU, and checks the DSRG condition using batched matrix operations.

Key optimization: instead of enumerating all C(n-1, k) subsets and filtering
for t-validity, we decompose non-identity elements into involutions and
non-self-inverse pairs, then directly construct only subsets with |S ∩ S⁻¹| = t.

This module is I/O-free apart from the GAP subprocess in `load_group_tables`.
File writing, progress bars, and CLI parsing live in `orchestration.py` and
`cli.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
import subprocess
import time
from typing import override

from jaxtyping import Bool, Int
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Parameter and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSRGParams:
    """The five parameters of a DSRG(n, k, t, λ, μ) search."""

    n: int
    k: int
    t: int
    lambda_: int
    mu: int

    def is_feasible(self) -> bool:
        """The standard DSRG necessary condition k(k - λ) - t = (n - k - 1) μ."""
        return self.k * (self.k - self.lambda_) - self.t == (self.n - self.k - 1) * self.mu

    @override
    def __str__(self) -> str:
        return f"DSRG({self.n}, {self.k}, {self.t}, {self.lambda_}, {self.mu})"


@dataclass
class GroupTable:
    """Multiplication table and inverse map for a single group."""

    library_id: int
    name: str
    order: int
    identity: int  # 0-indexed position of the identity element
    inv: Int[Tensor, "n"]  # inv[i] = index of elements[i]^{-1}
    table: Int[Tensor, "n n"]  # table[i, j] = index of elements[i] * elements[j]


@dataclass
class GroupStructure:
    """Cached structural info for one (group, params) pair.

    Computed once up-front so orchestrators can show progress estimates
    without re-running the classification during the search loop.
    """

    involutions: list[int]
    pairs: list[tuple[int, int]]
    t_valid_count: int
    total_batches: int


@dataclass
class BatchUpdate:
    """Per-batch progress signal passed to search callbacks."""

    batch_idx: int  # 0-based
    total_batches: int
    subsets_checked: int
    found_so_far: int


@dataclass
class GroupSearchResult:
    """Result of searching one group for one parameter set."""

    structure: GroupStructure
    found: Int[Tensor, "count k"] | None
    elapsed_s: float

    @property
    def count(self) -> int:
        return 0 if self.found is None else int(self.found.shape[0])


# ---------------------------------------------------------------------------
# Group loading (GAP subprocess)
# ---------------------------------------------------------------------------


def load_group_tables(
    n: int,
    device: torch.device | str = "cpu",
    include_abelian: bool = False,
) -> list[GroupTable]:
    """Run GAP to get multiplication tables for groups of order *n*.

    When *include_abelian* is False (default), only nonabelian groups are
    loaded (directed DSRG search). When True, all groups are loaded
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
# Group structure analysis
# ---------------------------------------------------------------------------


def analyze_group(
    group: GroupTable, params: DSRGParams, batch_size: int
) -> GroupStructure:
    """Classify a group's non-identity elements and pre-count the search space.

    Returns the structure needed both to display progress estimates and to
    feed `search_one_group` without redoing the classification.
    """
    involutions, pairs = _classify_elements(group)
    t_valid = _count_t_valid_subsets(len(involutions), len(pairs), params.k, params.t)
    total_batches = _count_batches(
        len(involutions), len(pairs), params.k, params.t, batch_size
    )
    return GroupStructure(
        involutions=involutions,
        pairs=pairs,
        t_valid_count=t_valid,
        total_batches=total_batches,
    )


def _classify_elements(group: GroupTable) -> tuple[list[int], list[tuple[int, int]]]:
    """Partition non-identity elements into involutions and paired elements.

    Returns:
        involutions: indices x with inv[x] == x and x != identity.
        pairs: (x, x_inv) tuples with x < x_inv and inv[x] != x. Each unordered
            pair appears exactly once.
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


def _count_t_valid_subsets(num_involutions: int, num_pairs: int, k: int, t: int) -> int:
    """Count t-valid k-subsets without enumerating them.

    Decomposition: pick a involutions, b complete pairs, c = k-t half-pairs.
    Constraint: a + 2b = t. For half-pairs: choose c pairs (not already fully
    included), pick 1 of 2 from each → C(num_pairs - b, c) * 2^c.
    """
    c = k - t
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
        total += (
            comb(num_involutions, a)
            * comb(num_pairs, b)
            * comb(remaining_pairs, c)
            * (2**c)
        )

    return total


def _count_batches(
    num_involutions: int, num_pairs: int, k: int, t: int, batch_size: int
) -> int:
    """Count the exact number of batches that `_t_valid_batches` will yield."""
    c = k - t
    if c < 0:
        return 0

    two_to_c = 1 << c if c > 0 else 1
    hp_chunk_size = max(1, batch_size // two_to_c) if c > 0 else batch_size

    total = 0
    for b in range(min(t // 2, num_pairs) + 1):
        a = t - 2 * b
        if a < 0 or a > num_involutions:
            continue
        R = num_pairs - b
        if c > R:
            continue
        num_triples = comb(num_involutions, a) * comb(num_pairs, b)
        if c == 0:
            total += num_triples
        else:
            chunks_per_triple = (comb(R, c) + hp_chunk_size - 1) // hp_chunk_size
            total += num_triples * chunks_per_triple

    return total


# ---------------------------------------------------------------------------
# Direct t-valid subset generation
# ---------------------------------------------------------------------------


def _unrank_combinations(
    indices: Int[Tensor, "B"], n: int, c: int
) -> Int[Tensor, "B c"]:
    """Decode flat indices to combinations using the combinatorial number system.

    Given indices in [0, C(n, c)), returns the corresponding c-combinations
    of {0, ..., n-1} in sorted order. Fully vectorized.
    """
    B = indices.shape[0]
    device = indices.device
    result = torch.empty(B, c, dtype=torch.int64, device=device)
    remaining = indices.clone()

    # binom_table[m, j-1] = C(m, j)
    binom_table = torch.zeros(n, c, dtype=torch.int64, device=device)
    for m in range(n):
        for j in range(1, c + 1):
            binom_table[m, j - 1] = comb(m, j)

    for j in range(c, 0, -1):
        binom_col = binom_table[:, j - 1].contiguous()
        m = torch.searchsorted(binom_col, remaining, side="right") - 1
        m = m.clamp(min=0)
        result[:, c - j] = m
        remaining -= binom_col[m]

    return result


def _t_valid_batches(
    structure: GroupStructure,
    k: int,
    t: int,
    batch_size: int,
    device: torch.device | str,
) -> Iterator[Int[Tensor, "B k"]]:
    """Yield batches of t-valid k-subsets as (B, k) int32 tensors on device.

    See module docstring for the decomposition strategy.
    """
    c = k - t
    if c < 0:
        return

    involutions = structure.involutions
    pairs = structure.pairs
    num_inv = len(involutions)
    num_pairs = len(pairs)
    pair_tensor = torch.tensor(pairs, dtype=torch.int32, device=device)
    inv_tensor = torch.tensor(involutions, dtype=torch.int32, device=device)

    if c > 0:
        two_to_c = 1 << c
        bits_range = torch.arange(two_to_c, device=device, dtype=torch.int32)
        shifts = torch.arange(c, device=device, dtype=torch.int32)
        bit_patterns = (bits_range.unsqueeze(1) >> shifts.unsqueeze(0)) & 1
    else:
        two_to_c = 1
        bit_patterns = None

    hp_chunk_size = max(1, batch_size // two_to_c) if c > 0 else batch_size

    for b in range(min(t // 2, num_pairs) + 1):
        a = t - 2 * b
        if a < 0 or a > num_inv:
            continue
        if c > num_pairs - b:
            continue

        for inv_choice in combinations(range(num_inv), a):
            if a > 0:
                inv_elems = inv_tensor[list(inv_choice)]
            else:
                inv_elems = torch.empty(0, dtype=torch.int32, device=device)

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

                assert bit_patterns is not None  # c > 0 branch
                remaining_pair_indices = [
                    pi for pi in range(num_pairs) if pi not in pair_full_set
                ]
                remaining_pairs = pair_tensor[remaining_pair_indices]
                R = len(remaining_pair_indices)
                total_hp_combos = comb(R, c)

                for chunk_start in range(0, total_hp_combos, hp_chunk_size):
                    chunk_end = min(chunk_start + hp_chunk_size, total_hp_combos)
                    chunk_len = chunk_end - chunk_start

                    flat_indices = torch.arange(
                        chunk_start, chunk_end, dtype=torch.int64, device=device
                    )
                    hp_combos = _unrank_combinations(flat_indices, R, c)
                    selected_pairs = remaining_pairs[hp_combos]

                    expanded = selected_pairs.unsqueeze(1).expand(
                        chunk_len, two_to_c, c, 2
                    )
                    bp = (
                        bit_patterns.unsqueeze(0)
                        .unsqueeze(-1)
                        .expand(chunk_len, two_to_c, c, 1)
                        .long()
                    )
                    half_elems = expanded.gather(-1, bp).squeeze(-1)

                    total = chunk_len * two_to_c
                    half_elems = half_elems.reshape(total, c)
                    prefix_block = prefix.unsqueeze(0).expand(total, -1)
                    subsets = torch.cat([prefix_block, half_elems], dim=1)

                    for start in range(0, total, batch_size):
                        yield subsets[start : start + batch_size]


# ---------------------------------------------------------------------------
# DSRG checking (GPU-accelerated)
# ---------------------------------------------------------------------------


def check_dsrg(
    subsets: Int[Tensor, "B k"],
    group: GroupTable,
    t: int,
    lambda_: int,
    mu: int,
) -> Int[Tensor, "valid k"]:
    """Return the subset of *subsets* that produce a DSRG.

    Uses the identity-row shortcut: only verifies the e-row of A² against the
    expected pattern, which suffices when the connection set is closed under
    conjugation properties of the Cayley graph.
    """
    if subsets.shape[0] == 0:
        return subsets

    n = group.order
    e = group.identity
    device = subsets.device
    B = subsets.shape[0]

    # left_mult[g, h] = table[inv[g], h] = g⁻¹h
    left_mult = group.table[group.inv]

    mask = torch.zeros(B, n, dtype=torch.bool, device=device)
    mask.scatter_(1, subsets.long(), True)

    A = mask[:, left_mult.long()]  # (B, n, n) bool

    # A²[e, :] = A[e, :] @ A, and A[e, :] = mask (since g⁻¹h ∈ S iff h ∈ gS, e gives S)
    A_float = A.float()
    a_sq_e = torch.bmm(mask.unsqueeze(1).float(), A_float).squeeze(1)

    expected = torch.where(mask, lambda_, mu).float()
    expected[:, e] = t

    is_dsrg = (a_sq_e == expected).all(dim=1)
    return subsets[is_dsrg]


def build_adjacency(
    subsets: Int[Tensor, "B k"], group: GroupTable
) -> Bool[Tensor, "B n n"]:
    """Build adjacency matrices from connection sets."""
    n = group.order
    left_mult = group.table[group.inv]
    mask = torch.zeros(subsets.shape[0], n, dtype=torch.bool, device=subsets.device)
    mask.scatter_(1, subsets.long(), True)
    return mask[:, left_mult.long()]


# ---------------------------------------------------------------------------
# Per-group search
# ---------------------------------------------------------------------------


@torch.no_grad()
def search_one_group(
    params: DSRGParams,
    group: GroupTable,
    structure: GroupStructure,
    *,
    batch_size: int = 100_000,
    device: torch.device | str = "cpu",
    on_batch: Callable[[BatchUpdate], None] | None = None,
) -> GroupSearchResult:
    """Search a single group for DSRGs at the given parameter set.

    Pure compute: no file I/O, no printing. Pass an `on_batch` callback if you
    want progress reporting (e.g. tqdm) from an orchestration layer.

    If `structure.t_valid_count == 0`, returns immediately with `found=None`.
    """
    result_empty = GroupSearchResult(structure=structure, found=None, elapsed_s=0.0)
    if structure.t_valid_count == 0:
        return result_empty

    found_chunks: list[Tensor] = []
    checked = 0
    t0 = time.perf_counter()

    batch_iter = _t_valid_batches(structure, params.k, params.t, batch_size, device)
    for batch_idx, batch in enumerate(batch_iter):
        checked += int(batch.shape[0])
        hits = check_dsrg(batch, group, params.t, params.lambda_, params.mu)
        if hits.shape[0] > 0:
            found_chunks.append(hits)
        if on_batch is not None:
            found_so_far = sum(int(h.shape[0]) for h in found_chunks)
            on_batch(
                BatchUpdate(
                    batch_idx=batch_idx,
                    total_batches=structure.total_batches,
                    subsets_checked=checked,
                    found_so_far=found_so_far,
                )
            )

    elapsed = time.perf_counter() - t0
    found = torch.cat(found_chunks, dim=0) if found_chunks else None
    return GroupSearchResult(structure=structure, found=found, elapsed_s=elapsed)
