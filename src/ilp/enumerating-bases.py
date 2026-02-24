#!/usr/bin/env python3
"""
Enumerate all sets of 6 linearly independent weight-10 binary vectors in
GF(2)^24, with one vector fixed as col0, up to coordinate permutations
that fix col0.  These are the candidate column-space bases for
DSRG(n=24, k=10, t=5, λ=3, μ=5) with GF(2)-rank 6.

Approach
--------
A 6-dimensional subspace V ≤ GF(2)^24 containing col0 is characterised by
its *block type*: a function  f : GF(2)^6 → ℤ≥0  that records, for every
possible 6-bit signature (col0-value, v1-value, …, v5-value), how many of
the 24 coordinates carry that signature.

Constraints on f
    (C1) Σ_p f(p) = 24
    (C2) ∀ bit i ∈ {0,…,5}: Σ_{p : p_i=1} f(p) = 10      (weight-10 basis)
    (C3) supp(f) spans GF(2)^6                              (linear independence)
    (C4) ≥ 24 of the 63 non-zero vectors in V have weight 10 (DSRG feasibility)

The stabiliser of col0 under S_24 is S_14 × S_10 (permuting the 14 zero-
positions and the 10 one-positions independently).  Bit 0 of the signature
records which group a coordinate belongs to, giving a natural decomposition:

    Z-part (bit 0 = 0):  g : {0,1}^5 → ℤ≥0,  Σg = 14,  col-sums = z_i
    O-part (bit 0 = 1):  h : {0,1}^5 → ℤ≥0,  Σh = 10,  col-sums = 10−z_i

We enumerate *sorted* z-tuples (z_1 ≥ … ≥ z_5) for the S_5 symmetry on
bits 1–5, then combine every (g, h) pair and check (C3)–(C4).  Finally each
surviving f is canonicalised under the full S_5 and deduplicated.
"""

from __future__ import annotations

import sys
import time
from itertools import permutations as perms
from math import comb

import numpy as np

# ── DSRG parameters ─────────────────────────────────────────────────────────
N = 24
K = 10
RANK = 6
MIN_WT_K = 24  # minimum # of weight-K vectors in the subspace for DSRG

# ── 5-bit pattern helpers ───────────────────────────────────────────────────
PAT5 = [tuple(int(bool(p & (1 << i))) for i in range(5)) for p in range(32)]

# For every non-zero c ∈ GF(2)^5, record which PAT5 indices satisfy c·b = 1.
HALVES: dict[tuple[int, ...], frozenset[int]] = {}
for _ci in range(1, 32):
    _c = tuple(int(bool(_ci & (1 << i))) for i in range(5))
    HALVES[_c] = frozenset(
        j for j, b in enumerate(PAT5)
        if sum(_c[i] * b[i] for i in range(5)) % 2 == 1
    )

# Order the 32 five-bit patterns by *decreasing* Hamming weight so that the
# most-constrained patterns (touching the most column-sum constraints) are
# tried first.  This dramatically improves pruning in the recursive search.
PAT5_ORDER = sorted(range(32), key=lambda j: (-sum(PAT5[j]), PAT5[j]))


# ── Recursive contingency-table enumerator ──────────────────────────────────
def enumerate_part(total: int, col_sums: list[int]) -> list[list[int]]:
    """Yield every  g : {0,1}^5 → ℤ≥0  with the given total and column sums.

    *col_sums[i]* is  Σ_{b : b_i=1} g(b).  The patterns are visited in
    PAT5_ORDER (high-weight first) so that constraint propagation cuts
    branches early.
    """
    n_pat = 32
    order = PAT5_ORDER  # indices into PAT5

    # Pre-compute bit masks for the ordered patterns
    bit_of = [[PAT5[order[pos]][i] for pos in range(n_pat)] for i in range(5)]

    # For each (bit i, position pos), count how many *later* ordered patterns
    # have bit i = 1.  Used for lower-bound / feasibility pruning.
    future_count: list[list[int]] = [[0] * (n_pat + 1) for _ in range(5)]
    for i in range(5):
        acc = 0
        for pos in range(n_pat - 1, -1, -1):
            future_count[i][pos + 1] = acc
            if bit_of[i][pos]:
                acc += 1
        future_count[i][0] = acc

    result: list[list[int]] = []
    g = [0] * n_pat  # g[pos] = value assigned to ordered pattern at position pos

    def recurse(pos: int, rem_total: int, rem_col: list[int]) -> None:
        if pos == n_pat:
            if rem_total == 0 and all(r == 0 for r in rem_col):
                # Un-shuffle back to natural PAT5 order
                out = [0] * 32
                for p in range(n_pat):
                    out[order[p]] = g[p]
                result.append(out)
            return

        # ── bounds on g[pos] ────────────────────────────────────────────
        ub = rem_total
        lb = 0
        for i in range(5):
            if bit_of[i][pos]:
                ub = min(ub, rem_col[i])
                if future_count[i][pos + 1] == 0:  # last pattern with this bit
                    lb = max(lb, rem_col[i])
        if lb > ub:
            return

        for v in range(lb, ub + 1):
            new_rem = rem_total - v
            new_col = rem_col[:]
            ok = True
            for i in range(5):
                if bit_of[i][pos]:
                    new_col[i] -= v
            # feasibility of remaining assignments
            for i in range(5):
                if new_col[i] < 0:
                    ok = False
                    break
                if new_col[i] > new_rem:
                    ok = False
                    break
                if new_col[i] > 0 and future_count[i][pos + 1] == 0:
                    ok = False
                    break
            if not ok:
                continue
            g[pos] = v
            recurse(pos + 1, new_rem, new_col)
        g[pos] = 0

    recurse(0, total, list(col_sums))
    return result


# ── Weight helpers ──────────────────────────────────────────────────────────
def _half_weights(arr: list[int]) -> dict[tuple[int, ...], int]:
    """For every non-zero c ∈ GF(2)^5 compute  Σ_{c·b=1} arr[b]."""
    return {c: sum(arr[j] for j in idx) for c, idx in HALVES.items()}


def count_weight_k(zw: dict, hw: dict) -> int:
    """Count non-zero c ∈ GF(2)^6 whose corresponding subspace vector has
    weight exactly K.

    Writing  c = (c0, c'),  c' ∈ GF(2)^5:
        weight((0, c')) = Zw(c') + Hw(c')          for c' ≠ 0
        weight((1, c')) = Zw(c') + 10 − Hw(c')     for any c'
        weight((1, 0))  = 0 + 10 − 0 = 10          (this is col0 itself)
    """
    count = 1  # (1, 0…0) = col0 always has weight K
    for c in HALVES:
        if zw[c] + hw[c] == K:
            count += 1          # c0 = 0 contribution
        if zw[c] + 10 - hw[c] == K:
            count += 1          # c0 = 1 contribution
    return count


# ── Spanning check ──────────────────────────────────────────────────────────
def spans_gf2_6(f_dict: dict[tuple[int, ...], int]) -> bool:
    """True iff the support of *f_dict* spans GF(2)^6."""
    support = [list(p) for p, v in f_dict.items() if v > 0]
    if len(support) < RANK:
        return False
    rows = [r[:] for r in support]
    pivot = 0
    for col in range(RANK):
        found = -1
        for r in range(pivot, len(rows)):
            if rows[r][col]:
                found = r
                break
        if found == -1:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        for r in range(len(rows)):
            if r != pivot and rows[r][col]:
                for c in range(RANK):
                    rows[r][c] ^= rows[pivot][c]
        pivot += 1
    return pivot == RANK


# ── Canonical form under S_5 on bits 1–5 ───────────────────────────────────
def canonical_form(
    f_dict: dict[tuple[int, ...], int],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return the lexicographically smallest version of *f_dict* over all 5!
    permutations of the last five signature bits."""
    items = [(p, v) for p, v in f_dict.items() if v > 0]
    best: tuple | None = None
    for perm in perms(range(5)):
        permuted = []
        for p, v in items:
            new_p = (p[0],) + tuple(p[1 + perm[i]] for i in range(5))
            permuted.append((new_p, v))
        permuted.sort()
        key = tuple(permuted)
        if best is None or key < best:
            best = key
    assert best is not None
    return best


# ── Sorted z-tuple generator ───────────────────────────────────────────────
def gen_sorted_z(dim: int, max_val: int) -> list[tuple[int, ...]]:
    """Return all non-increasing tuples of length *dim* with values in [0, max_val]."""
    out: list[tuple[int, ...]] = []

    def _rec(remaining: int, ceiling: int, cur: list[int]) -> None:
        if remaining == 0:
            out.append(tuple(cur))
            return
        for z in range(ceiling, -1, -1):
            cur.append(z)
            _rec(remaining - 1, z, cur)
            cur.pop()

    _rec(dim, max_val, [])
    return out


# ── Upper-bound filter on max achievable weight-K count ─────────────────────
def max_possible_wt_k(zw: dict) -> int:
    """Given Z-weight vector, return an upper bound on count_weight_k over
    all admissible h.  Used to skip g's that can never reach MIN_WT_K."""
    # For each non-zero c' ∈ GF(2)^5:
    #   • (0, c') contributes to weight-K iff Hw(c') = K − Zw(c')
    #   • (1, c') contributes iff Hw(c') = Zw(c')
    #   A single h gives one Hw(c') per c'.  Best case: it hits both targets
    #   when they coincide (Zw = 5), else it can hit at most one per c'.
    ub = 1  # col0 always counts
    for c in HALVES:
        zv = zw[c]
        target_a = K - zv  # need Hw = target_a for (0,c') to count
        target_b = zv      # need Hw = target_b for (1,c') to count
        if not (0 <= target_a <= 10):
            contrib_a = False
        else:
            contrib_a = True
        if not (0 <= target_b <= 10):
            contrib_b = False
        else:
            contrib_b = True
        if contrib_a and contrib_b and target_a == target_b:
            ub += 2
        elif contrib_a and contrib_b:
            ub += 2  # h might satisfy either; one h can only satisfy one,
            # but across different c' the maxima add up
        elif contrib_a or contrib_b:
            ub += 1
    return ub


# ── Full weight spectrum ────────────────────────────────────────────────────
def weight_spectrum(f_dict: dict[tuple[int, ...], int]) -> dict[int, int]:
    """Return {weight: count} for all 63 non-zero subspace vectors."""
    spec: dict[int, int] = {}
    for c_int in range(1, 64):
        c = tuple(int(bool(c_int & (1 << i))) for i in range(RANK))
        wt = sum(
            v for p, v in f_dict.items()
            if sum(c[i] * p[i] for i in range(RANK)) % 2 == 1
        )
        spec[wt] = spec.get(wt, 0) + 1
    return spec


# ── Main enumeration ────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()

    z_tuples = gen_sorted_z(5, K)
    print(f"DSRG({N},{K},5,3,5)  rank {RANK}")
    print(f"Sorted z-tuples to explore: {len(z_tuples)}")
    print(f"Minimum weight-{K} vectors required: {MIN_WT_K}")
    print(flush=True)

    results: dict[
        tuple, tuple[tuple[int, ...], dict[tuple[int, ...], int], int]
    ] = {}
    stats = dict(z_done=0, g_total=0, h_total=0, pairs=0, wt_pass=0, span_pass=0)

    for z_idx, z_tuple in enumerate(z_tuples):
        z = list(z_tuple)
        o = [K - zi for zi in z]
        if any(oi < 0 or oi > K for oi in o):
            stats["z_done"] += 1
            continue

        g_list = enumerate_part(14, z)
        if not g_list:
            stats["z_done"] += 1
            continue

        h_list = enumerate_part(10, o)
        if not h_list:
            stats["z_done"] += 1
            continue

        stats["g_total"] += len(g_list)
        stats["h_total"] += len(h_list)

        # Pre-compute H-weight vectors for all h solutions
        hw_list = [_half_weights(h) for h in h_list]

        for g in g_list:
            zw = _half_weights(g)

            # Quick upper-bound prune
            if max_possible_wt_k(zw) < MIN_WT_K:
                stats["pairs"] += len(h_list)
                continue

            for hi, h in enumerate(h_list):
                stats["pairs"] += 1
                hw = hw_list[hi]

                wt_k = count_weight_k(zw, hw)
                if wt_k < MIN_WT_K:
                    continue
                stats["wt_pass"] += 1

                # Build full f : GF(2)^6 → ℤ≥0
                f_dict: dict[tuple[int, ...], int] = {}
                for j, b in enumerate(PAT5):
                    if g[j] > 0:
                        f_dict[(0,) + b] = g[j]
                    if h[j] > 0:
                        f_dict[(1,) + b] = h[j]

                if not spans_gf2_6(f_dict):
                    continue
                stats["span_pass"] += 1

                cf = canonical_form(f_dict)
                if cf not in results:
                    results[cf] = (z_tuple, f_dict, wt_k)

        stats["z_done"] += 1

        # Progress report every 200 z-tuples
        if stats["z_done"] % 200 == 0 or stats["z_done"] == len(z_tuples):
            elapsed = time.time() - t0
            print(
                f"  [{stats['z_done']:>5}/{len(z_tuples)}]  "
                f"g={stats['g_total']:>9,}  h={stats['h_total']:>9,}  "
                f"pairs={stats['pairs']:>12,}  "
                f"wt✓={stats['wt_pass']:>8,}  span✓={stats['span_pass']:>6,}  "
                f"unique={len(results):>5}  {elapsed:7.1f}s",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n{'═' * 72}")
    print(f"Enumeration complete in {elapsed:.1f}s")
    print(f"  z-tuples examined : {stats['z_done']}")
    print(f"  total g-solutions : {stats['g_total']:,}")
    print(f"  total h-solutions : {stats['h_total']:,}")
    print(f"  (g,h) pairs tested: {stats['pairs']:,}")
    print(f"  passed weight-{K}  : {stats['wt_pass']:,}")
    print(f"  passed spanning   : {stats['span_pass']:,}")
    print(f"  canonical types   : {len(results)}")
    print(f"{'═' * 72}\n")

    # ── Detailed output ─────────────────────────────────────────────────
    for i, (cf, (z_tuple, f_dict, wt_k)) in enumerate(sorted(results.items()), 1):
        ws = weight_spectrum(f_dict)
        print(f"Type {i}")
        print(f"  z-tuple (col-sums in Z-block) : {z_tuple}")
        print(f"  weight-{K} vectors in subspace: {wt_k} / 63")
        print(f"  full weight spectrum           : {dict(sorted(ws.items()))}")
        print(f"  non-zero block sizes (signature → count):")
        for p in sorted(f_dict):
            v = f_dict[p]
            if v > 0:
                tag = "Z" if p[0] == 0 else "O"
                print(f"    {tag} {p[1:]}: {v:>2}")

        # Orbit size: product of C(block_size, sub_count) across blocks that
        # will be refined at each stage.  For now just report the product of
        # binomial coefficients for each block as a rough indicator.
        orbit = 1
        for p, v in f_dict.items():
            pass  # orbit accounting is complex; omitted for clarity
        print()

    print(f"Total non-isomorphic basis types: {len(results)}")


if __name__ == "__main__":
    main()
