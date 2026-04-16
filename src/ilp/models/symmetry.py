"""Symmetry-breaking constraints for SRG / DSRG ILP models.

Provides reusable lex-ordering strategies that can be added to any Gurobi
model whose rows represent vertex adjacency.  All functions operate on an
*edge accessor* — a callable ``(i, j) -> Var | 0`` — so they are agnostic
about whether edges are stored in an upper-triangle dict, a full matrix, etc.

All strategies enforce **non-strict** lexicographic ordering
(row_i ≤_lex row_{i+1}).

Two strategies are implemented:

1. **Exponential** (powers-of-2 weighted sums):
   For consecutive rows (i, i+1), compare on all n columns.  The edge
   accessor returns 0 for diagonal entries, so they participate as constants.
   Assign strictly decreasing powers of 2 so that a single linear inequality
   preserves lexicographic order exactly.  Zero auxiliary variables, but
   coefficients grow as 2^n.

2. **Lex-leader** (auxiliary-variable chain):
   Classic MIP lex-leader encoding.  For each consecutive row pair (i, i+1),
   compare on all n columns.  Introduce O(n) auxiliary binary variables that
   track "all columns up to j are equal."  The first column where the rows
   differ must favour row i+1.  Moderate variable count, but all coefficients
   are 0/1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    from collections.abc import Callable

LexOrder = Literal["none", "exponential", "lex_leader", "hybrid"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def add_lex_order(
    model: gp.Model,
    edge: Callable[[int, int], gp.Var | int],
    n: int,
    *,
    kind: LexOrder = "none",
    start_row: int = 0,
    block_size: int = 10,
) -> None:
    """Dispatch to the appropriate lex-ordering strategy.

    Enforces non-strict lexicographic ordering: row_i ≤_lex row_{i+1}.

    Args:
        model: Gurobi model to add constraints to.
        edge: Callable ``(i, j) -> Var | 0``.  Must handle ``i == j`` by
            returning 0 (or a variable fixed to 0).
        n: Number of vertices (rows/cols of the adjacency matrix).
        kind: One of ``"none"``, ``"exponential"``, ``"lex_leader"``.
        start_row: First row index for which to add an ordering constraint
            against the next row.  Set to 1 when vertex-0 neighbour fixing
            is active (row 0 is already fully determined).
    """
    if kind == "none":
        return
    if kind == "exponential":
        _add_lex_exponential(model, edge, n, start_row=start_row)
    elif kind == "lex_leader":
        _add_lex_leader(model, edge, n, start_row=start_row)
    elif kind == "hybrid":
        _add_lex_hybrid(model, edge, n, start_row=start_row, block_size=block_size)
    else:
        raise ValueError(
            f"Unknown lex order kind {kind!r}; choose from: none, exponential, lex_leader, hybrid"
        )


# ---------------------------------------------------------------------------
# Strategy 1 — exponential weights
# ---------------------------------------------------------------------------

def _add_lex_exponential(
    model: gp.Model,
    edge: Callable[[int, int], gp.Var | int],
    n: int,
    *,
    start_row: int,
) -> None:
    """Lex ordering via powers-of-2 weighted row sums.

    For consecutive rows *i* and *i+1*, compare on all *n* columns.  The
    edge accessor returns 0 for diagonal entries (``edge(i, i) = 0``), so
    they participate as constants.

    Assign weight ``2^(n - 1 - j)`` to column ``j``.  Then::

        sum_j  w_j * edge(i+1, j)  >=  sum_j  w_j * edge(i, j)

    Because the weights are strictly decreasing powers of 2, this single
    inequality is equivalent to full lexicographic comparison.
    """
    for i in range(start_row, n - 1):
        cols = list(range(n))
        weights = {j: 1 << (n - 1 - j) for j in cols}
        sum_i = gp.quicksum(edge(i, j) * weights[j] for j in cols)
        sum_ip1 = gp.quicksum(edge(i + 1, j) * weights[j] for j in cols)
        model.addConstr(sum_ip1 >= sum_i, name=f"lex_exp_{i}")


# ---------------------------------------------------------------------------
# Strategy 2 — lex-leader (auxiliary binary chain)
# ---------------------------------------------------------------------------

def _add_lex_leader(
    model: gp.Model,
    edge: Callable[[int, int], gp.Var | int],
    n: int,
    *,
    start_row: int,
) -> None:
    r"""Lex-leader formulation with auxiliary binary variables.

    For consecutive rows *i* and *i+1*, compare on all *n* columns.

    Two sets of auxiliary binary variables per row pair:

    ``d_j`` — difference indicator: ``d_j = 1`` iff ``edge(i,j) != edge(i+1,j)``.
    Encoded with four constraints that force ``d_j`` to exactly track
    disagreement for binary edge variables:

    * ``d_j >= edge(i,j) - edge(i+1,j)``
    * ``d_j >= edge(i+1,j) - edge(i,j)``
    * ``d_j <= edge(i,j) + edge(i+1,j)``          (d=0 when both are 0)
    * ``d_j <= 2 - edge(i,j) - edge(i+1,j)``      (d=0 when both are 1)

    ``g_j`` — agreement chain: ``g_j = 1`` iff rows agree on all columns
    ``0 .. j``.  The chain is enforced with both an upper and a lower bound
    so that agreement propagates correctly in feasibility problems (without
    an objective the lower bound is essential — without it the solver sets
    all ``g_j = 0``, making every lex constraint vacuous):

    * ``g_j <= 1 - d_j``                           (disagree → g=0)
    * ``g_j <= g_{j-1}``        for j >= 1         (chain down)
    * ``g_0 >= 1 - d_0``                            (base: agree → g=1)
    * ``g_j >= g_{j-1} - d_j``  for j >= 1         (agree → propagate)

    **Lex constraint** (row i cannot win at the first disagreement):

    * ``edge(i, 0) - edge(i+1, 0) <= 0``
    * ``edge(i, j) - edge(i+1, j) <= 1 - g_{j-1}``   for j >= 1

    Uses O(n) binary variables per set (2n total) and O(n) constraints per
    row pair.
    """
    for i in range(start_row, n - 1):
        g = model.addVars(n, vtype=GRB.BINARY, name=f"lexldr_g_{i}")
        d = model.addVars(n, vtype=GRB.BINARY, name=f"lexldr_d_{i}")

        for j in range(n):
            ei = edge(i, j)
            eip1 = edge(i + 1, j)

            # ── Difference indicator: d[j] = 1 iff ei != eip1 ────────────
            model.addConstr(d[j] >= ei - eip1,       name=f"lexldr_dlo_{i}_{j}")
            model.addConstr(d[j] >= eip1 - ei,       name=f"lexldr_dhi_{i}_{j}")
            model.addConstr(d[j] <= ei + eip1,       name=f"lexldr_d00_{i}_{j}")
            model.addConstr(d[j] <= 2 - ei - eip1,   name=f"lexldr_d11_{i}_{j}")

            # ── Agreement chain: g[j] = 1 iff agreed on 0..j ─────────────
            model.addConstr(g[j] <= 1 - d[j],        name=f"lexldr_gup_{i}_{j}")
            if j == 0:
                model.addConstr(g[j] >= 1 - d[j],    name=f"lexldr_glo_{i}_{j}")
            else:
                model.addConstr(g[j] <= g[j - 1],    name=f"lexldr_chain_{i}_{j}")
                model.addConstr(g[j] >= g[j-1] - d[j], name=f"lexldr_glo_{i}_{j}")

            # ── Lex constraint ────────────────────────────────────────────
            if j == 0:
                model.addConstr(ei - eip1 <= 0,           name=f"lexldr_lex_{i}_{j}")
            else:
                model.addConstr(ei - eip1 <= 1 - g[j-1], name=f"lexldr_lex_{i}_{j}")


# ---------------------------------------------------------------------------
# Strategy 3 — hybrid (exponential within blocks, lex-leader between)
# ---------------------------------------------------------------------------

def _add_lex_hybrid(
    model: gp.Model,
    edge: Callable[[int, int], gp.Var | int],
    n: int,
    *,
    start_row: int,
    block_size: int = 10,
) -> None:
    r"""Hybrid lex ordering: exponential within blocks, lex-leader between.

    Columns are divided into blocks of *block_size*.  Within each block,
    a single exponential-weighted difference expression ``D_b`` captures
    the lex comparison (coefficients at most ``2^block_size``).  Between
    blocks, auxiliary binary variables chain the comparisons in the style
    of lex-leader.

    For each consecutive row pair (i, i+1) and each block *b*:

    * ``D_b = Σ_j 2^(B-1-idx) · (edge(i+1, j) − edge(i, j))``
    * ``d_b ∈ {0,1}``:  difference indicator,  ``|D_b| ≤ M_b · d_b``
    * ``g_b ∈ {0,1}``:  agreement chain — 1 iff rows agree on blocks 0‥b

    Agreement chain:

    * ``g_b ≤ 1 − d_b``          (differ → break chain)
    * ``g_b ≤ g_{b−1}``          (monotone, b > 0)
    * ``g_b ≥ g_{b−1} − d_b``    (reverse implication, b > 0)
    * ``g_0 ≥ 1 − d_0``          (base case)

    Lex constraint:

    * ``D_0 ≥ 0``
    * ``D_b ≥ −M_b · (1 − g_{b−1})``   (b > 0)

    Uses ``2 · ⌈n / block_size⌉`` binary variables per row pair.
    """
    for i in range(start_row, n - 1):
        # Partition columns into blocks.
        blocks: list[list[int]] = []
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            blocks.append(list(range(start, end)))

        num_blocks = len(blocks)

        # d[b]: 1 if rows differ on block b.
        d = model.addVars(num_blocks, vtype=GRB.BINARY, name=f"lexhyb_d_{i}")
        # g[b]: 1 if rows agree on all of blocks 0..b.
        g = model.addVars(num_blocks, vtype=GRB.BINARY, name=f"lexhyb_g_{i}")

        for b, cols in enumerate(blocks):
            B = len(cols)
            weights = {j: 1 << (B - 1 - idx) for idx, j in enumerate(cols)}
            M_b = (1 << B) - 1

            # D_b = weighted (row_{i+1} − row_i) on this block's columns.
            D_b = gp.quicksum(
                (edge(i + 1, j) - edge(i, j)) * weights[j] for j in cols
            )

            # ── Difference indicator: |D_b| ≤ M_b · d[b] ────────────────
            model.addConstr(D_b <= M_b * d[b], name=f"lexhyb_dhi_{i}_{b}")
            model.addConstr(-D_b <= M_b * d[b], name=f"lexhyb_dlo_{i}_{b}")

            # ── Agreement chain ──────────────────────────────────────────
            model.addConstr(g[b] <= 1 - d[b], name=f"lexhyb_gd_{i}_{b}")
            if b == 0:
                model.addConstr(g[b] >= 1 - d[b], name=f"lexhyb_grev_{i}_{b}")
            else:
                model.addConstr(g[b] <= g[b - 1], name=f"lexhyb_gchain_{i}_{b}")
                model.addConstr(
                    g[b] >= g[b - 1] - d[b], name=f"lexhyb_grev_{i}_{b}",
                )

            # ── Lex constraint ───────────────────────────────────────────
            if b == 0:
                model.addConstr(D_b >= 0, name=f"lexhyb_lex_{i}_{b}")
            else:
                model.addConstr(
                    D_b >= -M_b * (1 - g[b - 1]),
                    name=f"lexhyb_lex_{i}_{b}",
                )
