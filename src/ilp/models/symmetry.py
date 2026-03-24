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

LexOrder = Literal["none", "exponential", "lex_leader"]


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
    else:
        raise ValueError(
            f"Unknown lex order kind {kind!r}; choose from: none, exponential, lex_leader"
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
    r"""Classic lex-leader formulation with auxiliary binary variables.

    For consecutive rows *i* and *i+1*, compare on all *n* columns
    ``C = (0, 1, ..., n-1)``.  The edge accessor returns 0 for diagonal
    entries, so columns *i* and *i+1* participate as constants.

    Introduce binary variables ``g_j`` (j = 0 .. n-1) where ``g_j = 1``
    means "rows i and i+1 agree on columns 0 .. j".  Then:

    **Equality tracking** (g_j = 0 whenever rows disagree at column j):

    * ``g_j <= 1 - edge(i, j) + edge(i+1, j)``
    * ``g_j <= 1 + edge(i, j) - edge(i+1, j)``

    **Chain** (agreement through j requires agreement through j-1):

    * ``g_j <= g_{j-1}``   for j >= 1

    **Lex constraint** (row i cannot win at the first disagreement):

    * ``edge(i, 0) - edge(i+1, 0) <= 0``
    * ``edge(i, j) - edge(i+1, j) <= 1 - g_{j-1}``   for j >= 1

    Uses O(n) binary variables and O(n) constraints per row pair.
    """
    for i in range(start_row, n - 1):
        g = model.addVars(
            n, vtype=GRB.BINARY, name=f"lexldr_g_{i}",
        )

        for j in range(n):
            ei = edge(i, j)
            eip1 = edge(i + 1, j)

            # g[j] = 1  =>  edge(i,j) == edge(i+1,j)
            model.addConstr(
                g[j] <= 1 - ei + eip1,
                name=f"lexldr_eq_hi_{i}_{j}",
            )
            model.addConstr(
                g[j] <= 1 + ei - eip1,
                name=f"lexldr_eq_lo_{i}_{j}",
            )

            # Chain: g[j] <= g[j-1]
            if j > 0:
                model.addConstr(
                    g[j] <= g[j - 1],
                    name=f"lexldr_chain_{i}_{j}",
                )

            # Lex constraint: if all previous columns are equal,
            # row i must not beat row i+1 at this column.
            if j == 0:
                model.addConstr(
                    ei - eip1 <= 0,
                    name=f"lexldr_lex_{i}_{j}",
                )
            else:
                model.addConstr(
                    ei - eip1 <= 1 - g[j - 1],
                    name=f"lexldr_lex_{i}_{j}",
                )
