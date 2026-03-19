"""Symmetry-breaking constraints for SRG / DSRG ILP models.

Provides reusable lex-ordering strategies that can be added to any Gurobi
model whose rows represent vertex adjacency.  All functions operate on an
*edge accessor* — a callable ``(i, j) -> Var | 0`` — so they are agnostic
about whether edges are stored in an upper-triangle dict, a full matrix, etc.

All strategies enforce **non-strict** lexicographic ordering
(row_i ≤_lex row_{i+1}).

Two strategies are implemented:

1. **Exponential** (powers-of-2 weighted sums):
   For consecutive rows (i, i+1), compare on the aligned column set excluding
   the structurally-different diagonal positions j=i and j=i+1.  Assign
   strictly decreasing powers of 2 so that a single linear inequality
   preserves lexicographic order exactly.  Zero auxiliary variables, but
   coefficients grow as 2^n.

2. **Lex-leader** (auxiliary-variable chain):
   Classic MIP lex-leader encoding.  For each consecutive row pair (i, i+1)
   and column set C (excluding i and i+1), introduce O(|C|) auxiliary binary
   variables that track "all columns up to j are equal."  The first column
   where the rows differ must favour row i+1.  Moderate variable count, but
   all coefficients are 0/1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    from collections.abc import Callable

LexOrder = Literal["none", "exponential", "lex_leader"]

# Sentinel for "no edge" (diagonal).  We use the int 0 so that gp.quicksum
# and arithmetic work transparently.
_ZERO = 0

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

    For consecutive rows *i* and *i+1*, we compare on the "aligned" column
    set ``C = {j : j != i and j != i+1}`` (these are the columns where both
    rows are ordinary edge variables rather than structural zeros/ones from
    the diagonal).

    Assign weight ``2^(|C| - pos)`` to column ``C[pos]``.  Then::

        sum_{j in C} w_j * edge(i+1, j)  >=  sum_{j in C} w_j * edge(i, j)

    Because the weights are strictly decreasing powers of 2, this single
    inequality is equivalent to full lexicographic comparison on the column
    set.
    """
    for i in range(start_row, n - 1):
        # Exclude diagonal positions and any column where both entries are
        # fixed constants (e.g. due to fix_neighbors).  Fixed entries that
        # are *equal* in both rows carry no lex information; fixed entries
        # that differ would make the constraint trivially infeasible or
        # trivially satisfied, so we handle them separately below.
        all_cols = [j for j in range(n) if j != i and j != i + 1]

        # Classify each column.
        free_cols: list[int] = []
        for j in all_cols:
            ei_j = edge(i, j)
            eip1_j = edge(i + 1, j)
            both_fixed = not isinstance(ei_j, gp.Var) and not isinstance(eip1_j, gp.Var)
            if both_fixed:
                # If row i already beats row i+1 at a fixed column, the
                # entire lex constraint is violated — skip adding constraints
                # for this row pair (the model is already over-constrained by
                # the fixed bounds).
                if int(ei_j) > int(eip1_j):
                    break
                # If row i+1 already beats row i, lex order is satisfied for
                # this row pair regardless of free columns — no constraint needed.
                if int(eip1_j) > int(ei_j):
                    free_cols = []  # signal: constraint trivially satisfied
                    break
                # Equal fixed values: neutral, continue scanning.
            else:
                free_cols.append(j)
        else:
            # Loop completed without break — add constraint on free columns.
            if not free_cols:
                continue
            weights = {j: 1 << (len(free_cols) - idx) for idx, j in enumerate(free_cols)}
            sum_i = gp.quicksum(edge(i, j) * weights[j] for j in free_cols)
            sum_ip1 = gp.quicksum(edge(i + 1, j) * weights[j] for j in free_cols)
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

    For consecutive rows *i* and *i+1* on the aligned column set
    ``C = [c_0, c_1, ..., c_{m-1}]``:

    Introduce binary variables ``g_j`` (j = 0 .. m-1) where ``g_j = 1``
    means "rows i and i+1 agree on columns c_0 .. c_j".  Then:

    * ``g_0 = 1  =>  edge(i, c_0) == edge(i+1, c_0)``
    * ``g_j = 1  =>  g_{j-1} = 1  and  edge(i, c_j) == edge(i+1, c_j)``
    * At the first position where they disagree, row i+1 must be ≥ row i.

    We model this as follows.  Let ``d_j = edge(i+1, c_j) - edge(i, c_j)``
    (in {-1, 0, 1}).  Define ``g_j`` = "equal through column j":

    * ``g_0 <= 1 - (edge(i, c_0) - edge(i+1, c_0))``   (no row_i wins at 0)
    * ``g_0 <= 1 - (edge(i+1, c_0) - edge(i, c_0))``   (no row_i+1 wins at 0)
    * ``g_j <= g_{j-1}``                                 (chain)
    * ``g_j <= 1 - (edge(i, c_j) - edge(i+1, c_j))``
    * ``g_j <= 1 - (edge(i+1, c_j) - edge(i, c_j))``

    The lex constraint: ``edge(i, c_j) - edge(i+1, c_j) <= 1 - g_{j-1}``
    for all j >= 1 (and j=0 unconditionally).  This says "if all previous
    columns are equal, row i cannot win at column j."

    Uses O(m) binary variables and O(m) constraints per row pair.
    """
    for i in range(start_row, n - 1):
        all_cols = [j for j in range(n) if j != i and j != i + 1]

        # Pre-scan fixed columns to determine the lex status before free cols.
        # fixed_winner: +1 if row i+1 already won (lex satisfied trivially),
        #               -1 if row i already won (lex violated — skip pair),
        #                0 if all fixed cols seen so far are equal.
        fixed_winner = 0
        free_cols: list[int] = []
        skip_pair = False
        for j in all_cols:
            ei_j = edge(i, j)
            eip1_j = edge(i + 1, j)
            both_fixed = not isinstance(ei_j, gp.Var) and not isinstance(eip1_j, gp.Var)
            if both_fixed:
                if fixed_winner != 0:
                    # Winner already decided by an earlier fixed column.
                    continue
                ei_val = int(ei_j)
                eip1_val = int(eip1_j)
                if eip1_val > ei_val:
                    fixed_winner = 1   # row i+1 already leads
                elif ei_val > eip1_val:
                    fixed_winner = -1  # row i leads — lex violated
                    skip_pair = True
                    break
            else:
                free_cols.append(j)

        if skip_pair:
            # Fixed edges already make this row pair lex-infeasible; the
            # model's fixed bounds render the constraint unsatisfiable.
            # Don't add any constraints — the fixed bounds will be
            # propagated by the solver automatically.
            continue
        if fixed_winner == 1:
            # Row i+1 already leads at a fixed column — lex is satisfied
            # regardless of free columns.  No constraint needed.
            continue

        # All fixed columns are equal; enforce lex order on free columns.
        m = len(free_cols)
        if m == 0:
            continue

        # g[j] = 1  iff  rows i and i+1 agree on free_cols[0 .. j].
        g = model.addVars(
            m, vtype=GRB.BINARY, name=f"lexldr_g_{i}",
        )

        for j_idx in range(m):
            c = free_cols[j_idx]
            ei = edge(i, c)
            eip1 = edge(i + 1, c)

            # g[j] = 1  =>  edge(i,c) == edge(i+1,c)
            model.addConstr(
                g[j_idx] <= 1 - ei + eip1,
                name=f"lexldr_eq_hi_{i}_{j_idx}",
            )
            model.addConstr(
                g[j_idx] <= 1 + ei - eip1,
                name=f"lexldr_eq_lo_{i}_{j_idx}",
            )

            # Chain: g[j] <= g[j-1]
            if j_idx > 0:
                model.addConstr(
                    g[j_idx] <= g[j_idx - 1],
                    name=f"lexldr_chain_{i}_{j_idx}",
                )

            # Lex constraint: if all previous (free) columns are equal,
            # row i must not beat row i+1 at this column.
            if j_idx == 0:
                model.addConstr(
                    ei - eip1 <= 0,
                    name=f"lexldr_lex_{i}_{j_idx}",
                )
            else:
                model.addConstr(
                    ei - eip1 <= 1 - g[j_idx - 1],
                    name=f"lexldr_lex_{i}_{j_idx}",
                )
