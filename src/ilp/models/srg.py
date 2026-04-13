"""Strongly Regular Graph (SRG) ILP formulations using native gurobipy.

An SRG(n, k, λ, μ) is an undirected graph on n vertices where every vertex
has degree k, every pair of adjacent vertices has exactly λ common neighbours,
and every pair of non-adjacent vertices has exactly μ common neighbours.

Three formulations are provided:

1. **Exact** — all constraints are hard.  The solver either finds a feasible
   SRG or proves infeasibility.  Uses ``addGenConstrAnd`` for product
   linearisation so Gurobi can exploit the AND structure directly.

2. **Relaxed** — degree constraints remain hard; the λ/μ common-neighbour
   conditions become a soft objective that *minimises the number of violated
   vertex pairs* via big-M indicators.  A solution with objective value 0 is
   a valid SRG.  Requires O(n³) auxiliary AND-product variables.

3. **Quadratic** — degree constraints remain hard; the objective directly
   minimises ``Σ_{x<z} (common(x,z) − target(x,z))²`` where the
   common-neighbour count ``Σ_y e(x,y)·e(y,z)`` is expressed as a native
   quadratic form.  No auxiliary p or v variables at all — just O(n²/2)
   continuous residual variables and O(n²/2) quadratic constraints.  Solved
   as a mixed-integer quadratically-constrained quadratic program (MIQCQP)
   with ``NonConvex=2``.  Objective value 0 certifies a valid SRG.

All formulations store edges in an upper-triangle dict
``edges[(i, j)]`` for ``i < j`` and expose an accessor ``e(i, j)``
that handles ordering and the diagonal.

Optimisations (exact and relaxed):

* **p-variable symmetry** — In an undirected graph ``p[x, y, z]`` and
  ``p[z, y, x]`` are identical (both encode x~y ∧ y~z = z~y ∧ y~x).
  We only create ``p[x, y, z]`` for ``x < z``, halving the AND-product
  variable count from ~n³ to ~n³/2.
* **No degenerate variables** — We skip creating p variables entirely when
  any two of x, y, z coincide, rather than creating fixed-to-zero dummies.
* **v-variable symmetry** — In the relaxed formulation, violation indicator
  ``v[x, z]`` is only created for ``x < z`` (unordered pairs).
* **Tighter big-M** — Uses ``k`` instead of ``n − 2`` since k-regularity
  is a hard constraint bounding the maximum common-neighbour count.

Symmetry-breaking options (undirected only):

* **Neighbour fixing** — pin the neighbours of vertex 0 to the *last* k
  vertices ``{n−k, …, n−1}``, and simultaneously fix vertex 1's neighbours
  to ``{n−2k+μ, …, n−k−1} ∪ {n−μ, …, n−1}``.  Placing all fixed entries at
  the *high* end of each column-0 sub-block keeps every fixed consecutive
  row pair lex-sorted, so neighbour fixing and lex ordering are fully
  compatible with ``start_row=1``.
* **Lex ordering** — enforce lexicographic row ordering via one of the
  strategies in :mod:`models.symmetry` (``"exponential"`` or
  ``"lex_leader"``).  Can be combined with neighbour fixing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import gurobipy as gp
from gurobipy import GRB

if TYPE_CHECKING:
    from collections.abc import Callable

from .symmetry import add_lex_order

# ---------------------------------------------------------------------------
# Public type alias for the lex-order parameter
# ---------------------------------------------------------------------------

LexOrder = Literal["none", "exponential", "lex_leader", "hybrid"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_edge_accessor(
    edges: dict[tuple[int, int], gp.Var],
) -> Callable[[int, int], gp.Var | int]:
    """Return a callable ``e(i, j)`` that looks up the undirected edge var.

    Returns 0 (the integer) when ``i == j`` (diagonal / loopless).
    """

    def _e(i: int, j: int) -> gp.Var | int:
        if i == j:
            return 0
        return edges[min(i, j), max(i, j)]

    return _e


def _extract_status(model: gp.Model) -> str:
    return {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
    }.get(model.Status, f"Unknown({model.Status})")


def _add_edges(
    model: gp.Model,
    n: int,
    k: int,
    mu: int,
    *,
    fix_neighbors: bool,
) -> tuple[dict[tuple[int, int], gp.Var], Callable[[int, int], gp.Var | int]]:
    """Create upper-triangle binary edge variables and return ``(edges, e)``.

    Symmetry-breaking via neighbour fixing (``fix_neighbors=True``):

    * Vertex 0's neighbours are pinned to the **last** k vertices
      ``{n−k, …, n−1}``.  This places neighbour rows (which start with 1 in
      column 0) *after* non-neighbour rows (which start with 0), so the fixed
      structure is already lex-sorted and compatible with lex-ordering
      constraints.

    * Vertex 1's neighbours are simultaneously pinned to
      ``{n−2k+μ, …, n−k−1} ∪ {n−μ, …, n−1}``:

      - k−μ additional neighbours from non-neighbours of v0:
        the **last** k−μ in that range, ``{n−2k+μ, …, n−k−1}``
      - μ common neighbours with v0:
        the **last** μ of v0's neighbours, ``{n−μ, …, n−1}``

      Placing v1's neighbours at the *high* end of each block keeps every
      fixed consecutive row pair lex-sorted (column-1 goes 0→1 at the
      sub-block boundary), so neighbour fixing and lex ordering are
      compatible with ``start_row=1``.
    """
    edges: dict[tuple[int, int], gp.Var] = {}

    # v1's fixed neighbours (size k):
    #   k-μ from non-neighbours of v0: {n-2k+μ, …, n-k-1}  (last k-μ in non-nbr range)
    #   μ   from neighbours of v0:     {n-μ, …, n-1}        (last μ in nbr range)
    # Placing them at the high end of each block keeps every fixed pair of
    # consecutive rows lex-sorted: within each block, the 0s (not-v1-neighbour)
    # come before the 1s (v1-neighbour), so column-1 goes 0→1 at the boundary,
    # which the exponential/lex-leader constraints can accept.
    v1_neighbors: set[int] = (
        set(range(n - 2 * k + mu, n - k)) | set(range(n - mu, n))
    ) if fix_neighbors else set()

    for i in range(n):
        for j in range(i + 1, n):
            if fix_neighbors and i == 0:
                # v0 neighbours = {n-k, …, n-1}.
                # Use INTEGER type with fixed value to avoid Gurobi issues
                # with lb==ub on BINARY variables.
                val = 1.0 if j >= n - k else 0.0
                edges[i, j] = model.addVar(
                    lb=val, ub=val, vtype=GRB.INTEGER, name=f"e_{i}_{j}",
                )
            elif fix_neighbors and i == 1:
                # v1 neighbours among {2, …, n−1} (edge (0,1) handled above)
                val = 1.0 if j in v1_neighbors else 0.0
                edges[i, j] = model.addVar(
                    lb=val, ub=val, vtype=GRB.INTEGER, name=f"e_{i}_{j}",
                )
            else:
                edges[i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )
    return edges, _make_edge_accessor(edges)


def _add_degree_constraints(
    model: gp.Model,
    e: Callable[[int, int], gp.Var | int],
    n: int,
    k: int,
) -> None:
    """Add k-regularity constraints for every vertex."""
    for x in range(n):
        model.addConstr(
            gp.quicksum(e(x, y) for y in range(n)) == k,
            name=f"deg_{x}",
        )


def _add_path_products(
    model: gp.Model,
    e: Callable[[int, int], gp.Var | int],
    n: int,
) -> dict[tuple[int, int, int], gp.Var]:
    """Add AND-product variables ``p[x, y, z] = e(x,y) ∧ e(y,z)``.

    Exploits undirected symmetry: only creates variables for ``x < z``
    (since ``p[x,y,z] == p[z,y,x]`` in an undirected graph).  Skips
    degenerate triples where any two indices coincide.

    The λ/μ constraints must use ``p[min(x,z), y, max(x,z)]`` to look up
    the correct variable.
    """
    p: dict[tuple[int, int, int], gp.Var] = {}
    for x in range(n):
        for z in range(x + 1, n):
            for y in range(n):
                if y == x or y == z:
                    continue
                pvar = model.addVar(vtype=GRB.BINARY, name=f"p_{x}_{y}_{z}")
                # e(x, y) and e(y, z) are guaranteed to be Var (not int 0)
                # because x != y and y != z.
                exy: gp.Var = e(x, y)  # type: ignore[assignment]
                eyz: gp.Var = e(y, z)  # type: ignore[assignment]
                model.addGenConstrAnd(pvar, [exy, eyz], name=f"p_and_{x}_{y}_{z}")
                p[x, y, z] = pvar
    return p


def _common_neighbors_expr(
    p: dict[tuple[int, int, int], gp.Var],
    n: int,
    x: int,
    z: int,
) -> gp.LinExpr:
    """Return ``Σ_y p[x, y, z]`` using the canonical (x < z) key."""
    lo, hi = min(x, z), max(x, z)
    return gp.quicksum(p[lo, y, hi] for y in range(n) if y != lo and y != hi)


# ── Exact formulation ────────────────────────────────────────────────────────


def build_srg_exact(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors: bool = True,
    lex_order: LexOrder = "none",
    lex_block_size: int = 20,
    quiet: bool = True,
) -> tuple[gp.Model, dict[tuple[int, int], gp.Var], Callable]:
    """Build a Gurobi model for SRG(n, k, λ, μ) with all constraints hard.

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: Common neighbours for adjacent pairs.
        mu: Common neighbours for non-adjacent pairs.
        fix_neighbors: Pin neighbours of vertices 0 and 1 to high-index
            positions (see :func:`_add_edges`).  Compatible with *lex_order*.
        lex_order: Lex-ordering strategy (``"none"``, ``"exponential"``,
            ``"lex_leader"``).
        quiet: Suppress Gurobi console output.

    Returns:
        ``(model, edges, e)`` where *edges* is the upper-triangle variable
        dict and *e* is the accessor callable.
    """
    model = gp.Model(f"SRG_exact_{n}_{k}_{lambda_param}_{mu}")
    if quiet:
        model.setParam("OutputFlag", 0)

    edges, e = _add_edges(model, n, k, mu, fix_neighbors=fix_neighbors)

    # Feasibility problem — no objective.
    model.setObjective(0, GRB.MINIMIZE)

    _add_degree_constraints(model, e, n, k)
    p = _add_path_products(model, e, n)

    # ── λ/μ constraints ───────────────────────────────────────────────────
    # For each unordered pair {x, z} (x < z):
    #   |{y : x~y~z}| = λ  if x~z,  μ  otherwise.
    # Written as: Σ_y p[x,y,z] = μ + (λ−μ)·e(x,z)
    #
    # By undirected symmetry the constraint for ordered pair (x,z) and
    # (z,x) are identical, so we only need x < z.
    for x in range(n):
        for z in range(x + 1, n):
            model.addConstr(
                _common_neighbors_expr(p, n, x, z)
                == mu + e(x, z) * (lambda_param - mu),
                name=f"lm_{x}_{z}",
            )

    # ── Lex ordering (symmetry breaking) ──────────────────────────────────
    # When fix_neighbors=True, row 0 is fully determined and the lex
    # constraint row_0 ≤_lex row_1 is automatically satisfied (row_1 gains
    # a 1 at position n-2k+μ while row_0 is still 0 there), so start_row=1.
    add_lex_order(
        model, e, n,
        kind=lex_order,
        start_row=1 if fix_neighbors else 0,
        block_size=lex_block_size,
    )

    model.update()
    return model, edges, e


# ── Relaxed formulation (minimise violation count) ────────────────────────────


def build_srg_relaxed(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors: bool = True,
    lex_order: LexOrder = "none",
    lex_block_size: int = 20,
    quiet: bool = True,
) -> tuple[gp.Model, dict[tuple[int, int], gp.Var], Callable]:
    """Build a Gurobi model that minimises λ/μ violation count for SRG.

    Degree constraints are hard.  For each unordered pair {x, z} with
    x < z, a binary violation indicator ``v[x, z]`` is introduced:

    * ``v[x,z] = 0``  ⟹  the common-neighbour count equals the target
    * ``v[x,z] = 1``  ⟹  the constraint is relaxed (big-M)

    Objective: ``min Σ_{x<z} v[x, z]``.

    An optimal objective of 0 certifies a valid SRG.

    The big-M value is ``k`` — tight because k-regularity (a hard
    constraint) bounds the maximum number of common neighbours.

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: Target common neighbours for adjacent pairs.
        mu: Target common neighbours for non-adjacent pairs.
        fix_neighbors: Pin neighbours of vertices 0 and 1 (see
            :func:`_add_edges`).  Compatible with *lex_order*.
        lex_order: Lex-ordering strategy.
        quiet: Suppress Gurobi console output.

    Returns:
        ``(model, edges, e)`` — same interface as :func:`build_srg_exact`.
    """
    model = gp.Model(f"SRG_relaxed_{n}_{k}_{lambda_param}_{mu}")
    if quiet:
        model.setParam("OutputFlag", 0)

    edges, e = _add_edges(model, n, k, mu, fix_neighbors=fix_neighbors)

    _add_degree_constraints(model, e, n, k)
    p = _add_path_products(model, e, n)

    # ── Violation indicators + big-M relaxation of λ/μ ────────────────────
    # For each unordered pair {x, z} (x < z):
    #   common(x,z) = Σ_y p[x,y,z]
    #   target(x,z) = μ + (λ−μ)·e(x,z)
    #
    # When v[x,z] = 0:  common = target        (satisfied)
    # When v[x,z] = 1:  |common - target| ≤ M  (relaxed)
    #
    # M = k suffices: every vertex has degree exactly k, so the max
    # number of common neighbours for any pair is at most k.
    big_m = k

    v: dict[tuple[int, int], gp.Var] = {}
    for x in range(n):
        for z in range(x + 1, n):
            v[x, z] = model.addVar(vtype=GRB.BINARY, name=f"v_{x}_{z}")

            common_xz = _common_neighbors_expr(p, n, x, z)
            target_xz = mu + e(x, z) * (lambda_param - mu)

            model.addConstr(
                common_xz - target_xz <= big_m * v[x, z],
                name=f"viol_hi_{x}_{z}",
            )
            model.addConstr(
                target_xz - common_xz <= big_m * v[x, z],
                name=f"viol_lo_{x}_{z}",
            )

    # ── Objective: minimise total violations (unordered pairs) ────────────
    model.setObjective(
        gp.quicksum(v.values()),
        GRB.MINIMIZE,
    )

    # ── Lex ordering (symmetry breaking) ──────────────────────────────────
    add_lex_order(
        model, e, n,
        kind=lex_order,
        start_row=1 if fix_neighbors else 0,
        block_size=lex_block_size,
    )

    model.update()
    return model, edges, e


# ── Quadratic formulation (MIQCQP, no auxiliary variables) ────────────────────

Formulation = Literal["exact", "relaxed", "quadratic"]


def build_srg_quadratic(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors: bool = True,
    lex_order: LexOrder = "none",
    lex_block_size: int = 20,
    quiet: bool = True,
) -> tuple[gp.Model, dict[tuple[int, int], gp.Var], Callable]:
    """Build a MIQCQP that minimises sum of squared λ/μ residuals for SRG.

    Degree constraints are hard.  For each unordered pair {x, z} with x < z,
    a continuous residual variable ``s[x, z]`` is defined by the quadratic
    constraint::

        s[x, z] = Σ_y e(x,y)·e(y,z) − (λ−μ)·e(x,z) − μ

    The bilinear products ``e(x,y)·e(y,z)`` are handled natively by Gurobi
    (``NonConvex=2``), so **no auxiliary AND-product or violation-indicator
    variables** are needed.

    Objective: ``min Σ_{x<z} s[x, z]²``.

    An optimal objective of 0 certifies a valid SRG.

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: Target common neighbours for adjacent pairs.
        mu: Target common neighbours for non-adjacent pairs.
        fix_neighbors: Pin neighbours of vertices 0 and 1 (see
            :func:`_add_edges`).  Compatible with *lex_order*.
        lex_order: Lex-ordering strategy.
        quiet: Suppress Gurobi console output.

    Returns:
        ``(model, edges, e)`` — same interface as :func:`build_srg_exact`.
    """
    model = gp.Model(f"SRG_quadratic_{n}_{k}_{lambda_param}_{mu}")
    if quiet:
        model.setParam("OutputFlag", 0)

    # Bilinear products in the residual constraints require this.
    model.setParam("NonConvex", 2)

    edges, e = _add_edges(model, n, k, mu, fix_neighbors=fix_neighbors)
    _add_degree_constraints(model, e, n, k)

    lam_minus_mu = lambda_param - mu

    # ── Residual variables + quadratic constraints ────────────────────────
    # For each unordered pair {x, z} (x < z):
    #   s[x,z] = (A²)_{xz} − (λ−μ)·A_{xz} − μ
    #          = Σ_y e(x,y)·e(y,z) − (λ−μ)·e(x,z) − μ
    #
    # Bounds: (A²)_{xz} ∈ [0, k], and (λ−μ)·e ∈ {0, λ−μ}, so:
    s_lb = -abs(lam_minus_mu) - mu
    s_ub = k - min(0, lam_minus_mu) - mu

    s: dict[tuple[int, int], gp.Var] = {}
    for x in range(n):
        for z in range(x + 1, n):
            s[x, z] = model.addVar(
                lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name=f"s_{x}_{z}",
            )

            # Quadratic constraint: s = Σ_y e(x,y)*e(y,z) - (λ-μ)*e(x,z) - μ
            #
            # Build the bilinear sum over intermediate vertices y.
            # e(x, y) and e(y, z) return 0 when y == x or y == z (diagonal),
            # so including those terms is harmless but we skip them for clarity.
            bilinear_sum = gp.QuadExpr()
            for y in range(n):
                if y == x or y == z:
                    continue
                exy = e(x, y)
                eyz = e(y, z)
                bilinear_sum += exy * eyz  # type: ignore[arg-type]

            model.addConstr(
                s[x, z] == bilinear_sum - lam_minus_mu * e(x, z) - mu,
                name=f"res_{x}_{z}",
            )

    # ── Objective: minimise Σ_{x<z} s[x, z]² ─────────────────────────────
    model.setObjective(
        gp.quicksum(s[x, z] * s[x, z] for x, z in s),
        GRB.MINIMIZE,
    )

    # ── Lex ordering (symmetry breaking) ──────────────────────────────────
    add_lex_order(
        model, e, n,
        kind=lex_order,
        start_row=1 if fix_neighbors else 0,
        block_size=lex_block_size,
    )

    model.update()
    return model, edges, e


# ── Solve wrapper ─────────────────────────────────────────────────────────────

_BUILDERS = {
    "exact": build_srg_exact,
    "relaxed": build_srg_relaxed,
    "quadratic": build_srg_quadratic,
}


def solve_srg(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    formulation: Formulation = "exact",
    fix_neighbors: bool = True,
    lex_order: LexOrder = "none",
    lex_block_size: int = 20,
    threads: int = -1,
    time_limit: float | None = None,
    heuristics: float | None = None,
    quiet: bool = False,
    gurobi_params: dict[str, str] | None = None,
) -> dict:
    """Build, solve, and return a results dict for SRG(n, k, λ, μ).

    Args:
        n: Number of vertices.
        k: Degree.
        lambda_param: λ parameter.
        mu: μ parameter.
        formulation: ``"exact"``, ``"relaxed"`` (violation count), or
            ``"quadratic"`` (sum of squared residuals).
        fix_neighbors: Pin neighbours of vertices 0 and 1 to high-index
            positions (see :func:`_add_edges`).  Compatible with *lex_order*.
        lex_order: Lex-ordering strategy.
        threads: Solver threads (-1 = Gurobi default / all).
        time_limit: Wall-clock limit in seconds.
        heuristics: Fraction of solve time on MIP heuristics (0.0–1.0).
            ``None`` uses the Gurobi default (0.05).
        quiet: Suppress Gurobi output.

    Returns:
        Dict with keys: ``status``, ``wall_seconds``, ``n``, ``k``,
        ``lambda``, ``mu``, and optionally ``adjacency``, ``obj_val``.
    """
    builder = _BUILDERS[formulation]
    model, edges, e = builder(
        n, k, lambda_param, mu,
        fix_neighbors=fix_neighbors,
        lex_order=lex_order,
        lex_block_size=lex_block_size,
        quiet=quiet,
    )

    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if heuristics is not None:
        model.setParam("Heuristics", heuristics)
    if formulation in ("exact", "relaxed"):
        model.setParam("MIPFocus", 1)

    # Apply arbitrary Gurobi parameters.
    for key, val in (gurobi_params or {}).items():
        # Auto-convert to int or float if possible.
        for conv in (int, float):
            try:
                val = conv(val)  # type: ignore[assignment]
                break
            except ValueError:
                continue
        model.setParam(key, val)

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    status = _extract_status(model)

    adjacency = None
    obj_val = None
    if model.SolCount > 0:
        if formulation == "relaxed":
            obj_val = round(model.ObjVal)
        elif formulation == "quadratic":
            obj_val = model.ObjVal
        adjacency = [
            [
                int(round(e(i, j).X)) if isinstance(e(i, j), gp.Var) else 0
                for j in range(n)
            ]
            for i in range(n)
        ]

    return {
        "status": status,
        "wall_seconds": round(elapsed, 4),
        "obj_val": obj_val,
        "n": n,
        "k": k,
        "lambda": lambda_param,
        "mu": mu,
        "adjacency": adjacency,
    }
