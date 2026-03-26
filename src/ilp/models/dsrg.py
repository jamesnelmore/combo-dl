"""Directed Strongly Regular Graph (DSRG) ILP formulations using native gurobipy.

A DSRG(n, k, t, λ, μ) is a directed graph on n vertices where every vertex
has in- and out-degree k, exactly t reciprocal neighbours, and for any two
distinct vertices x, z the number of directed paths x→y→z equals λ if x→z
and μ otherwise.

Two formulations are provided:

1. **Exact** — all constraints are hard.  The solver either finds a feasible
   DSRG or proves infeasibility.  Uses ``addGenConstrAnd`` for product
   linearisation so Gurobi can exploit the AND structure directly.

2. **Relaxed** — degree and reciprocal-count (t) constraints remain hard;
   the λ/μ path-count conditions become a soft objective that *minimises
   the number of violated vertex pairs*.  A solution with objective value 0
   is a valid DSRG.

Edge variables are stored in a full ``n × n`` dict ``edges[(i, j)]`` (with
diagonal entries fixed to 0 for loopless graphs).

Symmetry-breaking options (directed):

* **Neighbour fixing** — pin the out-neighbours of vertex 0 to {1, …, k}.

Lex ordering is not applied for directed graphs (per convention — the row
ordering in a directed adjacency matrix does not have the same symmetry
interpretation as in the undirected case).
"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_status(model: gp.Model) -> str:
    return {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
    }.get(model.Status, f"Unknown({model.Status})")


# ── Shared variable / constraint builders ────────────────────────────────────


def _add_edge_vars(
    model: gp.Model,
    n: int,
    k: int,
    *,
    fix_neighbors: bool,
) -> dict[tuple[int, int], gp.Var]:
    """Add n×n binary edge variables to *model*.

    Diagonal entries are fixed to 0 (loopless).  When *fix_neighbors* is
    ``True``, the out-neighbours of vertex 0 are pinned to {1, …, k}.
    """
    edges: dict[tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                edges[i, j] = model.addVar(
                    lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"e_{i}_{j}",
                )
            elif fix_neighbors and i == 0:
                val = 1.0 if 1 <= j <= k else 0.0
                edges[i, j] = model.addVar(
                    lb=val, ub=val, vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )
            else:
                edges[i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )
    return edges


def _add_degree_constraints(
    model: gp.Model,
    edges: dict[tuple[int, int], gp.Var],
    n: int,
    k: int,
) -> None:
    """Add in-degree and out-degree regularity constraints."""
    for x in range(n):
        model.addConstr(
            gp.quicksum(edges[x, j] for j in range(n)) == k,
            name=f"out_deg_{x}",
        )
        model.addConstr(
            gp.quicksum(edges[i, x] for i in range(n)) == k,
            name=f"in_deg_{x}",
        )


def _add_path_products(
    model: gp.Model,
    edges: dict[tuple[int, int], gp.Var],
    n: int,
) -> dict[tuple[int, int, int], gp.Var]:
    """Add auxiliary variables p[x,y,z] = 1 iff x→y and y→z.

    Uses ``addGenConstrAnd`` so Gurobi exploits the AND structure.
    Degenerate cases (x==y or y==z) are fixed to 0.
    """
    p: dict[tuple[int, int, int], gp.Var] = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if x == y or y == z:
                    p[x, y, z] = model.addVar(
                        lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS,
                        name=f"p_{x}_{y}_{z}",
                    )
                else:
                    p[x, y, z] = model.addVar(
                        vtype=GRB.BINARY, name=f"p_{x}_{y}_{z}",
                    )
                    model.addGenConstrAnd(
                        p[x, y, z],
                        [edges[x, y], edges[y, z]],
                        name=f"p_and_{x}_{y}_{z}",
                    )
    return p


def _add_reciprocal_vars_and_constraints(
    model: gp.Model,
    edges: dict[tuple[int, int], gp.Var],
    n: int,
    t: int,
) -> dict[tuple[int, int], gp.Var]:
    """Add reciprocal-arc variables r[x,y] and the t-constraint.

    r[x,y] = 1 iff x→y and y→x.  Each vertex has exactly t reciprocal
    neighbours (hard constraint).
    """
    r: dict[tuple[int, int], gp.Var] = {}
    for x in range(n):
        for y in range(n):
            if x == y:
                r[x, y] = model.addVar(
                    lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"r_{x}_{y}",
                )
            else:
                r[x, y] = model.addVar(vtype=GRB.BINARY, name=f"r_{x}_{y}")
                model.addGenConstrAnd(
                    r[x, y],
                    [edges[x, y], edges[y, x]],
                    name=f"r_and_{x}_{y}",
                )

    for x in range(n):
        model.addConstr(
            gp.quicksum(r[x, y] for y in range(n) if y != x) == t,
            name=f"recip_{x}",
        )

    return r


# ── Exact formulation ────────────────────────────────────────────────────────


def build_dsrg_exact(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors: bool = True,
    quiet: bool = True,
) -> tuple[gp.Model, dict[tuple[int, int], gp.Var]]:
    """Build a Gurobi model for DSRG(n, k, t, λ, μ) with all constraints hard.

    Args:
        n: Number of vertices.
        k: In- and out-degree of every vertex.
        t: Number of reciprocal neighbours per vertex.
        lambda_param: Number of directed 2-paths x→y→z when x→z exists.
        mu: Number of directed 2-paths x→y→z when x→z does not exist.
        fix_neighbors: Pin out-neighbours of vertex 0 to {1, …, k}.
        quiet: Suppress Gurobi console output.

    Returns:
        ``(model, edges)`` where ``edges[(i, j)]`` is the binary variable
        for arc i→j.
    """
    model = gp.Model(f"DSRG_exact_{n}_{k}_{t}_{lambda_param}_{mu}")
    if quiet:
        model.setParam("OutputFlag", 0)

    edges = _add_edge_vars(model, n, k, fix_neighbors=fix_neighbors)

    # Dummy objective — feasibility problem.
    model.setObjective(0, GRB.MINIMIZE)

    _add_degree_constraints(model, edges, n, k)
    p = _add_path_products(model, edges, n)
    _add_reciprocal_vars_and_constraints(model, edges, n, t)

    # ── λ/μ constraint (Duval's definition) ───────────────────────────────
    # For x ≠ z: |{y : x→y→z}| = λ if x→z, μ otherwise.
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            model.addConstr(
                gp.quicksum(p[x, y, z] for y in range(n))
                == mu + edges[x, z] * (lambda_param - mu),
                name=f"lm_{x}_{z}",
            )

    model.update()
    return model, edges


# ── Relaxed formulation (minimise violation count) ────────────────────────────


def build_dsrg_relaxed(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors: bool = True,
    quiet: bool = True,
) -> tuple[gp.Model, dict[tuple[int, int], gp.Var]]:
    """Build a Gurobi model that minimises λ/μ violation count for DSRG.

    Degree and reciprocal-count (t) constraints are hard.  For each
    off-diagonal pair (x, z) an indicator ``v[x,z] ∈ {0,1}`` is introduced:

    * ``v[x,z] = 0``  ⟹  the path-count equals the target
    * ``v[x,z] = 1``  ⟹  the constraint is relaxed (big-M)

    Objective: ``min Σ_{x≠z} v[x,z]``.

    An optimal objective of 0 certifies a valid DSRG.

    Args:
        n: Number of vertices.
        k: In- and out-degree.
        t: Reciprocal neighbours per vertex.
        lambda_param: Target path count for arc-present pairs.
        mu: Target path count for arc-absent pairs.
        fix_neighbors: Pin out-neighbours of vertex 0.
        quiet: Suppress Gurobi console output.

    Returns:
        ``(model, edges)`` — same interface as :func:`build_dsrg_exact`.
    """
    model = gp.Model(f"DSRG_relaxed_{n}_{k}_{t}_{lambda_param}_{mu}")
    if quiet:
        model.setParam("OutputFlag", 0)

    edges = _add_edge_vars(model, n, k, fix_neighbors=fix_neighbors)

    _add_degree_constraints(model, edges, n, k)
    p = _add_path_products(model, edges, n)
    _add_reciprocal_vars_and_constraints(model, edges, n, t)

    # ── Violation indicators + big-M relaxation of λ/μ ────────────────────
    # For x ≠ z:  common(x,z) = Σ_y p[x,y,z]
    #             target(x,z) = μ + (λ−μ)·e[x,z]
    #
    # When v[x,z] = 0: common = target  (satisfied)
    # When v[x,z] = 1: |common - target| <= M  (relaxed)
    #
    # M = n − 2 suffices (max possible directed 2-paths between any pair).
    big_m = n - 2

    v: dict[tuple[int, int], gp.Var] = {}
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            v[x, z] = model.addVar(vtype=GRB.BINARY, name=f"v_{x}_{z}")

            common_xz = gp.quicksum(p[x, y, z] for y in range(n))
            target_xz = mu + edges[x, z] * (lambda_param - mu)

            model.addConstr(
                common_xz - target_xz <= big_m * v[x, z],
                name=f"viol_hi_{x}_{z}",
            )
            model.addConstr(
                target_xz - common_xz <= big_m * v[x, z],
                name=f"viol_lo_{x}_{z}",
            )

    # ── Objective: minimise total violations ──────────────────────────────
    model.setObjective(
        gp.quicksum(v[x, z] for x in range(n) for z in range(n) if x != z),
        GRB.MINIMIZE,
    )

    model.update()
    return model, edges


# ── Solve wrapper ─────────────────────────────────────────────────────────────


def solve_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    relaxed: bool = False,
    fix_neighbors: bool = True,
    threads: int = -1,
    time_limit: float | None = None,
    heuristics: float | None = None,
    quiet: bool = False,
    gurobi_params: dict[str, str] | None = None,
) -> dict:
    """Build, solve, and return a results dict for DSRG(n, k, t, λ, μ).

    Args:
        n: Number of vertices.
        k: In- and out-degree.
        t: Reciprocal neighbours per vertex.
        lambda_param: λ parameter.
        mu: μ parameter.
        relaxed: Use the violation-count relaxed formulation.
        fix_neighbors: Pin out-neighbours of vertex 0.
        threads: Solver threads (-1 = Gurobi default / all).
        time_limit: Wall-clock limit in seconds.
        heuristics: Fraction of solve time on MIP heuristics (0.0–1.0).
            ``None`` uses the Gurobi default (0.05).
        quiet: Suppress Gurobi output.

    Returns:
        Dict with keys: ``status``, ``wall_seconds``, ``n``, ``k``, ``t``,
        ``lambda``, ``mu``, and optionally ``adjacency``, ``obj_val``.
    """
    if relaxed:
        model, edges = build_dsrg_relaxed(
            n, k, t, lambda_param, mu,
            fix_neighbors=fix_neighbors,
            quiet=quiet,
        )
    else:
        model, edges = build_dsrg_exact(
            n, k, t, lambda_param, mu,
            fix_neighbors=fix_neighbors,
            quiet=quiet,
        )

    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if heuristics is not None:
        model.setParam("Heuristics", heuristics)
    if not relaxed:
        model.setParam("MIPFocus", 1)

    # Apply arbitrary Gurobi parameters.
    for key, val in (gurobi_params or {}).items():
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
        if relaxed:
            obj_val = round(model.ObjVal)
        adjacency = [
            [int(round(edges[i, j].X)) for j in range(n)]
            for i in range(n)
        ]

    return {
        "status": status,
        "wall_seconds": round(elapsed, 4),
        "obj_val": obj_val,
        "n": n,
        "k": k,
        "t": t,
        "lambda": lambda_param,
        "mu": mu,
        "adjacency": adjacency,
    }
