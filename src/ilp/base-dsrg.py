"""Directed Strongly Regular Graph (DSRG) ILP formulation using native gurobipy.

A DSRG(n, k, t, λ, μ) is a directed graph on n vertices where every vertex
has in- and out-degree k, exactly t reciprocal neighbours, and for any two
distinct vertices x, z the number of directed paths x→y→z equals λ if x→z
and μ otherwise.

The model uses Gurobi's addGenConstrAnd for the two product linearisations,
which lets Gurobi exploit the AND structure directly rather than relying on
Big-M relaxations.
"""

import time

import gurobipy as gp
from gurobipy import GRB


def build_dsrg_model(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    fix_out_neighbors_of_zero: bool = True,
    lex_order: bool = True,
    quiet: bool = True,
) -> tuple[gp.Model, dict]:
    """Build and return a Gurobi model for DSRG(n, k, t, lambda_param, mu).

    Args:
        n: Number of vertices.
        k: In- and out-degree of every vertex.
        t: Number of reciprocal neighbours per vertex.
        lambda_param: Number of directed 2-paths x→y→z when x→z exists.
        mu: Number of directed 2-paths x→y→z when x→z does not exist.
        fix_out_neighbors_of_zero: Symmetry break that pins the out-neighbours
            of vertex 0 to {1, 2, …, k}.
        lex_order: Enforce lexicographic row ordering as symmetry breaking.
        quiet: Suppress Gurobi console output.

    Returns:
        (model, edges) where edges[(i, j)] is the binary variable for arc i→j.
    """
    model = gp.Model("DSRG")
    if quiet:
        model.setParam("OutputFlag", 0)

    # ── Edge variables ────────────────────────────────────────────────────────
    # edges[i, j] = 1  iff there is a directed edge i → j.
    # Diagonal is always 0 (loopless).  Symmetry-break on vertex 0's out-nbrs.
    edges: dict[tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                # Fixed to 0: loopless graph.
                edges[i, j] = model.addVar(
                    lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"e_{i}_{j}"
                )
            elif fix_out_neighbors_of_zero and i == 0:
                # Symmetry break: out-nbrs of vertex 0 are exactly {1, …, k}.
                val = 1.0 if 1 <= j <= k else 0.0
                edges[i, j] = model.addVar(
                    lb=val, ub=val, vtype=GRB.BINARY, name=f"e_{i}_{j}"
                )
            else:
                edges[i, j] = model.addVar(vtype=GRB.BINARY, name=f"e_{i}_{j}")

    # Dummy objective — we just want a feasible point.
    model.setObjective(
        gp.quicksum(edges[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE,
    )

    # ── Degree constraints ────────────────────────────────────────────────────
    for x in range(n):
        model.addConstr(
            gp.quicksum(edges[x, j] for j in range(n)) == k,
            name=f"out_deg_{x}",
        )
        model.addConstr(
            gp.quicksum(edges[i, x] for i in range(n)) == k,
            name=f"in_deg_{x}",
        )

    # ── Auxiliary p[x, y, z]: 1 iff x→y and y→z ─────────────────────────────
    # Degenerate cases (x==y or y==z) are fixed to 0.
    p: dict[tuple[int, int, int], gp.Var] = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if x == y or y == z:
                    p[x, y, z] = model.addVar(
                        lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"p_{x}_{y}_{z}"
                    )
                else:
                    p[x, y, z] = model.addVar(
                        vtype=GRB.BINARY, name=f"p_{x}_{y}_{z}"
                    )
                    model.addGenConstrAnd(
                        p[x, y, z],
                        [edges[x, y], edges[y, z]],
                        name=f"p_and_{x}_{y}_{z}",
                    )

    # ── Auxiliary r[x, y]: 1 iff x→y and y→x (reciprocal arc) ───────────────
    r: dict[tuple[int, int], gp.Var] = {}
    for x in range(n):
        for y in range(n):
            if x == y:
                r[x, y] = model.addVar(
                    lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"r_{x}_{y}"
                )
            else:
                r[x, y] = model.addVar(vtype=GRB.BINARY, name=f"r_{x}_{y}")
                model.addGenConstrAnd(
                    r[x, y],
                    [edges[x, y], edges[y, x]],
                    name=f"r_and_{x}_{y}",
                )

    # ── t constraint: each vertex has exactly t reciprocal neighbours ─────────
    for x in range(n):
        model.addConstr(
            gp.quicksum(r[x, y] for y in range(n) if y != x) == t,
            name=f"recip_{x}",
        )

    # ── λ/μ constraint (Duval's definition) ──────────────────────────────────
    # For x ≠ z: |{y : x→y→z}| = λ  if x→z,  μ  otherwise.
    # Written as a single linear constraint:
    #   sum_y p[x,y,z] = μ + e[x,z] * (λ - μ)
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            model.addConstr(
                gp.quicksum(p[x, y, z] for y in range(n))
                == mu + edges[x, z] * (lambda_param - mu),
                name=f"lm_{x}_{z}",
            )

    # ── Lexicographic row ordering (symmetry breaking) ─────────────────────
    # For consecutive rows i, i+1: enforce row_i ≤_lex row_{i+1}.
    # agree[j] = 1 iff rows i and i+1 match on all columns up to and including j.
    # At the first disagreement column j, row i must have 0 (row i+1 has 1).
    if lex_order:
        start_row = 1 if fix_out_neighbors_of_zero else 0
        for i in range(start_row, n - 1):
            cols = [j for j in range(n) if j != i and j != i + 1]
            agree: dict[int, gp.Var] = {}
            for idx, j in enumerate(cols):
                agree[j] = model.addVar(vtype=GRB.BINARY, name=f"agree_{i}_{j}")
                # agree[j] = 1 → e[i,j] == e[i+1,j] (and all prior columns agreed)
                # Linking: agree[j] ≤ 1 - (e[i,j] - e[i+1,j])
                #          agree[j] ≤ 1 + (e[i,j] - e[i+1,j])
                # Plus chaining: agree[j] ≤ agree[prev_j] (if not first column)
                diff = edges[i, j] - edges[i + 1, j]
                model.addConstr(
                    agree[j] <= 1 - diff, name=f"agree_ub1_{i}_{j}"
                )
                model.addConstr(
                    agree[j] <= 1 + diff, name=f"agree_ub2_{i}_{j}"
                )
                if idx > 0:
                    prev_j = cols[idx - 1]
                    model.addConstr(
                        agree[j] <= agree[prev_j], name=f"agree_chain_{i}_{j}"
                    )

            # At each column j: if all prior columns agreed (agree[prev] = 1)
            # and this is the first disagreement, then e[i,j] must be 0.
            # Equivalently: e[i,j] ≤ 1 - agree[prev] + agree[j]
            for idx, j in enumerate(cols):
                if idx == 0:
                    model.addConstr(
                        edges[i, j] <= agree[j], name=f"lex_first_{i}_{j}"
                    )
                else:
                    prev_j = cols[idx - 1]
                    model.addConstr(
                        edges[i, j] <= 1 - agree[prev_j] + agree[j],
                        name=f"lex_{i}_{j}",
                    )

            # Rows must not be identical.
            model.addConstr(
                gp.quicksum(agree[j] for j in cols) <= len(cols) - 1,
                name=f"lex_neq_{i}",
            )

    model.update()
    return model, edges


def solve_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    fix_out_neighbors_of_zero: bool = True,
    lex_order: bool = True,
    threads: int = -1,
    time_limit: float | None = None,
    quiet: bool = False,
) -> dict:
    """Build, solve, and return a results dict for DSRG(n, k, t, λ, μ).

    Args:
        n: Number of vertices.
        k: In- and out-degree.
        t: Reciprocal neighbours per vertex.
        lambda_param: λ parameter.
        mu: μ parameter.
        fix_out_neighbors_of_zero: Enable symmetry breaking on vertex 0.
        threads: Number of solver threads (-1 = all available).
        time_limit: Optional wall-clock limit in seconds.
        quiet: Suppress Gurobi output.

    Returns:
        Dict with keys: status, wall_seconds, adjacency (list[list[int]] or None).
    """
    model, edges = build_dsrg_model(
        n, k, t, lambda_param, mu,
        fix_out_neighbors_of_zero=fix_out_neighbors_of_zero,
        lex_order=lex_order,
        quiet=quiet,
    )

    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    status = {
        GRB.OPTIMAL:    "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
    }.get(model.Status, f"Unknown({model.Status})")

    adjacency = None
    if model.SolCount > 0:
        adjacency = [
            [int(round(edges[i, j].X)) for j in range(n)]
            for i in range(n)
        ]

    return {
        "status":       status,
        "wall_seconds": round(elapsed, 4),
        "n": n, "k": k, "t": t, "lambda": lambda_param, "mu": mu,
        "adjacency":    adjacency,
    }


if __name__ == "__main__":
    # Reproduce the open case from the pulp version comment
    n = 16
    k = 8
    t = 6
    lambda_param = 2
    mu = 6
    params = dict(n=n, k=k, t=t, lambda_param=lambda_param, mu=mu)
    print(f"Solving DSRG({params['n']}, {params['k']}, {params['t']}, "
          f"{params['lambda_param']}, {params['mu']}) ...")

    result = solve_dsrg(**params, threads=-1)
    print(f"Status:      {result['status']}")
    print(f"Wall time:   {result['wall_seconds']:.3f}s")

    if result["adjacency"] is not None:
        print("Adjacency matrix:")
        for row in result["adjacency"]:
            print(row)
