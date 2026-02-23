"""Penalty-relaxation model for Directed Strongly Regular Graphs (DSRGs).

A DSRG(n, k, t, λ, μ) is a directed graph on n vertices where every vertex
has in- and out-degree k, exactly t reciprocal neighbours, and for any two
distinct vertices x, z the number of directed paths x→y→z equals λ if x→z
and μ otherwise.

This formulation relaxes e_ij ∈ {0,1} to e_ij ∈ [0,1], keeps degree
regularity as hard constraints, and replaces the quadratic DSRG constraints
(reciprocal count t and path-count λ/μ) with a soft penalty objective:

    min Σ_{i,j} ((A²)_{ij} - T_{ij})²

where T_{ij} = t·δ_{ij} + (λ−μ)·e_{ij} + μ·(1−δ_{ij}).

A solution with objective value 0 is a valid DSRG.  The objective is a
degree-4 polynomial over the continuous hypercube [0,1]^{n(n-1)}, solved
via Gurobi's non-convex QCQP support (NonConvex=2).

Uses Gurobi's MVar class for efficient matrix-based formulation.
"""

import time

import numpy as np
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
    quiet: bool = False,
) -> tuple[gp.Model, gp.MVar]:
    """Build and return a Gurobi penalty-relaxation model for DSRG(n, k, t, λ, μ).

    Args:
        n: Number of vertices.
        k: In- and out-degree of every vertex.
        t: Number of reciprocal neighbours per vertex.
        lambda_param: Number of directed 2-paths x→y→z when x→z exists.
        mu: Number of directed 2-paths x→y→z when x→z does not exist.
        fix_out_neighbors_of_zero: Symmetry break that pins the out-neighbours
            of vertex 0 to {1, 2, …, k}.
        quiet: Suppress Gurobi console output.

    Returns:
        (model, A) where A is the n×n MVar adjacency matrix with continuous
        entries in [0,1].
    """
    model = gp.Model("DSRG_relaxed")
    if quiet:
        model.setParam("OutputFlag", 0)

    # Non-convex quadratic support is required for degree-4 objective.
    model.setParam("NonConvex", 2)

    # ── Edge variables as an n×n MVar ─────────────────────────────────────────
    # A[i,j] ∈ [0,1] continuous.  Diagonal forced to 0 (loopless).
    lb = np.zeros((n, n))
    ub = np.ones((n, n))

    # Diagonal: loopless → fix to 0.
    for i in range(n):
        ub[i, i] = 0.0

    # Symmetry break: out-neighbours of vertex 0 are exactly {1, …, k}.
    if fix_out_neighbors_of_zero:
        for j in range(n):
            if j == 0:
                continue
            val = 1.0 if 1 <= j <= k else 0.0
            lb[0, j] = val
            ub[0, j] = val

    A = model.addMVar((n, n), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="e")

    # ── Degree constraints (hard) ─────────────────────────────────────────────
    ones = np.ones(n)
    for x in range(n):
        # out-degree: sum of row x == k
        model.addConstr(A[x, :] @ ones == k, name=f"out_deg_{x}")
        # in-degree: sum of column x == k
        model.addConstr(ones @ A[:, x] == k, name=f"in_deg_{x}")

    # ── Build target matrix T ─────────────────────────────────────────────────
    # T_{ij} = t·δ_{ij} + (λ−μ)·A_{ij} + μ·(1 − δ_{ij})
    #
    # Since A² + A should equal T for a valid DSRG, we want to minimize
    # Σ_{i,j} ((A²)_{ij} + A_{ij} - T_{ij})² = Σ_{i,j} ((A²)_{ij} - t·δ_{ij} - (λ−μ−1)·A_{ij} - μ·(1−δ_{ij}))²
    #
    # Actually, the DSRG condition is A² = T - A, but let's express it directly:
    # (A²)_{ij} = T_{ij} - A_{ij}  (since A² + A = T ... wait, let me re-derive)
    #
    # Duval's condition: for x ≠ z, Σ_y e_{xy}·e_{yz} = μ + (λ−μ)·e_{xz}
    # For x = z: Σ_y e_{xy}·e_{yx} = t  (reciprocal count)
    #
    # So (A²)_{xz} should equal:
    #   - μ + (λ−μ)·e_{xz}    when x ≠ z
    #   - t                    when x = z
    #
    # Which is T_{ij} = t·δ_{ij} + (λ−μ)·e_{ij} + μ·(1−δ_{ij})
    # And we want (A²)_{ij} = T_{ij}.

    # T has a constant part and a part proportional to A:
    #   T_{ij} = [t·δ_{ij} + μ·(1−δ_{ij})] + (λ−μ)·A_{ij}
    # Let C_{ij} = t·δ_{ij} + μ·(1−δ_{ij})  (constant matrix)
    C = np.full((n, n), float(mu))
    np.fill_diagonal(C, float(t))

    lam_minus_mu = lambda_param - mu

    # Residual R = A² - T = A² - (λ−μ)·A - C
    # We want to minimize Σ_{i,j} R_{ij}²

    # To handle the degree-4 objective, introduce auxiliary n×n variable matrix
    # S where S_{ij} = R_{ij} = (A²)_{ij} - (λ−μ)·A_{ij} - C_{ij}
    # Then constrain S = A² - (λ−μ)·A - C (quadratic constraints)
    # and minimize Σ S_{ij}².

    S = model.addMVar(
        (n, n), lb=-GRB.INFINITY, ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS, name="s",
    )

    # Constraint: S = A@A - (λ−μ)·A - C
    # i.e. for each (i,j): S_{ij} = Σ_k A_{ik}·A_{kj} - (λ−μ)·A_{ij} - C_{ij}
    for i in range(n):
        for j in range(n):
            model.addConstr(
                S[i, j] == A[i, :] @ A[:, j] - lam_minus_mu * A[i, j] - C[i, j],
                name=f"res_{i}_{j}",
            )

    # Objective: minimize Σ_{i,j} S_{ij}²
    # S.flatten() @ S.flatten() would give us Σ S_{ij}², but let's use reshape.
    s_flat = S.reshape(-1)
    model.setObjective(s_flat @ s_flat, GRB.MINIMIZE)

    model.update()
    return model, A


def verify_dsrg(
    adj: np.ndarray,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
) -> bool:
    """Check whether an integer adjacency matrix is a valid DSRG(n, k, t, λ, μ).

    Returns True iff all DSRG conditions hold exactly.
    """
    n = adj.shape[0]

    # Must be 0/1 with zero diagonal.
    if not np.all((adj == 0) | (adj == 1)):
        return False
    if np.any(np.diag(adj) != 0):
        return False

    # Degree regularity.
    if not np.all(adj.sum(axis=1) == k):
        return False
    if not np.all(adj.sum(axis=0) == k):
        return False

    A2 = adj @ adj

    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal of A²: reciprocal count should be t.
                if A2[i, j] != t:
                    return False
            else:
                expected = lambda_param if adj[i, j] == 1 else mu
                if A2[i, j] != expected:
                    return False

    return True


def solve_dsrg(
    n: int,
    k: int,
    t: int,
    lambda_param: int,
    mu: int,
    *,
    fix_out_neighbors_of_zero: bool = True,
    threads: int = -1,
    time_limit: float | None = None,
    quiet: bool = False,
) -> dict:
    """Build, solve, and return a results dict for DSRG(n, k, t, λ, μ).

    Returns:
        Dict with keys: status, wall_seconds, obj_val, adjacency.
        obj_val == 0 means a valid DSRG was found.
    """
    model, A = build_dsrg_model(
        n, k, t, lambda_param, mu,
        fix_out_neighbors_of_zero=fix_out_neighbors_of_zero,
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
    obj_val = None
    is_dsrg = False
    frac_integral = None
    if model.SolCount > 0:
        obj_val = model.ObjVal
        adj_raw = A.X
        # Count how many off-diagonal entries are already integral vs fractional.
        off_diag = adj_raw[~np.eye(n, dtype=bool)]
        is_int = np.abs(off_diag - np.round(off_diag)) < 1e-6
        frac_integral = float(is_int.mean())
        adj_rounded = np.round(adj_raw).astype(int)
        adjacency = adj_rounded.tolist()
        is_dsrg = verify_dsrg(adj_rounded, k, t, lambda_param, mu)

    return {
        "status":       status,
        "wall_seconds": round(elapsed, 4),
        "obj_val":      obj_val,
        "is_dsrg":      is_dsrg,
        "frac_integral": frac_integral,
        "n": n, "k": k, "t": t, "lambda": lambda_param, "mu": mu,
        "adjacency":    adjacency,
    }


if __name__ == "__main__":
    n = 15
    k = 4
    t = 2
    lambda_param = 1
    mu = 1
    params = dict(n=n, k=k, t=t, lambda_param=lambda_param, mu=mu)
    print(f"Solving DSRG({params['n']}, {params['k']}, {params['t']}, "
          f"{params['lambda_param']}, {params['mu']}) via penalty relaxation ...")

    result = solve_dsrg(**params, threads=-1)
    print(f"Status:      {result['status']}")
    print(f"Wall time:   {result['wall_seconds']:.3f}s")
    print(f"Objective:   {result['obj_val']}")
    if result["frac_integral"] is not None:
        pct = result["frac_integral"] * 100
        print(f"Integrality: {pct:.1f}% of off-diagonal variables are integral")

    if result["adjacency"] is not None:
        print("Adjacency matrix (rounded):")
        for row in result["adjacency"]:
            print(row)
        print(f"\nRounded matrix is valid DSRG: {result['is_dsrg']}")
        if not result["is_dsrg"]:
            print(f"(Objective = {result['obj_val']:.6f}, likely a local minimum)")
