"""Penalty-relaxation model for undirected Strongly Regular Graphs (SRGs).

An SRG(n, k, λ, μ) is an undirected graph on n vertices where every vertex
has degree k, every pair of adjacent vertices has exactly λ common neighbours,
and every pair of non-adjacent vertices has exactly μ common neighbours.

This formulation keeps k-regularity as the sole hard constraint and relaxes
the λ/μ conditions into a soft penalty objective that counts the total number
of vertex-pair violations.

For an adjacency matrix A (symmetric, zero diagonal, binary), the SRG
conditions require:

    (A²)_{xz} = λ   if A_{xz} = 1   (adjacent pair)
    (A²)_{xz} = μ   if A_{xz} = 0   (non-adjacent pair, x ≠ z)

Equivalently, for every x ≠ z:

    (A²)_{xz} = μ + (λ − μ) · A_{xz}

The residual for each off-diagonal pair is:

    R_{xz} = (A²)_{xz} − (λ − μ) · A_{xz} − μ

A solution with all R_{xz} = 0 is a valid SRG.  The objective minimises
Σ_{x≠z} R_{xz}², which equals the squared Frobenius norm of the residual
matrix (off-diagonal part).

Since the edge variables are binary, each entry (A²)_{xz} = Σ_y A_{xy}·A_{yz}
is an integer-valued quadratic form in the edge variables.  Rather than
introducing O(n³) auxiliary linearisation variables p_{xyz} (as the naive ILP
does), this formulation works directly with the quadratic products via
Gurobi's non-convex QCQP support (NonConvex=2):

  • O(n²) binary edge variables  e_{ij} for i < j  (upper triangle; symmetry
    gives A_{ij} = A_{ji} = e_{min(i,j), max(i,j)}).
  • n linear degree constraints (k-regularity).
  • O(n²) continuous residual variables S_{xz}, each defined by a single
    quadratic constraint  S_{xz} = Σ_y e(x,y)·e(y,z) − (λ−μ)·e(x,z) − μ.
  • Quadratic objective  min Σ S_{xz}².

This is a mixed-integer quadratically-constrained quadratic program (MIQCQP),
solved by Gurobi with NonConvex=2.
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_srg_model(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors_of_zero: bool = True,
    lex_order: bool = False,
    quiet: bool = False,
) -> tuple[gp.Model, gp.MVar]:
    """Build and return a Gurobi MIQCQP for the relaxed SRG(n, k, λ, μ).

    The model keeps k-regularity as hard constraints and minimises the sum of
    squared residuals of the λ/μ common-neighbour conditions.

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: Target number of common neighbours for adjacent pairs.
        mu: Target number of common neighbours for non-adjacent pairs.
        fix_neighbors_of_zero: Symmetry break that pins the neighbours of
            vertex 0 to {1, 2, …, k}.
        lex_order: Enforce lexicographic row ordering as symmetry breaking.
        quiet: Suppress Gurobi console output.

    Returns:
        (model, A) where A is the n×n symmetric binary MVar adjacency matrix.
    """
    model = gp.Model("SRG_relaxed")
    if quiet:
        model.setParam("OutputFlag", 0)

    # Non-convex quadratic constraints (S = A² − …) require this.
    model.setParam("NonConvex", 2)

    # ── Edge variables as an n×n MVar ─────────────────────────────────────────
    # A[i,j] ∈ {0,1} binary.  Diagonal forced to 0 (loopless).
    # A is symmetric: A[i,j] = A[j,i] for all i,j.
    lb = np.zeros((n, n))
    ub = np.ones((n, n))

    # Diagonal: loopless → fix to 0.
    for i in range(n):
        ub[i, i] = 0.0

    # Symmetry break: neighbours of vertex 0 are exactly {1, …, k}.
    if fix_neighbors_of_zero:
        for j in range(1, n):
            val = 1.0 if j <= k else 0.0
            lb[0, j] = val
            ub[0, j] = val
            lb[j, 0] = val
            ub[j, 0] = val

    A = model.addMVar((n, n), lb=lb, ub=ub, vtype=GRB.BINARY, name="e")

    # ── Symmetry constraints: A[i,j] = A[j,i] ───────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(A[i, j] == A[j, i], name=f"sym_{i}_{j}")

    # ── Degree constraints (hard): every vertex has degree k ──────────────────
    ones = np.ones(n)
    for x in range(n):
        model.addConstr(A[x, :] @ ones == k, name=f"deg_{x}")

    # ── Lexicographic row ordering (symmetry breaking) ────────────────────────
    # Uses exponential weights on aligned columns to enforce lex order between
    # consecutive rows, following the same scheme as the naive SRG ILP.
    if lex_order:
        start_row = 1 if fix_neighbors_of_zero else 0

        for i in range(start_row, n - 1):
            # Compare rows i and i+1 on columns excluding j=i and j=i+1.
            cols = [j for j in range(n) if j != i and j != i + 1]

            weights = {j: 1 << (len(cols) - idx) for idx, j in enumerate(cols)}

            sum_i = gp.quicksum(A[i, j] * weights[j] for j in cols)
            sum_ip1 = gp.quicksum(A[i + 1, j] * weights[j] for j in cols)

            model.addConstr(sum_ip1 >= sum_i, name=f"lex_{i}")

    # ── Build constant target matrix C ────────────────────────────────────────
    # For x ≠ z the SRG condition is:
    #   (A²)_{xz} = μ + (λ − μ) · A_{xz}
    #
    # Define the residual  R_{xz} = (A²)_{xz} − (λ − μ)·A_{xz} − μ.
    # On the diagonal (A²)_{xx} = deg(x) = k, which is automatically satisfied
    # by the degree constraints, so we only penalise off-diagonal entries.
    #
    # C_{xz} = μ for x ≠ z,  0 on diagonal (unused).
    lam_minus_mu = lambda_param - mu

    # ── Residual variables S_{xz} for x ≠ z ──────────────────────────────────
    # S_{xz} = (A²)_{xz} − (λ−μ)·A_{xz} − μ
    #
    # (A²)_{xz} = Σ_y A_{xy}·A_{yz}  is quadratic in the edge variables.
    # The constraint defining S_{xz} is therefore a quadratic equality.
    #
    # Bounds: (A²)_{xz} ∈ [0, n−2] (at most n−2 common neighbours),
    # and (λ−μ)·A_{xz} ∈ {0, λ−μ}, so S_{xz} is bounded.
    max_common = n - 2
    s_lb = -abs(lam_minus_mu) - mu
    s_ub = max_common - min(0, lam_minus_mu) - mu

    S = model.addMVar(
        (n, n),
        lb=s_lb,
        ub=s_ub,
        vtype=GRB.CONTINUOUS,
        name="s",
    )

    # Fix diagonal residuals to 0 (not part of the penalty).
    for i in range(n):
        model.addConstr(S[i, i] == 0, name=f"s_diag_{i}")

    # Quadratic constraints: S_{xz} = row_x · col_z − (λ−μ)·A_{xz} − μ
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            model.addConstr(
                S[x, z] == A[x, :] @ A[:, z] - lam_minus_mu * A[x, z] - mu,
                name=f"res_{x}_{z}",
            )

    # ── Objective: minimise Σ_{x≠z} S_{xz}² ─────────────────────────────────
    s_flat = S.reshape(-1)
    model.setObjective(s_flat @ s_flat, GRB.MINIMIZE)

    model.update()
    return model, A


def verify_srg(
    adj: np.ndarray,
    k: int,
    lambda_param: int,
    mu: int,
) -> bool:
    """Check whether an integer adjacency matrix is a valid SRG(n, k, λ, μ).

    Returns True iff all SRG conditions hold exactly.
    """
    n = adj.shape[0]

    # Must be 0/1, symmetric, with zero diagonal.
    if not np.all((adj == 0) | (adj == 1)):
        return False
    if np.any(np.diag(adj) != 0):
        return False
    if not np.array_equal(adj, adj.T):
        return False

    # Degree regularity.
    if not np.all(adj.sum(axis=1) == k):
        return False

    A2 = adj @ adj
    for x in range(n):
        for z in range(x + 1, n):
            expected = lambda_param if adj[x, z] == 1 else mu
            if A2[x, z] != expected:
                return False

    return True


def count_violations(
    adj: np.ndarray,
    lambda_param: int,
    mu: int,
) -> tuple[int, int, int]:
    """Count the number of λ/μ violations in an adjacency matrix.

    Returns:
        (total_violations, lambda_violations, mu_violations)
        where counts are over *unordered* pairs {x, z} with x < z.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    lam_viol = 0
    mu_viol = 0
    for x in range(n):
        for z in range(x + 1, n):
            common = A2[x, z]
            if adj[x, z] == 1:
                if common != lambda_param:
                    lam_viol += 1
            else:
                if common != mu:
                    mu_viol += 1
    return lam_viol + mu_viol, lam_viol, mu_viol


def solve_srg(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors_of_zero: bool = True,
    lex_order: bool = False,
    threads: int = -1,
    time_limit: float | None = None,
    quiet: bool = False,
) -> dict:
    """Build, solve, and return a results dict for relaxed SRG(n, k, λ, μ).

    Returns:
        Dict with keys: status, wall_seconds, obj_val, is_srg, violations,
        lambda_violations, mu_violations, adjacency.
        obj_val == 0 means a valid SRG was found.
    """
    model, A = build_srg_model(
        n,
        k,
        lambda_param,
        mu,
        fix_neighbors_of_zero=fix_neighbors_of_zero,
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
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "Suboptimal",
    }.get(model.Status, f"Unknown({model.Status})")

    adjacency = None
    obj_val = None
    is_srg = False
    violations = None
    lam_viol = None
    mu_viol = None

    if model.SolCount > 0:
        obj_val = model.ObjVal
        adj = A.X.astype(int)

        adjacency = adj.tolist()
        is_srg = verify_srg(adj, k, lambda_param, mu)
        violations, lam_viol, mu_viol = count_violations(
            adj, lambda_param, mu
        )

    return {
        "status": status,
        "wall_seconds": round(elapsed, 4),
        "obj_val": obj_val,
        "is_srg": is_srg,
        "violations": violations,
        "lambda_violations": lam_viol,
        "mu_violations": mu_viol,
        "n": n,
        "k": k,
        "lambda": lambda_param,
        "mu": mu,
        "adjacency": adjacency,
    }


if __name__ == "__main__":
    # Petersen graph: SRG(10, 3, 0, 1)
    # Paley 13:       SRG(13, 6, 2, 3)
    # SRG(16, 6, 2, 2)
    n, k, lambda_param, mu = 17, 8, 3, 4
    lex = False
    params = dict(n=n, k=k, lambda_param=lambda_param, mu=mu)
    print(
        f"Solving relaxed SRG({n}, {k}, {lambda_param}, {mu}) "
        f"via penalty MIQCQP ..."
    )

    result = solve_srg(
        **params,
        fix_neighbors_of_zero=not lex,
        lex_order=lex,
        threads=-1,
    )
    print(f"Status:      {result['status']}")
    print(f"Wall time:   {result['wall_seconds']:.3f}s")
    print(f"Objective:   {result['obj_val']}")

    if result["adjacency"] is not None:
        print(f"\nViolations (unordered pairs):  {result['violations']}")
        print(f"  λ-violations: {result['lambda_violations']}")
        print(f"  μ-violations: {result['mu_violations']}")
        print(f"\nValid SRG: {result['is_srg']}")
        print("\nAdjacency matrix:")
        for row in result["adjacency"]:
            print(row)

        if not result["is_srg"] and result["obj_val"] is not None:
            print(
                f"\nObjective = {result['obj_val']:.6f}; "
                f"solution is a local/global minimum of violations."
            )
