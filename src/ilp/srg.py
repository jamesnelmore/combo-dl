"""Strongly Regular Graph (SRG) ILP formulation using native gurobipy.

An SRG(n, k, λ, μ) is an undirected graph on n vertices where every vertex
has degree k, every pair of adjacent vertices has exactly λ common neighbours,
and every pair of non-adjacent vertices has exactly μ common neighbours.

Three formulations for the λ/μ common-neighbour constraints are supported:

  Linearised (default):
    For each unordered pair {x, z} and each potential common neighbour y,
    introduces one auxiliary binary variable q[x,z,y] = e(x,y) AND e(y,z).
    Exploits the undirected symmetry: the common-neighbour count of {x, z}
    is the same regardless of vertex order, so only n(n-1)/2 constraints
    are needed (not n(n-1)), and each product e(x,y)·e(y,z) is created
    once per unordered {x,z} rather than twice.  Degenerate zero-fixed
    variables are no longer created.

    Objective: max Σ q.  At integrality Σq = n·k·(k-1)/2 (a constant), so
    the integer feasible set is unchanged.  In the LP relaxation the AND
    constraints leave slack when edges are fractional; maximising Σq pushes
    each q to its upper bound, eliminating the degenerate optimal face that
    causes crossover to stall and tightening the relaxation for
    branch-and-bound.

    Variables:  n(n-1)/2 edges  +  n(n-1)(n-2)/2 products
    Constraints: n degrees + n(n-1)/2 λ/μ + 3·n(n-1)(n-2)/2 AND links
    For n=99:  4,851 edges + 470,448 products = 475,299 binary vars
               99 + 4,851 + 1,411,344 = ~1.42M constraints

  Quadratic (--quadratic):
    Uses Gurobi's native non-convex quadratic constraints (NonConvex=2) to
    express Σ_y e(x,y)·e(y,z) directly.  Zero auxiliary variables — the
    model contains only the n(n-1)/2 edge variables.  NOTE: Gurobi
    internally linearises these via McCormick envelopes that are weaker
    than the binary AND encoding, and the presolved model can be larger
    than the explicit linearisation.  Best suited for small n.

Two formulations for lexicographic row ordering are supported:

  Exponential weights (default for small n):
    Single constraint per row pair using powers-of-2 weights.  Numerically
    unstable when n > ~55 (weights exceed float64 precision).

  Chained carry (--stable-lex, default when n > 55):
    Introduces (n-2) binary "tied-through-column-j" variables per
    consecutive row pair — numerically stable for any n.
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

# Column count threshold above which we automatically switch to stable lex.
_STABLE_LEX_THRESHOLD = 55


def build_srg_model(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors_of_zero: bool = True,
    fix_hamiltonian_cycle: bool = False,
    lex_order: bool = True,
    stable_lex: bool | None = None,
    quadratic: bool = False,
    quiet: bool = True,
) -> tuple[gp.Model, dict]:
    """Build and return a Gurobi model for SRG(n, k, λ, μ).

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: Number of common neighbours for adjacent vertices.
        mu: Number of common neighbours for non-adjacent vertices.
        fix_neighbors_of_zero: Symmetry break that pins the neighbours
            of vertex 0 to {1, 2, …, k}.
        fix_hamiltonian_cycle: Symmetry break that fixes a Hamiltonian
            cycle 0–1–2–…–(n-1)–0.  Fixes n edges to 1 and reduces
            each vertex's free degree to k − 2.  Mutually exclusive
            with fix_neighbors_of_zero.
        lex_order: Enforce lexicographic row ordering as symmetry breaking.
        stable_lex: Use numerically stable chained-carry lex encoding.
            If None (default), automatically enabled when n > 55.
        quadratic: Use quadratic common-neighbour constraints instead of
            explicit product linearisation.  Requires NonConvex=2 but
            eliminates all auxiliary variables.
        quiet: Suppress Gurobi console output.

    Returns:
        (model, edges) where edges[(i, j)] (i < j) is the binary variable
        for undirected edge {i, j}.
    """
    if fix_neighbors_of_zero and fix_hamiltonian_cycle:
        msg = "fix_neighbors_of_zero and fix_hamiltonian_cycle are mutually exclusive"
        raise ValueError(msg)
    if stable_lex is None:
        stable_lex = n > _STABLE_LEX_THRESHOLD

    model = gp.Model("SRG")
    if quiet:
        model.setParam("OutputFlag", 0)
    if quadratic:
        model.setParam("NonConvex", 2)

    # ── Hamiltonian cycle edge set (for fixing) ───────────────────────────────
    # Cycle: 0–1–2–…–(n-1)–0.  Store as a set of (min, max) pairs.
    ham_edges: set[tuple[int, int]] = set()
    if fix_hamiltonian_cycle:
        for v in range(n):
            u = (v + 1) % n
            ham_edges.add((min(v, u), max(v, u)))

    # ── Edge variables ────────────────────────────────────────────────────────
    # Only upper-triangle variables: edges[i, j] for i < j represents {i, j}.
    edges: dict[tuple[int, int], gp.Var] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if fix_neighbors_of_zero and i == 0:
                # Symmetry break: neighbours of vertex 0 are exactly {1, …, k}.
                val = 1.0 if j <= k else 0.0
                edges[i, j] = model.addVar(
                    lb=val, ub=val, vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )
            elif (i, j) in ham_edges:
                # Hamiltonian cycle edge — fixed to 1.
                edges[i, j] = model.addVar(
                    lb=1.0, ub=1.0, vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )
            else:
                edges[i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"e_{i}_{j}",
                )

    def e(i: int, j: int) -> gp.Var:
        """Return the edge variable for {i, j}, regardless of order.

        Caller must ensure i != j.
        """
        return edges[min(i, j), max(i, j)]

    # Objective is set after the common-neighbour block:
    #   - Linearised: max Σ q  (breaks LP relaxation degeneracy)
    #   - Quadratic:  min Σ edges  (dummy; no q variables to maximise)

    # ── Degree constraints ────────────────────────────────────────────────────
    for x in range(n):
        model.addConstr(
            gp.quicksum(
                edges[min(x, y), max(x, y)] for y in range(n) if y != x
            ) == k,
            name=f"deg_{x}",
        )

    # ── Common-neighbour (λ/μ) constraints ────────────────────────────────────
    if quadratic:
        # Quadratic formulation: no auxiliary variables.
        # For x < z:  Σ_y e(x,y)·e(y,z) = μ + e(x,z)·(λ − μ)
        for x in range(n):
            for z in range(x + 1, n):
                exz = e(x, z)
                common = gp.quicksum(
                    e(x, y) * e(y, z)
                    for y in range(n)
                    if y != x and y != z
                )
                model.addConstr(
                    common == mu + exz * (lambda_param - mu),
                    name=f"lm_{x}_{z}",
                )

        # No q variables to maximise — fall back to a dummy objective.
        model.setObjective(
            gp.quicksum(edges[i, j] for i, j in edges), GRB.MINIMIZE,
        )
    else:
        # Linearised formulation exploiting undirected symmetry.
        #
        # For each unordered pair {x, z} (x < z) and each potential common
        # neighbour y ∉ {x, z}, create:
        #
        #   q[x, z, y] = e(x, y) AND e(y, z)
        #
        # Since the graph is undirected, the common-neighbour count for (x, z)
        # and (z, x) is identical — we only need one constraint per unordered
        # pair.  And since e(x,y)·e(y,z) = e(z,y)·e(y,x), each product is
        # created exactly once.
        #
        # This halves both the number of product variables and the number of
        # λ/μ constraints compared to iterating over all ordered pairs.
        q: dict[tuple[int, int, int], gp.Var] = {}
        for x in range(n):
            for z in range(x + 1, n):
                for y in range(n):
                    if y == x or y == z:
                        continue
                    exy = e(x, y)
                    eyz = e(y, z)
                    q[x, z, y] = model.addVar(
                        vtype=GRB.BINARY, name=f"q_{x}_{z}_{y}",
                    )
                    model.addGenConstrAnd(
                        q[x, z, y],
                        [exy, eyz],
                        name=f"q_and_{x}_{z}_{y}",
                    )

        # λ/μ constraint — one per unordered pair {x, z}:
        #   |{y : x~y~z}| = μ + e(x,z)·(λ − μ)
        for x in range(n):
            for z in range(x + 1, n):
                model.addConstr(
                    gp.quicksum(
                        q[x, z, y] for y in range(n) if y != x and y != z
                    ) == mu + e(x, z) * (lambda_param - mu),
                    name=f"lm_{x}_{z}",
                )

        # ── Objective: maximise Σ q ───────────────────────────────────────────
        #
        # At integrality q[x,z,y] = e(x,y)·e(y,z), so Σq = n·k·(k-1)/2
        # (every vertex y contributes C(k,2) neighbour pairs) — a constant.
        # The objective therefore does NOT change the integer feasible set.
        #
        # In the LP relaxation, however, the AND constraints only force
        #   q ≤ e(x,y),  q ≤ e(y,z),  q ≥ e(x,y) + e(y,z) − 1
        # When edges are fractional (e.g. 0.5), q can sit anywhere in
        # [0, 0.5].  Maximising Σq pushes every q to its upper bound,
        # which:
        #   1. Eliminates the degenerate optimal face that causes crossover
        #      to stall (the 386k PPushes problem).
        #   2. Tightens the relaxation by forcing q closer to the true
        #      binary product, giving branch-and-bound sharper bounds.
        model.setObjective(
            gp.quicksum(q[x, z, y] for x, z, y in q), GRB.MAXIMIZE,
        )

    # ── Lexicographic row ordering (symmetry breaking) ────────────────────────
    if lex_order:
        if stable_lex:
            _add_stable_lex(model, edges, e, n)
        else:
            _add_weighted_lex(model, e, n)

    model.update()
    return model, edges


def _add_weighted_lex(
    model: gp.Model,
    e,
    n: int,
) -> None:
    """Exponential-weight lex ordering — one constraint per row pair.

    Numerically unstable for n > ~55.
    """
    for i in range(n - 1):
        cols = [j for j in range(n) if j != i and j != i + 1]

        weights = {
            j: 1 << (len(cols) - idx)
            for idx, j in enumerate(cols)
        }

        sum_i = gp.quicksum(e(i, j) * weights[j] for j in cols)
        sum_ip1 = gp.quicksum(e(i + 1, j) * weights[j] for j in cols)

        model.addConstr(sum_ip1 >= sum_i, name=f"lex_{i}")


def _add_stable_lex(
    model: gp.Model,
    edges: dict,
    e,
    n: int,
) -> None:
    """Chained-carry lex ordering — numerically stable for any n.

    For each consecutive row pair (i, i+1) and comparison columns
    c_0, c_1, …, c_{m-1} (all columns except the two diagonal positions
    i and i+1), we introduce binary variables:

        t[i, j]  =  1  iff rows i and i+1 are identical on columns c_0 … c_j

    Constraints (for row pair i, column index j with a = row_i[c_j],
    b = row_{i+1}[c_j]):

        Upper bounds on t (force t=0 when rows differ):
            t[0] <= 1 - a + b
            t[0] <= 1 + a - b
            t[j] <= t[j-1]              (can't be tied if previously untied)
            t[j] <= 1 - a + b
            t[j] <= 1 + a - b

        Lex constraint at each column:
            j = 0:  b >= a               (first column: row i+1 must not lose)
            j > 0:  b >= a - 1 + t[j-1]  (if tied through j-1, i+1 must not lose)

    Total per row pair: (n-2) binary variables, 4(n-2) constraints.
    All coefficients are in {-1, 0, 1}.
    """
    for i in range(n - 1):
        cols = [j for j in range(n) if j != i and j != i + 1]
        m = len(cols)
        if m == 0:
            continue

        # Binary "tied through column j" variables for this row pair.
        t = [
            model.addVar(vtype=GRB.BINARY, name=f"lex_t_{i}_{j}")
            for j in range(m)
        ]

        for idx, c in enumerate(cols):
            a = e(i, c)      # row i,   column c
            b = e(i + 1, c)  # row i+1, column c

            if idx == 0:
                # t[0]: tied iff a_0 == b_0
                model.addConstr(t[0] <= 1 - a + b, name=f"lex_tu0a_{i}")
                model.addConstr(t[0] <= 1 + a - b, name=f"lex_tu0b_{i}")
                # Lex at column 0: row i+1 must not lose.
                model.addConstr(b >= a, name=f"lex_c_{i}_0")
            else:
                # t[idx]: tied iff previously tied AND a_idx == b_idx
                model.addConstr(
                    t[idx] <= t[idx - 1],
                    name=f"lex_chain_{i}_{idx}",
                )
                model.addConstr(
                    t[idx] <= 1 - a + b,
                    name=f"lex_tua_{i}_{idx}",
                )
                model.addConstr(
                    t[idx] <= 1 + a - b,
                    name=f"lex_tub_{i}_{idx}",
                )
                # Lex at column idx: if tied through idx-1, then b >= a.
                #   b >= a - (1 - t[idx-1])   ⟺   b >= a - 1 + t[idx-1]
                model.addConstr(
                    b >= a - 1 + t[idx - 1],
                    name=f"lex_c_{i}_{idx}",
                )


def solve_srg(
    n: int,
    k: int,
    lambda_param: int,
    mu: int,
    *,
    fix_neighbors_of_zero: bool = True,
    fix_hamiltonian_cycle: bool = False,
    lex_order: bool = True,
    stable_lex: bool | None = None,
    quadratic: bool = False,
    threads: int = -1,
    time_limit: float | None = None,
    mip_focus: int = 0,
    heuristics: float | None = None,
    quiet: bool = False,
) -> dict:
    """Build, solve, and return a results dict for SRG(n, k, λ, μ).

    Args:
        n: Number of vertices.
        k: Degree of every vertex.
        lambda_param: λ parameter.
        mu: μ parameter.
        fix_neighbors_of_zero: Enable symmetry breaking on vertex 0.
        fix_hamiltonian_cycle: Fix a Hamiltonian cycle 0–1–…–(n-1)–0.
        lex_order: Enforce lexicographic row ordering as symmetry breaking.
        stable_lex: Use chained-carry lex (auto-enabled for n > 55).
        quadratic: Use quadratic common-neighbour constraints (no aux vars).
        threads: Number of solver threads (-1 = all available).
        time_limit: Optional wall-clock limit in seconds.
        mip_focus: Gurobi MIPFocus parameter (0 = balanced, 1 = find feasible
            solutions quickly, 2 = prove optimality, 3 = tighten bound).
        heuristics: Fraction of time spent on MIP heuristics (0.0–1.0).
            None leaves Gurobi's default (0.05).
        quiet: Suppress Gurobi output.

    Returns:
        Dict with keys: status, wall_seconds, adjacency (list[list[int]] or None).
    """
    model, edges = build_srg_model(
        n, k, lambda_param, mu,
        fix_neighbors_of_zero=fix_neighbors_of_zero,
        fix_hamiltonian_cycle=fix_hamiltonian_cycle,
        lex_order=lex_order,
        stable_lex=stable_lex,
        quadratic=quadratic,
        quiet=quiet,
    )

    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_focus != 0:
        model.setParam("MIPFocus", mip_focus)
    if heuristics is not None:
        model.setParam("Heuristics", heuristics)

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
        adjacency = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    row.append(int(round(edges[min(i, j), max(i, j)].X)))
            adjacency.append(row)

    return {
        "status":       status,
        "wall_seconds": round(elapsed, 4),
        "n": n, "k": k, "lambda": lambda_param, "mu": mu,
        "adjacency":    adjacency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find a Strongly Regular Graph SRG(n, k, λ, μ) via ILP.",
    )
    parser.add_argument("n", type=int, help="number of vertices")
    parser.add_argument("k", type=int, help="degree of every vertex")
    parser.add_argument("lam", type=int, metavar="lambda",
                        help="common neighbours for adjacent pairs")
    parser.add_argument("mu", type=int,
                        help="common neighbours for non-adjacent pairs")
    parser.add_argument("--quadratic", action="store_true",
                        help="use quadratic common-neighbour constraints "
                             "(no auxiliary variables, requires NonConvex=2)")
    parser.add_argument("--no-lex", action="store_true",
                        help="disable lexicographic row ordering")
    parser.add_argument("--stable-lex", action="store_true", default=False,
                        help="force numerically stable chained-carry lex "
                             "(auto-enabled when n > 55)")
    parser.add_argument("--fix-zero", action="store_true",
                        help="fix neighbours of vertex 0 to {1, …, k}")
    parser.add_argument("--fix-ham", action="store_true",
                        help="fix a Hamiltonian cycle 0-1-2-…-(n-1)-0 "
                             "(mutually exclusive with --fix-zero)")
    parser.add_argument("--threads", type=int, default=-1,
                        help="solver threads (-1 = all available, default: -1)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="wall-clock time limit in seconds")
    parser.add_argument("--mip-focus", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Gurobi MIPFocus: 0=balanced, 1=feasibility, "
                             "2=optimality, 3=bound (default: 0)")
    parser.add_argument("--heuristics", type=float, default=None,
                        help="fraction of time on MIP heuristics, 0.0-1.0 "
                             "(default: Gurobi's 0.05)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress Gurobi output")
    args = parser.parse_args()

    lex = not args.no_lex
    # --stable-lex flag forces it; otherwise None lets build_srg_model decide.
    stable: bool | None = True if args.stable_lex else None

    print(f"Solving SRG({args.n}, {args.k}, {args.lam}, {args.mu})")
    print(f"  quadratic={args.quadratic}  lex={lex}  stable_lex={stable}  "
          f"fix_zero={args.fix_zero}  fix_ham={args.fix_ham}")

    result = solve_srg(
        args.n, args.k, args.lam, args.mu,
        fix_neighbors_of_zero=args.fix_zero,
        fix_hamiltonian_cycle=args.fix_ham,
        lex_order=lex,
        stable_lex=stable,
        quadratic=args.quadratic,
        threads=args.threads,
        time_limit=args.time_limit,
        mip_focus=args.mip_focus,
        heuristics=args.heuristics,
        quiet=args.quiet,
    )
    print(f"Status:    {result['status']}")
    print(f"Wall time: {result['wall_seconds']:.2f}s")

    if result["adjacency"] is not None:
        print("Adjacency matrix:")
        for row in result["adjacency"]:
            print(row)
