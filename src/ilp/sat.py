"""SAT encoding for DSRG(n, k, t, λ, μ) using PySAT.

Encodes the matrix equation A² = (t−μ)I + (λ−μ)A + μJ over {0,1}
with zero diagonal and constant row/column sums k.

The off-diagonal entries give:  Σ_y a_{xy}·a_{yz} + (μ−λ)·a_{xz} = μ
encoded as cardinality constraints. The weighted a_{xz} term is handled
by adding (μ−λ) copies of the a_{xz} literal to the cardinality sum
(the "duplicate literal trick"), turning a pseudo-boolean constraint
into a plain cardinality constraint.

The diagonal constraint (reciprocal count = t) is implied by the
off-diagonal constraints + degree constraints, so it's omitted.

Requires: python-sat  (pip install python-sat)
"""

import sys
import time
from typing import Optional

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


def solve_dsrg_sat(
    n: int,
    k: int,
    t: int,
    lam: int,
    mu: int,
    *,
    solver_name: str = "cadical195",
    fix_vertex_zero: bool = True,
    verbose: bool = True,
) -> Optional[list[list[int]]]:
    """Encode DSRG(n,k,t,λ,μ) as SAT and solve.

    Args:
        n, k, t, lam, mu: DSRG parameters.
        solver_name: PySAT solver backend. Try "cadical195", "glucose42",
            or "minisat22" depending on what's installed.
        fix_vertex_zero: Pin vertex 0's out-neighborhood to {1,...,k}
            as a symmetry break.
        verbose: Print progress during encoding and solving.

    Returns:
        Adjacency matrix (list of lists) if SAT, None if UNSAT.
    """

    # ── Variable allocation ──────────────────────────────────────────────
    # edge[(i,j)] is the SAT variable (positive int, 1-indexed) for arc i→j.
    edge: dict[tuple[int, int], int] = {}
    next_var = 1
    for i in range(n):
        for j in range(n):
            if i != j:
                edge[i, j] = next_var
                next_var += 1
    top = next_var - 1  # highest variable ID so far

    cnf = CNF()

    # ── Symmetry breaking: fix out-neighbors of vertex 0 ────────────────
    if fix_vertex_zero:
        for j in range(1, n):
            if j <= k:
                cnf.append([edge[0, j]])    # force arc 0→j
            else:
                cnf.append([-edge[0, j]])   # forbid arc 0→j

    # ── Degree constraints ───────────────────────────────────────────────
    if verbose:
        print("Encoding degree constraints...")

    for i in range(n):
        out_lits = [edge[i, j] for j in range(n) if j != i]
        enc = CardEnc.equals(
            lits=out_lits, bound=k, top_id=top, encoding=EncType.totalizer
        )
        top = enc.nv
        cnf.extend(enc.clauses)

    for j in range(n):
        in_lits = [edge[i, j] for i in range(n) if i != j]
        enc = CardEnc.equals(
            lits=in_lits, bound=k, top_id=top, encoding=EncType.totalizer
        )
        top = enc.nv
        cnf.extend(enc.clauses)

    if verbose:
        print(f"  After degrees: {top} vars, {len(cnf.clauses)} clauses")

    # ── Off-diagonal path constraints ────────────────────────────────────
    # For each (x, z) with x ≠ z:
    #   Σ_y a_{xy}·a_{yz} = μ + (λ−μ)·a_{xz}
    #   ⟺  Σ_y p_{xyz} + (μ−λ)·a_{xz} = μ        when μ ≥ λ
    #   ⟺  Σ_y p_{xyz} + (λ−μ)·(1−a_{xz}) = λ     when λ > μ
    #
    # Each p_{xyz} = a_{xy} ∧ a_{yz} is a fresh AND variable (3 clauses).
    # The weighted edge term becomes duplicate copies of a literal in
    # the cardinality constraint.

    if mu >= lam:
        edge_weight = mu - lam    # copies of +edge[x,z] to add
        target = mu               # RHS of the cardinality equals
        negate_edge = False
    else:
        edge_weight = lam - mu    # copies of −edge[x,z] to add
        target = lam
        negate_edge = True

    total_pairs = n * (n - 1)

    def make_and(a_lit: int, b_lit: int) -> int:
        """Create a fresh variable v ⟺ (a_lit ∧ b_lit)."""
        nonlocal top
        top += 1
        v = top
        cnf.append([-v, a_lit])       # v → a
        cnf.append([-v, b_lit])       # v → b
        cnf.append([v, -a_lit, -b_lit])  # (a ∧ b) → v
        return v

    if verbose:
        print(f"Encoding {total_pairs} path constraints "
              f"(target={target}, edge_weight={edge_weight})...")

    for count, (x, z) in enumerate(
        (x, z) for x in range(n) for z in range(n) if x != z
    ):
        if verbose and count % 100 == 0:
            print(f"  {count}/{total_pairs} | {top} vars | "
                  f"{len(cnf.clauses)} clauses", end="\r")

        # AND variables for each intermediate vertex y
        p_lits: list[int] = []
        for y in range(n):
            if y == x or y == z:
                continue
            p_lits.append(make_and(edge[x, y], edge[y, z]))

        # Weighted edge term: add edge_weight copies of the (possibly
        # negated) edge literal to turn PB into cardinality
        if negate_edge:
            edge_lit = -edge[x, z]
        else:
            edge_lit = edge[x, z]
        extended = p_lits + [edge_lit] * edge_weight

        enc = CardEnc.equals(
            lits=extended, bound=target, top_id=top,
            encoding=EncType.totalizer,
        )
        top = enc.nv
        cnf.extend(enc.clauses)

    if verbose:
        print(f"\nFinal CNF: {top} variables, {len(cnf.clauses)} clauses")

    # ── Solve ────────────────────────────────────────────────────────────
    if verbose:
        print(f"Launching {solver_name}...")
        sys.stdout.flush()

    t0 = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf) as s:
        sat = s.solve()
        elapsed = time.perf_counter() - t0

        if verbose:
            status = "SAT" if sat else "UNSAT"
            print(f"Result: {status} ({elapsed:.1f}s)")

        if not sat:
            return None

        model_set = set(s.get_model())

    adj = [[0] * n for _ in range(n)]
    for (i, j), v in edge.items():
        if v in model_set:
            adj[i][j] = 1
    return adj


def verify_dsrg(
    adj: list[list[int]], n: int, k: int, t: int, lam: int, mu: int
) -> bool:
    """Check all DSRG conditions on a candidate adjacency matrix."""
    ok = True

    for i in range(n):
        if adj[i][i] != 0:
            print(f"FAIL: self-loop at {i}")
            ok = False

    for i in range(n):
        s = sum(adj[i])
        if s != k:
            print(f"FAIL: out-degree({i}) = {s}, expected {k}")
            ok = False

    for j in range(n):
        s = sum(adj[i][j] for i in range(n))
        if s != k:
            print(f"FAIL: in-degree({j}) = {s}, expected {k}")
            ok = False

    for x in range(n):
        rc = sum(adj[x][y] * adj[y][x] for y in range(n))
        if rc != t:
            print(f"FAIL: reciprocal({x}) = {rc}, expected {t}")
            ok = False

    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            paths = sum(adj[x][y] * adj[y][z] for y in range(n))
            want = lam if adj[x][z] else mu
            if paths != want:
                print(f"FAIL: paths {x}→?→{z} = {paths}, expected {want}")
                ok = False

    if ok:
        print("All DSRG conditions verified ✓")
    return ok


def print_adjacency(adj: list[list[int]]) -> None:
    """Print adjacency matrix as a compact binary grid."""
    for row in adj:
        print("".join(str(x) for x in row))


if __name__ == "__main__":
    # Default: the open case DSRG(24, 10, 5, 3, 5)
    # For a quick sanity check, try a known-to-exist case first, e.g.:
    #   DSRG(6, 2, 1, 0, 1)  — tiny, solves instantly
    #   DSRG(8, 3, 2, 1, 1)  — small, should solve in seconds
    #   DSRG(12, 4, 2, 1, 1) — moderate
    if len(sys.argv) == 6:
        n, k, t, lam, mu = (int(x) for x in sys.argv[1:6])
    else:
        n, k, t, lam, mu = 24, 10, 5, 3, 5

    print(f"DSRG({n}, {k}, {t}, {lam}, {mu})")
    print("=" * 50)

    adj = solve_dsrg_sat(n, k, t, lam, mu)

    if adj is not None:
        print("\nSolution found!")
        verify_dsrg(adj, n, k, t, lam, mu)
        print("\nAdjacency matrix:")
        print_adjacency(adj)
    else:
        print(f"\nUNSAT — no DSRG({n},{k},{t},{lam},{mu}) exists.")
