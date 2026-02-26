import time
import pulp as pl
from pulp import LpVariable


def build_srg_lp(n, k, lambda_param, mu, fix_neighbors_of_zero=True, lex_order=True):
    prob = pl.LpProblem('SRG_problem', pl.LpMinimize)

    # Only upper triangle variables; edges[i,j] for i < j represents the undirected edge {i,j}
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            if fix_neighbors_of_zero and i == 0:
                # Symmetry breaking: fix neighbors of vertex 0 to {1, 2, ..., k}
                val = 1 if j <= k else 0
                edges[i, j] = LpVariable(f"e_{i}_{j}", lowBound=val, upBound=val, cat="Binary")
            else:
                edges[i, j] = LpVariable(f"e_{i}_{j}", cat="Binary")

    def e(i, j):
        """Return the edge variable for {i, j}, regardless of order."""
        if i == j:
            return 0
        return edges[min(i, j), max(i, j)]

    # Regularity: every vertex has degree k
    for x in range(n):
        prob += pl.lpSum(e(x, y) for y in range(n) if y != x) == k, f"Degree_{x}"

    # Auxiliary variable: p[x,y,z] == 1 <==> x~y and y~z (and x!=z, no loops)
    # Counts common neighbors via paths of length 2
    p = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if x == y or y == z or x == z:
                    p[x, y, z] = LpVariable(name=f"p_{x}_{y}_{z}", lowBound=0, upBound=0, cat="Continuous")
                    continue
                exy = e(x, y)
                eyz = e(y, z)
                p[x, y, z] = LpVariable(name=f"p_{x}_{y}_{z}", lowBound=0, upBound=1, cat="Binary")
                prob += p[x, y, z] <= exy
                prob += p[x, y, z] <= eyz
                prob += p[x, y, z] >= exy + eyz - 1

    # Lambda-Mu constraint:
    # For x != z: |{y : x~y~z}| == lambda if x~z, else mu
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            prob += (
                pl.lpSum(p[x, y, z] for y in range(n))
                == mu + e(x, z) * (lambda_param - mu)
            )

    # Lexicographic row ordering (symmetry breaking) using exponential weights:
    # For each consecutive pair (i, i+1), compare rows on the aligned column
    # set excluding j = i and j = i+1 (diagonal positions). Assign strictly
    # decreasing powers of 2 so weighted sums preserve lex order.
    # Enforce non-strict order: row i <=_lex row i+1.
    if lex_order:
        for i in range(n - 1):
            cols = [j for j in range(n) if j != i and j != i + 1]

            weights = {
                j: 1 << (len(cols) - idx)
                for idx, j in enumerate(cols)
            }

            sum_i = pl.lpSum(
                e(i, j) * weights[j] for j in cols
            )
            sum_ip1 = pl.lpSum(
                e(i + 1, j) * weights[j] for j in cols
            )

            prob += sum_ip1 >= sum_i, f"lex_{i}"

    return prob, edges, e


if __name__ == "__main__":
    # Petersen graph: SRG(10, 3, 0, 1)
    # Paley 13:       SRG(13, 6, 2, 3)
    # SRG(16, 6, 2, 2)
    n, k, lambda_param, mu = 36,10,4,2
    lex = True
    prob, edges, e = build_srg_lp(n, k, lambda_param, mu,
        fix_neighbors_of_zero=not lex,
        lex_order=lex)
    t0 = time.perf_counter()
    prob.solve(pl.GUROBI(threads=-1))
    print(f"Solve time: {time.perf_counter() - t0:.2f}s")

    print("Status:", pl.LpStatus[prob.status])
    if prob.status == 1:
        print("Adjacency matrix:")
        for i in range(n):
            row = [e(i, j).varValue if i != j else 0 for j in range(n)]
            print([int(v) for v in row])
