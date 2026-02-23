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

    # Lexicographic row ordering (symmetry breaking):
    # For each consecutive pair of rows (i, i+1), enforce row i <_lex row i+1.
    # agree[i,j] == 1 iff e(i,0..j) == e(i+1,0..j) (prefix agreement up to column j)
    if lex_order:
        for i in range(n - 1):
            cols = [j for j in range(n) if j != i and j != i + 1]  # skip diagonal entries
            agree = {}
            for idx, j in enumerate(cols):
                agree[j] = LpVariable(name=f"agree_{i}_{j}", cat="Binary")
                eij  = e(i,     j)
                eij1 = e(i + 1, j)
                # agree[j] == 1 => e(i,j) == e(i+1,j)
                prob += eij - eij1 <= 1 - agree[j], f"agree_eq0_{i}_{j}"
                prob += eij1 - eij <= 1 - agree[j], f"agree_eq1_{i}_{j}"
                # Agreement is a prefix: once it breaks, it stays broken
                if idx > 0:
                    prev_j = cols[idx - 1]
                    prob += agree[j] <= agree[prev_j], f"agree_prefix_{i}_{j}"

            for idx, j in enumerate(cols):
                eij  = e(i,     j)
                # If all columns before j agree and this is the first disagreement,
                # row i must have 0 (so row i+1 has the 1).
                if idx == 0:
                    prob += eij <= agree[j], f"lex_first_{i}_{j}"
                else:
                    prev_j = cols[idx - 1]
                    prob += eij <= 1 - agree[prev_j] + agree[j], f"lex_{i}_{j}"

            # Rows must not be identical
            prob += pl.lpSum(agree[j] for j in cols) <= len(cols) - 1, f"lex_neq_{i}"

    return prob, edges, e


if __name__ == "__main__":
    # Petersen graph: SRG(10, 3, 0, 1)
    # Paley 13:       SRG(13, 6, 2, 3)
    # SRG(16, 6, 2, 2)
    n, k, lambda_param, mu = 25,12,5,6
    prob, edges, e = build_srg_lp(n, k, lambda_param, mu,
        fix_neighbors_of_zero=True,
        lex_order=False)
    t0 = time.perf_counter()
    prob.solve(pl.GUROBI(threads=-1))
    print(f"Solve time: {time.perf_counter() - t0:.2f}s")

    print("Status:", pl.LpStatus[prob.status])
    if prob.status == 1:
        print("Adjacency matrix:")
        for i in range(n):
            row = [e(i, j).varValue if i != j else 0 for j in range(n)]
            print([int(v) for v in row])
