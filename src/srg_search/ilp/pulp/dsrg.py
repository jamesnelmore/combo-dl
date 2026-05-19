import pulp as pl
from pulp import LpVariable


def build_dsrg_lp(n, k, t, lambda_param, mu, fix_out_neighbors_of_zero=True):
    prob = pl.LpProblem("DSRG_problem", pl.LpMinimize)

    edges = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                # Fixed to 0 — loopless graph, diagonal is always 0
                edges[i, j] = LpVariable(
                    f"e_{i}_{j}", lowBound=0, upBound=0, cat="Continuous"
                )
            elif fix_out_neighbors_of_zero and i == 0:
                # Symmetry breaking: fix out-neighbors of vertex 0 to {1, 2, ..., k}
                val = 1 if j <= k else 0
                edges[i, j] = LpVariable(
                    f"e_{i}_{j}", lowBound=val, upBound=val, cat="Binary"
                )
            else:
                edges[i, j] = LpVariable(f"e_{i}_{j}", cat="Binary")

    prob += pl.lpSum(edges[i, j] for i in range(n) for j in range(n))

    # Regularity constraints
    for x in range(n):  # Out degree
        prob += pl.lpSum(edges[x, i] for i in range(n)) == k, f"Vertex {x} out-degree"
    for x in range(n):  # In degree
        prob += pl.lpSum(edges[i, x] for i in range(n)) == k, f"Vertex {x} in-degree"

    # Auxiliary variable: p[x,y,z] == 1 <==> edge[x,y] == 1 AND edge[y,z] == 1
    # i.e., there is a directed path x->y->z
    p = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if x == y or y == z:  # loopless: intermediate y can't equal endpoints
                    p[x, y, z] = LpVariable(
                        name=f"p_{x}_{y}_{z}", lowBound=0, upBound=0, cat="Continuous"
                    )
                    continue
                p[x, y, z] = LpVariable(
                    name=f"p_{x}_{y}_{z}", lowBound=0, upBound=1, cat="Binary"
                )
                prob += p[x, y, z] <= edges[x, y]
                prob += p[x, y, z] <= edges[y, z]
                prob += p[x, y, z] >= edges[x, y] + edges[y, z] - 1

    # Auxiliary variable: r[x,y] == 1 <==> edge[x,y] == 1 AND edge[y,x] == 1 (reciprocal edge)
    r = {}
    for x in range(n):
        for y in range(n):
            if x == y:
                r[x, y] = LpVariable(
                    name=f"r_{x}_{y}", lowBound=0, upBound=0, cat="Continuous"
                )
                continue
            r[x, y] = LpVariable(name=f"r_{x}_{y}", lowBound=0, upBound=1, cat="Binary")
            prob += r[x, y] <= edges[x, y]
            prob += r[x, y] <= edges[y, x]
            prob += r[x, y] >= edges[x, y] + edges[y, x] - 1

    # t constraint: for each x, the number of y such that x->y and y->x equals t
    for x in range(n):
        prob += pl.lpSum(r[x, y] for y in range(n) if y != x) == t

    # Lambda-Mu constraint (Duval's definition):
    # For x != z: |{y : x->y->z}| == lambda if x->z, else mu
    for x in range(n):
        for z in range(n):
            if x == z:
                continue
            prob += pl.lpSum(p[x, y, z] for y in range(n)) == mu + edges[x, z] * (
                lambda_param - mu
            )

    return prob, edges


if __name__ == "__main__":
    # With CBC
    ## Found (6,2,1,0,1) in .06 s on MPB
    ## (8,3,2,1,1) in .15
    ## (10,4,2,1,2) in 10.27
    ## (12,3,1,0,1) in 1.22
    # Open case: 24,10,5,3,5
    # n, k, t, lambda_param, mu = 16,7,4,3,3 # 84 without first k set
    n, k, t, lambda_param, mu = 24, 10, 5, 3, 5
    prob, edges = build_dsrg_lp(
        n, k, t, lambda_param, mu, fix_out_neighbors_of_zero=True
    )
    prob.solve(pl.GUROBI())

    print("Status:", pl.LpStatus[prob.status])
    if prob.status == 1:
        print("Adjacency matrix:")
        for i in range(n):
            row = [int(edges[i, j].varValue) for j in range(n)]
            print(row)
