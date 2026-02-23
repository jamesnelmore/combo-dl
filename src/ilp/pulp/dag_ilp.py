"""
ILP: find a connected DAG on n vertices maximising edges.

Optional constraints
--------------------
upper_triangular : pin topological order to vertex labels (symmetry break)
no_2_path        : forbid i -> j -> k for all distinct i, j, k
forbid_diamond   : forbid the pattern a->b, a->c, b->d, c->d (b < c)
"""

import pulp


def max_dag(
    n: int,
    upper_triangular: bool = True,
    no_2_path: bool = True,
    forbid_diamond: bool = False,
) -> None:
    print("Formulating problem")
    print(f"  n={n}  upper_triangular={upper_triangular}  no_2_path={no_2_path}  forbid_diamond={forbid_diamond}")
    prob = pulp.LpProblem("MaxDAG", pulp.LpMaximize)

    # ── Edge variables ────────────────────────────────────────────────────────
    e = {
        (i, j): pulp.LpVariable(f"e_{i}_{j}", cat="Binary")
        for i in range(n) for j in range(n) if i != j
    }

    # ── Topological order variables ───────────────────────────────────────────
    topo = {
        i: pulp.LpVariable(f"t_{i}", lowBound=1, upBound=n, cat="Integer")
        for i in range(n)
    }

    # ── Objective: maximise edges ─────────────────────────────────────────────
    prob += pulp.lpSum(e.values())

    # ── Acyclicity: e_ij = 1  =>  t_j >= t_i + 1 ─────────────────────────────
    for (i, j), eij in e.items():
        prob += topo[j] >= topo[i] + 1 - n * (1 - eij)

    # ── Lex ordering / symmetry break: pin topo order to vertex labels ────────
    if upper_triangular:
        for i in range(n):
            prob += topo[i] == i + 1

    # ── No directed 2-path: forbid i -> j -> k ────────────────────────────────
    if no_2_path:
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    prob += e[i, j] + e[j, k] <= 1

    # ── Diamond prohibition: forbid a->b, a->c, b->d, c->d ───────────────────
    # Require b < c to enumerate each diamond exactly once.
    if forbid_diamond:
        for a in range(n):
            for b in range(n):
                if b == a:
                    continue
                for c in range(b + 1, n):
                    if c == a:
                        continue
                    for d in range(n):
                        if d in (a, b, c):
                            continue
                        prob += e[a, b] + e[a, c] + e[b, d] + e[c, d] <= 3

    # ── Weak connectivity via single-commodity flow ───────────────────────────
    # Vertex 0 is the source with supply (n-1); every other vertex is a sink
    # with demand 1.  Flow may traverse any undirected edge {i,j} in either
    # direction, but only when that edge exists (e[i,j] + e[j,i] == 1).
    f = {
        (i, j): pulp.LpVariable(f"f_{i}_{j}", lowBound=0, upBound=n - 1, cat="Continuous")
        for i in range(n) for j in range(n) if i != j
    }

    # Capacity: total flow across undirected edge {i,j} bounded by its existence
    for i in range(n):
        for j in range(i + 1, n):
            prob += f[i, j] + f[j, i] <= (n - 1) * (e[i, j] + e[j, i])

    # Flow conservation at source (vertex 0): net outflow = n-1
    prob += (
        pulp.lpSum(f[0, j] for j in range(1, n)) -
        pulp.lpSum(f[j, 0] for j in range(1, n))
    ) == n - 1

    # Flow conservation at every other vertex: net inflow = 1
    for v in range(1, n):
        prob += (
            pulp.lpSum(f[u, v] for u in range(n) if u != v) -
            pulp.lpSum(f[v, u] for u in range(n) if u != v)
        ) == 1

    # ── Solve ─────────────────────────────────────────────────────────────────
    print("Solving")
    prob.solve(pulp.GUROBI())

    print(f"n = {n}")
    print("Status:", pulp.LpStatus[prob.status])
    if prob.status == 1:
        print(f"Max edges: {int(pulp.value(prob.objective))}")
        print(f"Upper bound (balanced bipartite): {(n // 2) * (n - n // 2)}")
        print("Adjacency matrix:")
        for i in range(n):
            row = [int(pulp.value(e[i, j])) if i != j else 0 for j in range(n)]
            print(row)


if __name__ == "__main__":
    max_dag(40, upper_triangular=True, no_2_path=False, forbid_diamond=True)
