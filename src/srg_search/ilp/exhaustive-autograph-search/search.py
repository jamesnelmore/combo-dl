"""
Verify Conjecture 2.1 from Wagner (2021):
  For connected G on n >= 3 vertices, lambda_1 + mu >= sqrt(n - 1) + 1.

By the edge-deletion argument (removing edges outside a maximum matching
while preserving connectivity can only decrease lambda_1 + mu), it suffices
to check trees. We enumerate all non-isomorphic trees for each n.
"""

import math
import numpy as np
import networkx as nx


def largest_eigenvalue(G: nx.Graph) -> float:
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(A)
    return float(eigenvalues[-1])


def matching_number(G: nx.Graph) -> int:
    return len(nx.max_weight_matching(G))


def check_conjecture(n_max: int = 18) -> None:
    for n in range(3, n_max + 1):
        threshold = math.sqrt(n - 1) + 1
        min_val = float("inf")
        min_tree = None
        count = 0

        for T in nx.nonisomorphic_trees(n):
            count += 1
            lam1 = largest_eigenvalue(T)
            mu = matching_number(T)
            val = lam1 + mu

            if val < min_val:
                min_val = val
                min_tree = T

        margin = min_val - threshold
        status = "PASS" if margin >= -1e-9 else "FAIL"

        print(
            f"n={n:2d} | trees: {count:>7d} | "
            f"min(λ₁+μ) = {min_val:.6f} | "
            f"threshold = {threshold:.6f} | "
            f"margin = {margin:+.6f} | {status}"
        )

        if status == "FAIL":
            print(f"  Counterexample edges: {list(min_tree.edges())}")


if __name__ == "__main__":
    check_conjecture()
