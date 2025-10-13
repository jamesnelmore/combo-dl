"""Example Corollary shown in Wagner 2021."""

import math

import torch

from combo_dl.graph_utils import (
    compute_largest_eigenvalue,
    compute_maximum_matching,
    edge_vec_to_adj,
)


class WagnerCorollary21:
    r"""Corollary 2.1 from [Wagner 2021](http://arxiv.org/abs/2104.14516).

    Example problem to demonstrate the DCE method.
    Goal is to construct a graph G with matching number $m$ and largest eigenvalue $\mu$ such that
    .. math:
        m + \mu < \sqrt{n - 1} + 1
    """

    def __init__(self, n: int):
        """Creates WagnerCorollary21 class.

        Args:
        n: number of vertices in target graph
        """
        self.n = n
        self.edges = (n**2 - n) // 2
        self.goal_score = -(math.sqrt(n - 1) + 1)  # Negative to maximize
        print(f"Goal score (-(sqrt({n - 1}) + 1)): {self.goal_score:.6f}")
        print(
            f"Searching for graphs with score > {self.goal_score:.6f} "
            f"(eigenvalue + matching < {-self.goal_score:.6f})"
        )

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the score for each construction in the batch.

        Score = -(largest eigenvalue + matching number) (higher is better)

        Args:
            x: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph

        Returns:
            Tensor of shape (batch_size,) with scores for each construction
            Higher scores are better (negative of eigenvalue + matching)
        """
        adj_matrix = edge_vec_to_adj(x, self.n)

        largest_eigenvalue = compute_largest_eigenvalue(adj_matrix)
        matching_number = compute_maximum_matching(adj_matrix)

        # Return negative to make higher scores better
        return -(largest_eigenvalue + matching_number)

    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early (exclusive comparison for Wagner).

        Returns:
        tuple[bool, str]
            Tuple of (should_stop, reason_message)
        """
        if best_score > self.goal_score:
            return (
                True,
                f"Wagner Corollary 2.1 goal achieved: {best_score:.6f} > {self.goal_score:.6f}",
            )
        return False, ""

    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solution is valid (binary values and correct dimension).

        Args:
            solution: Tensor of shape (batch_size, edges) where each entry should be 0 or 1

        Returns:
            Tensor of shape (batch_size,) with boolean values indicating validity
        """
        # Check that the tensor has the correct shape
        if solution.dim() != 2 or solution.shape[1] != self.edges:
            # Return False for all batch items if shape is wrong
            batch_size = solution.shape[0] if solution.dim() >= 1 else 1
            return torch.zeros(batch_size, dtype=torch.bool, device=solution.device)

        # Check that each solution vector contains only binary values (0 or 1)
        # Use dim=1 to check across the edges dimension for each batch item
        return torch.all((solution == 0) | (solution == 1), dim=1)
