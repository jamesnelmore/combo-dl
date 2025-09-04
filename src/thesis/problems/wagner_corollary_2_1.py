import math
from typing import override
import networkx as nx
import torch

from .base_problem import BaseProblem

from edge_utils import edge_vector_to_adjacency_matrix

# TODO figuure out what this vibe coded scoring function does


class WagnerCorollary2_1(BaseProblem):
    def __init__(self, n: int):
        self.n = n
        self.edges = (n**2 - n) // 2
        self.goal_score = -(math.sqrt(n - 1) + 1)  # Negative because we want to maximize
        print(f"Goal score (-(sqrt({n - 1}) + 1)): {self.goal_score:.6f}")
        print(
            f"Searching for graphs with score > {self.goal_score:.6f} (eigenvalue + matching < {-self.goal_score:.6f})"
        )

    @override
    def reward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the score for each construction in the batch.

        Score = -(largest eigenvalue + matching number) (higher is better)

        Args:
            x: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph

        Returns
        -------
            Tensor of shape (batch_size,) with scores for each construction
            Higher scores are better (negative of eigenvalue + matching)
        """

        adj_matrix = edge_vector_to_adjacency_matrix(x, self.n)

        largest_eigenvalue = self._compute_largest_eigenvalue(adj_matrix)
        matching_number = self._compute_maximum_matching(adj_matrix)

        # Return negative to make higher scores better
        return -(largest_eigenvalue + matching_number)

    @override
    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early (exclusive comparison for Wagner)."""
        if best_score > self.goal_score:
            return (
                True,
                f"Wagner Corollary 2.1 goal achieved: {best_score:.6f} > {self.goal_score:.6f}",
            )
        return False, ""

    # def solution_space_info(self) -> dict:
    #     """Return information about the solution space."""
    #     return {
    #         "type": "tensor",
    #         "dim": self.edges,
    #         "dtype": torch.float32,
    #         "constraints": "binary",
    #         "description": f"Binary edge vector for {self.n}-node graph",
    #     } # TODO remove if no errors

    @override
    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solution is valid (binary values and correct dimension).

        Args:
            solution: Tensor of shape (batch_size, edges) where each entry should be 0 or 1

        Returns
        -------
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

    def _compute_largest_eigenvalue(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the largest eigenvalue of the adjacency matrix.

        Args:
            adj_matrix: Adjacency matrices of shape (batch_size, n, n)

        Returns
        -------
            Largest eigenvalue as tensor of shape (batch_size,)
        """
        # TODO find faster binary eigenvalue algorithm to avoid doubling memory usage
        original_device = adj_matrix.device

        # Convert to float and move to CPU for eigenvalue computation (eigenvals not supported on MPS)
        adj_float = adj_matrix.float()
        if adj_float.device.type == "mps":
            adj_float = adj_float.cpu()

        # Batch compute eigenvalues
        eigenvalues = torch.linalg.eigvals(adj_float).real  # Shape: (batch_size, n)

        # Get the largest eigenvalue for each matrix
        largest_eigenvalue = torch.max(eigenvalues, dim=1)[0]  # Shape: (batch_size,)

        return largest_eigenvalue.to(original_device)

    def _compute_maximum_matching(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the exact maximum matching number using NetworkX.

        Args:
            adj_matrix: Batch of adjacency matrices of shape (batch_size, n, n)

        Returns
        -------
            Maximum matching number as tensor of shape (batch_size,)
        """
        batch_size = adj_matrix.shape[0]
        device = adj_matrix.device
        matching_number = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Convert to numpy for NetworkX
        adj_np = adj_matrix.detach().cpu().numpy()

        # TODO multithread
        # Process each graph (this is still sequential but we batch the data prep)
        for i in range(batch_size):
            # Create graph from adjacency matrix
            G = nx.from_numpy_array(adj_np[i])

            # This uses Edmonds' blossom algorithm for general graphs
            matching = nx.max_weight_matching(G, maxcardinality=True)
            matching_number[i] = len(matching)

        return matching_number
