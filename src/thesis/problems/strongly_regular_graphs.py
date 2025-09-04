"""Problem class for SRG problem."""

import torch

from .base_problem import BaseProblem
from .edge_utils import edge_vector_to_adjacency_matrix


class StronglyRegularGraphs(BaseProblem):
    """Optimization problem for finding strongly regular graphs.

    A (n,k,λ,μ)-strongly regular graph is a k-regular graph with n vertices
    where every pair of adjacent vertices has λ common neighbors and every
    pair of non-adjacent vertices has μ common neighbors.
    """

    def __init__(self, n: int, k: int, lambda_param: int, mu: int):
        self.n = n
        self.k = k
        self.lambda_param = lambda_param
        self.mu = mu

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the negative squared Frobenius norm of the constraint residual.

        For a $(n,k,\lambda,\mu)$-SRG, the adjacency matrix $A$ must satisfy:

        .. math::
            A^2 + (\mu - \lambda)A + (\mu - k)I = \mu J

        where $J$ is the all-ones matrix. This function returns:

        .. math::
            - \| A^2 + (\mu - \lambda)A + (\mu - k)I - \mu J \|_F^2

        Args:
            x: Either a 2D tensor of shape (batch_size, edges) representing edge vectors,
               or a 3D tensor of shape (batch_size, n, n) representing adjacency matrices.
        """
        batch_size = x.shape[0]
        edges = (self.n * (self.n - 1)) // 2  # Number of edges in upper triangular matrix

        # Handle different input formats
        if x.dim() == 2:
            # Edge vector format: (batch_size, edges) - convert to adjacency matrix
            assert x.shape[1] == edges, f"Expected edge vector with {edges} edges, got {x.shape}"
            A = edge_vector_to_adjacency_matrix(x, self.n)
        elif x.dim() == 3:
            # Matrix format: (batch_size, n, n)
            assert x.shape[1:] == (self.n, self.n), (
                f"Expected shape (*, {self.n}, {self.n}), got {x.shape}"
            )
            A = x
        else:
            raise ValueError(
                f"Expected 2D (edge vector) or 3D (adjacency matrix) tensor, got {x.dim()}D"
            )
        A2 = A @ A
        mu_lambda_A = (self.mu - self.lambda_param) * A
        I = torch.eye(self.n, device=A.device, dtype=A.dtype).expand(batch_size, -1, -1)  # noqa: E741
        mu_k_I = (self.mu - self.k) * I
        mu_J = self.mu * torch.ones(batch_size, self.n, self.n, device=x.device, dtype=x.dtype)

        # A^2 + (μ - λ)A + (μ - k)I - μJ
        residual = A2 + mu_lambda_A + mu_k_I - mu_J

        # Return negative of squared Frobenius norm (higher is better, perfect SRG = 0)
        # dim=(1, 2) specifies the row and column dimension of the matrix
        return -(torch.frobenius_norm(residual, dim=(1, 2)) ** 2)

    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solutions represent valid adjacency matrices.

        Args:
            solution: Either a 2D tensor of shape (batch_size, edges) representing edge vectors,
                     or a 3D tensor of shape (batch_size, n, n) representing adjacency matrices.
        """
        btatch_size = solution.shape[0]
        edges = (self.n * (self.n - 1)) // 2  # Number of edges in upper triangular matrix

        # Handle different input formats
        if solution.dim() == 2:
            # Edge vector format: (batch_size, edges) - convert to adjacency matrix
            assert solution.shape[1] == edges, (
                f"Expected edge vector with {edges} edges, got {solution.shape}"
            )
            A = edge_vector_to_adjacency_matrix(solution, self.n)
        elif solution.dim() == 3:
            # Matrix format: (batch_size, n, n)
            assert solution.shape[1:] == (self.n, self.n), (
                f"Expected shape (*, {self.n}, {self.n}), got {solution.shape}"
            )
            A = solution
        else:
            raise ValueError(
                f"Expected 2D (edge vector) or 3D (adjacency matrix) tensor, got {solution.dim()}D"
            )

        # Check diagonal is zero
        diagonal_zero = (torch.diagonal(A, dim1=1, dim2=2) == 0).all(dim=1)

        # Check symmetry (element-wise for each matrix in batch)
        symmetric = torch.isclose(A, A.transpose(-1, -2), atol=1e-6).all(dim=(1, 2))

        # Check binary values (0 or 1)
        binary_values = ((A == 0) | (A == 1)).all(dim=(1, 2))

        return diagonal_zero & symmetric & binary_values

    def get_goal_score(self) -> float:
        """Return the goal score for SRG (perfect SRG has score 0)."""
        return 0.0

    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early (inclusive comparison for SRG)."""
        if best_score >= 0.0:
            return True, f"Perfect SRG found: {best_score:.6f} >= 0.0"
        return False, ""
