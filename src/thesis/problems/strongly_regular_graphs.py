"""Problem class for SRG problem."""

from typing import override

import torch

from .base_problem import BaseProblem
from .edge_utils import edge_vector_to_adjacency_matrix

# TODO frob norm is deprecated


class StronglyRegularGraphs(BaseProblem):
    """Optimization problem for finding strongly regular graphs.

    A (n,k,λ,μ)-strongly regular graph is a k-regular graph with n vertices
    where every pair of adjacent vertices has λ common neighbors and every
    pair of non-adjacent vertices has μ common neighbors.
    """

    n: int
    k: int
    lambda_param: int
    mu: int

    def __init__(
        self,
        n: int | None = None,
        k: int | None = None,
        lambda_param: int | None = None,
        mu: int | None = None,
        srg_params: tuple[int, int, int, int] | list[int] | None = None,
    ):
        """Initialize SRG problem with either individual parameters or tuple.

        Args:
            n: Number of vertices
            k: Degree of each vertex
            lambda_param: Number of common neighbors for adjacent vertices
            mu: Number of common neighbors for non-adjacent vertices
            srg_params: Tuple of (n, k, lambda_param, mu) - alternative to individual params
        """
        if srg_params is not None:
            if n is not None or k is not None or lambda_param is not None or mu is not None:
                raise ValueError("Cannot specify both individual parameters and srg_params tuple")
            self.n, self.k, self.lambda_param, self.mu = srg_params
        else:
            if n is None or k is None or lambda_param is None or mu is None:
                raise ValueError(
                    "Must specify either all individual parameters or srg_params tuple"
                )
            self.n = n
            self.k = k
            self.lambda_param = lambda_param
            self.mu = mu

    @override
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

        A = self._ensure_adjacency_matrix(x)

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

    @override
    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solutions represent valid adjacency matrices.

        Args:
            solution: Either a 2D tensor of shape (batch_size, edges) representing edge vectors,
                     or a 3D tensor of shape (batch_size, n, n) representing adjacency matrices.

        Returns
        -------
            Boolean tensor of shape (batch_size, 1): True if valid solution, False otherwise
        """
        A = self._ensure_adjacency_matrix(solution)

        # Check diagonal is zero
        diagonal_is_zero = (torch.diagonal(A, dim1=1, dim2=2) == 0).all(dim=1)

        is_symmetric = (A.transpose(-1, -2) == A).all(dim=(1, 2))

        # Check binary values (0 or 1)
        is_binary = ((A == 0) | (A == 1)).all(dim=(1, 2))

        return diagonal_is_zero & is_symmetric & is_binary

    def _ensure_adjacency_matrix(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Edge vector format: (batch_size, edges) - convert to adjacency matrix
            assert x.shape[1] == self.edges(), (
                f"Expected edge vector with {self.edges()} edges, got {x.shape}"
            )
            adj_matrix = edge_vector_to_adjacency_matrix(x, self.n)
        elif x.dim() == 3:
            # Matrix format: (batch_size, n, n)
            assert x.shape[1:] == (self.n, self.n), (
                f"Expected shape (*, {self.n}, {self.n}), got {x.shape}"
            )
            adj_matrix = x
        else:
            raise ValueError(
                f"Expected 2D (edge vector) or 3D (adjacency matrix) tensor, got {x.dim()}D"
            )
        return adj_matrix

    def edges(self) -> int:
        """Get number of edges for the current graph size."""
        return (self.n * (self.n - 1)) // 2

    @override
    def get_goal_score(self) -> float:
        """Return the goal score for SRG (perfect SRG has score 0)."""
        return 0.0
