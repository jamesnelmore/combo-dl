"""Problem class for SRG problem."""

import torch

from .base_problem import BaseProblem


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

    def score(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the negative squared Frobenius norm of the constraint residual.

        For a $(n,k,\lambda,\mu)$-SRG, the adjacency matrix $A$ must satisfy:

        .. math::
            A^2 + (\mu - \lambda)A + (\mu - k)I = \mu J

        where $J$ is the all-ones matrix. This function returns:

        .. math::
            - \| A^2 + (\mu - \lambda)A + (\mu - k)I - \mu J \|_F^2
        """
        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
        assert x.shape[1:] == (self.n, self.n), (
            f"Expected shape (*, {self.n}, {self.n}), got {x.shape}"
        )
        batch_size = x.shape[0]
        A = x
        A2 = A @ A
        mu_lambda_A = (self.mu - self.lambda_param) * A
        I = torch.eye(self.n, device=A.device, dtype=A.dtype).expand(batch_size, -1, -1)  # noqa: E741
        mu_k_I = (self.mu - self.k) * I
        mu_J = self.mu * torch.ones(batch_size, self.n, self.n, device=x.device, dtype=x.dtype)

        # A^2 + (μ - λ)A + (μ - k)I - μJ
        residual = A2 + mu_lambda_A + mu_k_I - mu_J

        # Return negative squared Frobenius norm
        # dim=(1, 2) specifies the row and column dimension of the matrix
        return -(torch.frobenius_norm(residual, dim=(1, 2)) ** 2)

    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solutions represent valid adjacency matrices."""
        # Check diagonal is zero
        diagonal_zero = (torch.diagonal(solution, dim1=1, dim2=2) == 0).all(dim=1)

        # Check symmetry (element-wise for each matrix in batch)
        symmetric = torch.isclose(solution, solution.transpose(-1, -2), atol=1e-6).all(dim=(1, 2))

        # Check binary values (0 or 1)
        binary_values = ((solution == 0) | (solution == 1)).all(dim=(1, 2))

        return diagonal_zero & symmetric & binary_values
