import torch

from .base_problem import BaseProblem


class StronglyRegularGraphs(BaseProblem):
    def __init__(self, n: int, k: int, lambda_param: int, mu: int):
        self.n = n
        self.k = k
        self.lambda_param = lambda_param
        self.mu = mu

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the negative squared frobenius norm of each residual"""
        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
        assert x.shape[1:] == (self.n, self.n), (
            f"Expected shape (*, {self.n}, {self.n}), got {x.shape}"
        )
        batch_size = x.shape[0]
        A2 = x @ x
        mu_k_I = torch.eye(self.n) * (self.mu - self.k)
        muJ = torch.ones(batch_size, self.n) * self.mu

        # A² + (μ - λ)A + (μ - k)I - μJ
        residual = A2 + mu_k_I - muJ
        frobenius_norm = torch.frobenius_norm(residual, dim=(-2, 1))

        return -(frobenius_norm**2)
