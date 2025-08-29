from abc import ABC, abstractmethod

import torch


class BaseProblem(ABC):
    """Defines an optimization goal and how to evaluate solutions."""

    @abstractmethod
    def score(self, solutions: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for each solution in the batch. Higher is better.

        Args:
            solutions: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph
        """
        pass

    @abstractmethod
    def is_valid_solution(self, solutions: torch.Tensor) -> torch.Tensor:
        """
        Whether each element in the batch is a valid solution to the problem.
        """
        pass
