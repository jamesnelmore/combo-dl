"""Base class that problems are derived from."""

from abc import ABC, abstractmethod

import torch


class BaseProblem(ABC):
    """Defines an optimization goal and how to evaluate solutions."""

    @abstractmethod
    def reward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for each solution in the batch. Higher is better.

        Args:
            solutions: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph
        """
        pass

    @abstractmethod
    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Whether each element in the batch is a valid solution to the problem."""
        pass

    def get_goal_score(self) -> float | None:
        """Get the target score for early stopping, if any.

        Returns
        -------
            Target score for early stopping, or None if no early stopping desired.
        """
        return None

    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early based on the current best score.

        Args:
            best_score: Current best score achieved

        Returns
        -------
            Tuple of (should_stop, reason_message)
        """
        goal_score = self.get_goal_score()
        if goal_score is None:
            return False, ""

        # Default implementation - subclasses can override for custom logic
        if best_score >= goal_score:
            return True, f"Goal achieved: {best_score:.6f} >= {goal_score:.6f}"
        return False, ""
