from abc import ABC, abstractmethod
from typing import Any

import torch

from experiment_logger import ExperimentLogger
from problems.base_problem import BaseProblem


class BaseAlgorithm(ABC):
    """Base class for all algorithms."""

    def __init__(
        self,
        model: torch.nn.Module,
        problem: BaseProblem,
        logger: ExperimentLogger | None = None,
    ):
        self.model = model
        self.problem = problem

        if logger is None:
            logger = ExperimentLogger(wandb_mode="disabled")
        self.logger = logger

    @abstractmethod
    def optimize(self, **kwargs) -> dict[str, Any]:
        """
        Train the model to solve the problem.

        Args:
            progress_callback: Optional callback function that receives (iteration, metrics)
            **kwargs: Algorithm-specific parameters

        Returns
        -------
            Dictionary containing optimization results, typically including:
                - best_score: The best score achieved during optimization
                - best_construction: The best solution found (if applicable)
                - early_stopped: Whether optimization stopped early
                - iterations: Number of iterations completed
                - final_metrics: Final iteration metrics
        """
        pass
