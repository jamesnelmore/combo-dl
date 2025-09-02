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
        self.logger = logger

    @abstractmethod
    def optimize(self, **kwargs) -> dict[str, Any]:
        """Train the model to solve the problem."""
        pass
