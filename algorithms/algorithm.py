from abc import ABC, abstractmethod

import torch

from experiment_logger.logger import BaseExperimentLogger
from problems.base_problem import BaseProblem


class BaseAlgorithm(ABC):
    """Base class for all algorithms."""

    def __init__(
        self,
        model: torch.nn.Module,
        problem: BaseProblem,
        logger: BaseExperimentLogger | None = None,
    ):
        self.model = model
        self.problem = problem
        self.logger = logger

    @abstractmethod
    def optimize(self, **kwargs) -> dict:
        """Train the model to solve the problem."""
        pass
