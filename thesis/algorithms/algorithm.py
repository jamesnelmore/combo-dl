import torch
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from thesis.problems.base_problem import BaseProblem
from thesis.experiment_logger.logger import BaseExperimentLogger

class BaseAlgorithm(ABC):
    """Base class for all algorithms."""
    
    def __init__(self, model: torch.nn.Module, problem: BaseProblem, logger: BaseExperimentLogger | None = None):
        self.model = model
        self.problem = problem
        self.logger = logger or []

    @abstractmethod
    def optimize(self, **kwargs) -> dict:
        """Train the model to solve the problem."""
        pass