import torch
from abc import ABC, abstractmethod

from thesis.core import BaseProblem

class BaseAlgorithm(ABC):
    """Base class for all algorithms."""
    
    def __init__(self, model: torch.nn.Module, problem: BaseProblem):
        self.model = model
        self.problem = problem

    @abstractmethod
    def optimize(self, **kwargs) -> dict:
        """Train the model to solve the problem."""
        pass