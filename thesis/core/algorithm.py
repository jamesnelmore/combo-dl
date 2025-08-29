import torch
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from .problem import BaseProblem

class BaseAlgorithm(ABC):
    """Base class for all algorithms."""
    
    def __init__(self, model: torch.nn.Module, problem: BaseProblem, loggers: List | None = None):
        self.model = model
        self.problem = problem
        self.loggers = loggers or []

    @abstractmethod
    def optimize(self, **kwargs) -> dict:
        """Train the model to solve the problem."""
        pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics to all attached loggers."""
        for logger in self.loggers:
            # Don't fail experiment if logging fails
            try:
                logger.log_metrics(metrics, step)
            except Exception:
                continue
    
    def log_info(self, message: str) -> None:
        """Log info message to all attached loggers."""
        for logger in self.loggers:
            # Don't fail experiment if logging fails
            try:
                logger.info(message)
            except Exception:
                continue
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all attached loggers."""
        for logger in self.loggers:
            try:
                logger.log_hyperparameters(params)
            except Exception:
                continue