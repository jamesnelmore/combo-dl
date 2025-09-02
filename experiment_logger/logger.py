from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import torch


class BaseExperimentLogger(ABC):
    """Base class for all loggers supporting ML experiment tracking."""

    def __init__(self, experiment_name: str | None = None, **kwargs):
        """Initialize logger with experiment name and configuration."""
        self.experiment_name = experiment_name
        self._is_setup = False

    @abstractmethod
    def setup(self, *args) -> None:
        """
        Setup logger with information from the algorithm.
        Should be run from inside the algorithm
        """
        pass

    @abstractmethod
    def _log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log scalar metrics (loss, accuracy, best_score, etc.).

        Args:
            metrics: Dictionary of metric_name -> scalar_value
            step: Optional step/iteration number for time series logging
        """
        pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._log_metrics(metrics, step)

    @abstractmethod
    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        Log experiment hyperparameters and configuration.

        Args:
            params: Dictionary of hyperparameter names and values
        """
        pass

    @abstractmethod
    def log_info(self, message: str) -> None:
        """
        Log text messages at different levels.

        Args:
            message: Text message to log
            level: Log level ("debug", "info", "warning", "error")
        """
        pass

    @abstractmethod
    def log_artifact(self, file_paths: str | Path, artifact_type: str = "general") -> None:
        """
        Log files as artifacts (models, plots, configs, etc.).

        Args:
            file_paths: List of file paths to upload/save
            artifact_type: Type of artifact ("model", "plot", "config", "data")
        """
        pass

    @abstractmethod
    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a PyTorch model.

        Args:
            model: The PyTorch model to save
            model_name: Name for the saved model
            metadata: Additional metadata about the model
        """
        pass

    @abstractmethod
    def log_construction(
        self,
        construction: torch.Tensor,
        score: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log problem-specific constructions (e.g., graph adjacency matrices).

        Args:
            construction: The construction tensor (e.g., edge vector)
            score: Score achieved by this construction
            step: Optional iteration step
            metadata: Additional metadata about the construction
        """
        pass

    def log_experiment_start(self, config: dict[str, Any]) -> None:
        """
        Log experiment start with full configuration.
        Default implementation logs config as hyperparameters.
        """
        if not self._is_setup:
            self.setup()
            self._is_setup = True

        self.log_hyperparameters(config)
        self.log_info(f"Starting experiment: {self.experiment_name or 'unnamed'}")

    def log_experiment_end(self, results: dict[str, Any], success: bool = False) -> None:
        """
        Log experiment completion with final results.
        Default implementation logs results as metrics and a completion message.
        """
        # Filter out non-serializable results (like tensors)
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (int | float | str | bool | None)):
                serializable_results[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                serializable_results[key] = value.item()

        self.log_metrics(serializable_results)
        status = "SUCCESS" if success else "COMPLETED"
        self.log_info(f"Experiment {status}: {self.experiment_name or 'unnamed'}")

    @abstractmethod
    def finalize(self) -> None:
        """Finalize logging (close files, upload pending data, etc.)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.finalize()


class MultiLogger(BaseExperimentLogger):
    """
    Composable logger that delegates operations to multiple BaseExperimentLogger instances.

    This allows combining different logging backends (e.g., file, console, wandb)
    into a single interface while maintaining the BaseExperimentLogger contract.
    """

    def __init__(
        self,
        loggers: list[BaseExperimentLogger],
        experiment_name: str | None = None,
        fail_fast: bool = False,
        **kwargs,
    ):
        """
        Initialize MultiLogger with a list of base loggers.

        Args:
            loggers: List of BaseExperimentLogger instances to compose
            experiment_name: Optional experiment name (passed to all loggers)
            fail_fast: If True, raise exceptions immediately. If False, log errors and continue.
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(experiment_name, **kwargs)
        self.loggers = loggers
        self.fail_fast = fail_fast
        self._logger = logging.getLogger(self.__class__.__name__)

        # Set experiment name on all composed loggers if provided
        if experiment_name:
            for logger in self.loggers:
                logger.experiment_name = experiment_name

    def _execute_on_all(self, method_name: str, *args, **kwargs) -> None:
        """
        Execute a method on all composed loggers with error handling.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        """
        errors = []

        for i, logger in enumerate(self.loggers):
            try:
                method = getattr(logger, method_name)
                method(*args, **kwargs)
            except Exception as e:
                error_msg = f"Logger {i} ({type(logger).__name__}) failed on {method_name}: {e}"
                errors.append(error_msg)

                if self.fail_fast:
                    raise RuntimeError(error_msg) from e
                self._logger.error(error_msg)

        if errors and not self.fail_fast:
            self._logger.warning(f"MultiLogger completed {method_name} with {len(errors)} errors")

    def setup(self) -> None:
        """Setup all composed loggers."""
        self._execute_on_all("setup")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to all composed loggers."""
        self._execute_on_all("log_metrics", metrics, step=step)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to all composed loggers."""
        self._execute_on_all("log_hyperparameters", params)

    def log_info(self, message: str, level: str = "info") -> None:
        """Log info messages to all composed loggers."""
        self._execute_on_all("log_info", message, level=level)

    def log_artifact(self, file_paths: str | Path, artifact_type: str = "general") -> None:
        """Log artifacts to all composed loggers."""
        self._execute_on_all("log_artifact", file_paths, artifact_type=artifact_type)

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log model to all composed loggers."""
        self._execute_on_all("log_model", model, model_name=model_name, metadata=metadata)

    def log_construction(
        self,
        construction: torch.Tensor,
        score: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log construction to all composed loggers."""
        self._execute_on_all("log_construction", construction, score, step=step, metadata=metadata)

    def finalize(self) -> None:
        """Finalize all composed loggers."""
        self._execute_on_all("finalize")

    def add_logger(self, logger: BaseExperimentLogger) -> None:
        """
        Add a new logger to the composition.

        Args:
            logger: BaseExperimentLogger instance to add
        """
        if self.experiment_name:
            logger.experiment_name = self.experiment_name

        # If MultiLogger is already setup, setup the new logger too
        if self._is_setup:
            try:
                logger.setup()
                logger._is_setup = True
            except Exception as e:
                error_msg = f"Failed to setup new logger {type(logger).__name__}: {e}"
                if self.fail_fast:
                    raise RuntimeError(error_msg) from e
                self._logger.error(error_msg)

        self.loggers.append(logger)

    def remove_logger(
        self, logger_index: int | BaseExperimentLogger
    ) -> BaseExperimentLogger | None:
        """
        Remove a logger from the composition.

        Args:
            logger_index: Index of logger to remove or logger instance itself

        Returns:
            The removed logger instance, or None if not found
        """
        if isinstance(logger_index, int):
            if 0 <= logger_index < len(self.loggers):
                return self.loggers.pop(logger_index)
        else:
            # Remove by instance
            try:
                self.loggers.remove(logger_index)
                return logger_index
            except ValueError:
                pass

        return None

    def get_loggers(self) -> list[BaseExperimentLogger]:
        """Get a copy of the current list of composed loggers."""
        return self.loggers.copy()

    def __len__(self) -> int:
        """Return the number of composed loggers."""
        return len(self.loggers)
