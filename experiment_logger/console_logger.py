from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .logger import BaseExperimentLogger


class ConsoleLogger(BaseExperimentLogger):
    """Console logger implementation with progress bar support using tqdm."""

    def __init__(self, experiment_name: str | None = None, **kwargs):
        super().__init__(experiment_name, **kwargs)
        self.total_iterations: int | None = None
        self.postfix_metrics: list[str] | None = None
        self.p_bar: tqdm | None = None
        self.current_iteration = 0

    def setup(self, postfix_metrics: list[str], total_iterations: int | None = None) -> None:
        """Setup the console logger. Run from inside the algorithm"""
        print(f"Setting up Console Logger for experiment: {self.experiment_name or 'unnamed'}")
        self.total_iterations = total_iterations
        self.postfix_metrics = postfix_metrics
        self.p_bar = tqdm(
            total=self.total_iterations,
            desc=f"Experiment: {self.experiment_name or ''}",
            unit="iter",
            ncols=100,
            position=0,
            leave=True,
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        assert self.p_bar is not None
        self.p_bar.update(step)
        self.p_bar.set_postfix(metrics)

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        print(params)

    def log_info(self, message: str, level: str = "info") -> None:
        print(f"[INFO] {message}")

    def log_artifact(self, file_paths: str | Path, artifact_type: str = "general") -> None:
        pass

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def log_construction(
        self,
        construction: torch.Tensor,
        score: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def log_experiment_start(self, config: dict[str, Any]) -> None:
        print(config)

    def log_experiment_end(self, results: dict[str, Any], success: bool = False) -> None:
        print(results)

    def finalize(self):
        pass


if __name__ == "__main__":
    logger = ConsoleLogger()
