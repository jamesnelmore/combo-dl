from .logger import BaseExperimentLogger
from pathlib import Path
import torch

class ConsoleLogger(BaseExperimentLogger):
    """Simple console logger implementation for testing MultiLogger."""
    
    def setup(self) -> None:
        print(f"[Console] Setting up logger for experiment: {self.experiment_name}")
    
    def log_metrics(self, metrics: dict[str, any], step: int | None = None) -> None:
        step_str = f" (step {step})" if step is not None else ""
        print(f"[Console] Metrics{step_str}: {metrics}")
    
    def log_hyperparameters(self, params: dict[str, any]) -> None:
        print(f"[Console] Hyperparameters: {params}")
    
    def log_info(self, message: str, level: str = "info") -> None:
        print(f"[Console] {level.upper()}: {message}")
    
    def log_artifact(self, file_paths: str | Path, artifact_type: str = "general") -> None:
        print(f"[Console] Artifact ({artifact_type}): {file_paths}")
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model", 
                  metadata: dict[str, any] | None = None) -> None:
        print(f"[Console] Model '{model_name}': {type(model).__name__}")
    
    def log_construction(self, construction: torch.Tensor, score: float, 
                        step: int | None = None, metadata: dict[str, any] | None = None) -> None:
        step_str = f" (step {step})" if step is not None else ""
        print(f"[Console] Construction{step_str}: score={score}, shape={construction.shape}")
    
    def finalize(self) -> None:
        print(f"[Console] Finalizing logger for experiment: {self.experiment_name}")