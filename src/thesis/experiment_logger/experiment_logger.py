"""Simple unified logger that handles both console output and WandB logging."""

import os
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """Unified logger that sends output to both console and WandB.

    Usage:
        - Instantiate class: will make wandb api connection. At this point it can accept logs
        - Log starting metrics, like the full config of the experiment
        - Pass logger to algorithm
        - Call algorithm.optimize()
        - From within algorithm.optimize(), call configure_progress_bar() to pass postfix metrics and
          total iterations, so that the progress bar will look nice
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        project: str = "undergrad-thesis",
        wandb_mode: Literal["online", "offline", "disabled"] = "online",
        api_key: str | None = None,
        use_progress_bar: bool = True,
        **wandb_kwargs: Any,
    ):
        self.experiment_name = experiment_name
        self.use_wandb = wandb_mode != "disabled" and WANDB_AVAILABLE
        self.use_progress_bar = use_progress_bar

        # Progress bar setup
        self.progress_bar: tqdm | None = None
        self.total_iterations: int | None = None

        # WandB setup
        self.project = project
        self.wandb_mode = wandb_mode
        self.api_key = api_key
        self.wandb_kwargs = wandb_kwargs
        self.wandb_run = None

        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: WandB not available, skipping WandB logging")
                self.use_wandb = False
                return

            try:
                # Handle authentication for HPC environments
                if self.api_key:
                    wandb.login(key=self.api_key)
                elif os.getenv("WANDB_API_KEY"):
                    wandb.login(key=os.getenv("WANDB_API_KEY"))
                else:
                    try:
                        wandb.login()
                    except Exception:
                        print("Warning: WandB authentication failed. Running in offline mode.")
                        self.wandb_mode = "offline"

                self.wandb_run = wandb.init(
                    project=self.project,
                    name=self.experiment_name,
                    mode=self.wandb_mode,
                    **self.wandb_kwargs,
                )
                # Randomly generated if omitted
                self.experiment_name = self.wandb_run.name
            except Exception as e:
                print(f"Warning: WandB setup failed ({e}), continuing without")
                self.use_wandb = False

    def configure_progress_bar(
        self, postfix_metrics: list[str] | None = None, total_iterations: int | None = None
    ) -> None:
        """
        Configure progress bar with algorithm-specific metrics and iteration count.

        This can be called multiple times to reconfigure the progress bar
        with different metrics or iteration counts as needed.
        """
        print(f"Configuring progress bar for experiment: {self.experiment_name or 'unnamed'}")
        if self.use_progress_bar:
            self._setup_progress_bar(postfix_metrics, total_iterations)

    def _setup_progress_bar(
        self, postfix_metrics: list[str] | None, total_iterations: int | None = None
    ) -> None:
        """Setup progress bar for training loops."""
        self.postfix_metrics = postfix_metrics
        if self.use_progress_bar:
            self.total_iterations = total_iterations
            self.progress_bar = tqdm(
                total=total_iterations,
                desc=self.experiment_name or None,
                position=0,
                leave=True,
                dynamic_ncols=True,
            )

    def log_metrics(self, metrics: dict[str, Any], current_iteration: int) -> None:
        """Log metrics to both console (via progress bar) and WandB.

        Args:
            metrics: dictionary of metrics to log
            current_iteration: current iteration/step number (absolute, not incremental)
        """
        self._log_metrics_wandb(metrics, current_iteration)
        self._log_metrics_progress_bar(metrics, current_iteration)

    def _log_metrics_wandb(self, metrics: dict[str, Any], current_iteration: int) -> None:
        if not self.use_wandb:
            return  # TODO add proper linting for docstrings
            # TODO fix issues with autoformatting
            # TODO add check that errors for functions without type signatures
        assert self.wandb_run is not None
        self.wandb_run.log(metrics, step=current_iteration)

    def _log_metrics_progress_bar(self, metrics: dict[str, Any], current_iteration: int) -> None:
        if not self.use_progress_bar:
            return
        if self.progress_bar is None:
            total_iter = getattr(self, "total_iterations", None)
            self._setup_progress_bar(list(metrics.keys()), total_iter)
            assert self.progress_bar is not None

        # Filter Metrics
        filtered_metrics = {}
        if self.postfix_metrics is None:
            filtered_metrics = metrics
        else:
            for key, value in metrics.items():
                if key in self.postfix_metrics:
                    filtered_metrics[key] = value

        # Format Metrics
        formatted_metrics = {}
        for key, value in filtered_metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 1000:
                    formatted_metrics[key] = f"{value:.2e}"
                else:
                    formatted_metrics[key] = f"{value:.4f}"
            elif isinstance(value, bool):
                formatted_metrics[key] = int(value)  # Convert boolean to 0/1
            else:
                formatted_metrics[key] = value

        # Update progress bar to current iteration
        self.progress_bar.n = current_iteration  # TODO see if this needs a +1

        self.progress_bar.set_postfix(formatted_metrics)
        self.progress_bar.refresh()

    def log_info(self, message: str) -> None:
        """Log info message to console."""
        print(f"[INFO] {message}")

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to console and WandB."""
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        if self.use_wandb and self.wandb_run:
            wandb.config.update(params)

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log model to WandB (console just prints confirmation)."""
        print(f"Saving model: {model_name}")

        if self.use_wandb and self.wandb_run:
            # Save model locally first
            model_path = f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)

            # Create WandB artifact
            artifact = wandb.Artifact(model_name, type="model", metadata=metadata)
            artifact.add_file(model_path)
            self.wandb_run.log_artifact(artifact)

            # Clean up
            Path(model_path).unlink(missing_ok=True)

    def log_artifact(self, file_paths: str | Path, artifact_type: str = "general") -> None:
        """Log file artifact to WandB."""
        print(f"Logging artifact: {file_paths}")

        if self.use_wandb and self.wandb_run:
            artifact = wandb.Artifact(f"{artifact_type}-artifact", type=artifact_type)
            artifact.add_file(str(file_paths))
            self.wandb_run.log_artifact(artifact)

    def log_construction(
        self,
        construction: torch.Tensor,
        score: float,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log problem-specific constructions."""
        if self.use_wandb and self.wandb_run:
            # Log the score as a metric
            metrics = {"construction_score": score}
            if metadata:
                # Add metadata as metrics if they're scalar values
                for key, value in metadata.items():
                    if isinstance(value, (int | float)):
                        metrics[f"construction_{key}"] = value

            self.wandb_run.log(metrics, step=step)

            # Optionally save construction tensor as artifact
            if step is not None:
                construction_path = f"construction_step_{step}.pt"
                torch.save(construction, construction_path)
                artifact = wandb.Artifact(f"construction-{step}", type="construction")
                artifact.add_file(construction_path)
                self.wandb_run.log_artifact(artifact)
                Path(construction_path).unlink(missing_ok=True)

    def log_experiment_start(self, config: dict[str, Any]) -> None:
        """Log experiment start."""
        self.log_info(f"Starting experiment: {self.experiment_name or 'unnamed'}")
        self.log_hyperparameters(config)

    def log_experiment_end(self, results: dict[str, Any], success: bool = True) -> None:
        """Log experiment completion."""
        # Filter serializable results
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, int | float | str | bool) or value is None:
                serializable_results[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                serializable_results[key] = value.item()

        print("Final Results:")
        for key, value in serializable_results.items():
            print(f"  {key}: {value}")

        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(serializable_results)

        status = "SUCCESS" if success else "COMPLETED"
        self.log_info(f"Experiment {status}")

    def __enter__(self):
        """Context manager entry - logger is ready to use."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.close()

        if self.use_wandb and self.wandb_run:
            wandb.finish()
            self.wandb_run = None


if __name__ == "__main__":
    from time import sleep

    logger = ExperimentLogger(project="test", wandb_mode="online")
    iters = 100
    logger.configure_progress_bar(total_iterations=iters)
    for i in range(iters):
        logger.log_metrics({"sample": 1}, i)
        sleep(0.25)
