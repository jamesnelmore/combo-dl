"""Deep Cross Entropy Algorithm."""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from .models import MLP
from .strongly_regular_graphs_problem import StronglyRegularGraphs


class WagnerDeepCrossEntropy:
    """Deep Cross Entropy Algorithm.

    Implementation of Deep Cross Entropy from [Wagner 2021](http://arxiv.org/abs/2104.14516).
    """

    model: MLP

    # TODO: Consider refactoring constructor to use a configuration object
    def __init__(
        self,
        model: MLP,
        problem: StronglyRegularGraphs,
        iterations: int = 10_000,
        batch_size: int = 512,
        learning_rate: float = 0.0001,
        elite_proportion: float = 0.1,
        device: str = "cpu",
        log_frequency: int = 1,
        model_save_frequency: int = 100,
        early_stopping_patience: int = 300,
        checkpoint_dir: str | Path | None = None,
        experiment_name: str | None = None,
        wandb_project: str = "combo-dl",
        wandb_mode: Literal["online", "offline"] = "online",
        use_progress_bar: bool = True,
    ):
        """Initialize Deep Cross Entropy algorithm.

        Args:
            model: The neural network model to train
            problem: The optimization problem to solve
            iterations: Maximum number of training iterations
            batch_size: Number of samples per iteration
            learning_rate: Learning rate for optimizer
            elite_proportion: Fraction of samples to use as elites
            device: Device to run on (cpu, cuda, mps)
            log_frequency: How often to log detailed metrics
            model_save_frequency: How often to save model checkpoints
            early_stopping_patience: Iterations to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this experiment run
            wandb_project: WandB project name
            wandb_mode: WandB logging mode (online/offline)
            use_progress_bar: Whether to show progress bar
        """
        self.model = model
        self.problem = problem

        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.log_frequency = log_frequency
        self.model_save_frequency = model_save_frequency
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.model.to(self.device)

        # WandB setup
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.wandb_run = None
        self.experiment_name = experiment_name

        # Progress bar setup
        self.use_progress_bar = use_progress_bar
        self.progress_bar: tqdm | None = None

        # Checkpoint setup
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints/")
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create run-specific checkpoint directory
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.experiment_name = f"dce_{timestamp}"

        self.run_checkpoint_dir = self.checkpoint_dir / self.experiment_name
        self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Optimization State (we want to maximize)
        self.best_score = float("-inf")
        self.best_construction = None
        self.samples_seen = 0

        # Early stopping tracking
        self.steps_since_best = 0
        self.best_score_iteration = 0

    def _setup_wandb(self) -> None:
        """Initialize WandB run."""
        try:
            # Handle authentication for HPC environments
            if os.getenv("WANDB_API_KEY"):
                wandb.login(key=os.getenv("WANDB_API_KEY"))
            else:
                try:
                    wandb.login()
                except Exception:
                    print("Warning: WandB authentication failed. Running in offline mode.")
                    self.wandb_mode = "offline"

            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                mode=self.wandb_mode,  # pyright: ignore[reportArgumentType]
                config={
                    "iterations": self.iterations,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "elite_proportion": self.elite_proportion,
                    "device": self.device,
                    "early_stopping_patience": self.early_stopping_patience,
                },
            )
            # Update experiment name if wandb generated one
            self.experiment_name = self.wandb_run.name
        except Exception as e:
            print(f"Warning: WandB setup failed ({e}), continuing without")
            self.wandb_run = None

    def _setup_progress_bar(self) -> None:
        """Setup progress bar for training."""
        if not self.use_progress_bar:
            return

        self.progress_bar = tqdm(
            total=self.iterations,
            desc=self.experiment_name or "DCE Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

    def _log_metrics(self, metrics: dict[str, Any], iteration: int) -> None:
        """Log metrics to WandB and progress bar."""
        # Log to WandB
        if self.wandb_run:
            self.wandb_run.log(metrics, step=iteration)

        if self.use_progress_bar and self.progress_bar:
            formatted_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) > 1000:
                        formatted_metrics[key] = f"{value:.2e}"
                    else:
                        formatted_metrics[key] = f"{value:.4f}"
                elif isinstance(value, bool):
                    formatted_metrics[key] = int(value)
                else:
                    formatted_metrics[key] = value

            self.progress_bar.n = iteration
            self.progress_bar.set_postfix(formatted_metrics)
            self.progress_bar.refresh()

    def _save_checkpoint(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Save model checkpoint and best construction."""
        # Get problem parameters
        problem_config = {
            "n": self.problem.n,
            "k": self.problem.k,
            "lambda": self.problem.lambda_param,
            "mu": self.problem.mu,
        }

        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "best_construction": (
                self.best_construction.cpu() if self.best_construction is not None else None
            ),
            "metrics": metrics,
            "config": {
                "iterations": self.iterations,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "elite_proportion": self.elite_proportion,
                "device": self.device,
                "early_stopping_patience": self.early_stopping_patience,
                **problem_config,  # Include problem parameters
            },
        }

        checkpoint_path = self.run_checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)

        if self.best_construction is not None:
            torch.save(checkpoint, self.run_checkpoint_dir / f"best_iter{iteration}.pt")

        print(f"  → Saved checkpoint: {checkpoint_path.name}")

    def optimize(self) -> dict[str, Any]:
        """Run the full Deep Cross-Entropy optimization.

        Returns:
            Dictionary of optimization results
        """
        # Setup logging and progress tracking
        self._setup_wandb()
        self._setup_progress_bar()

        print("Starting Deep Cross Entropy optimization")
        print(f"Experiment: {self.experiment_name}")
        print(f"Checkpoints: {self.run_checkpoint_dir}")
        print(f"Device: {self.device}")
        print()

        early_stopped = False
        final_metrics = None

        for iteration in range(self.iterations):
            metrics = self.run_iteration()
            final_metrics = metrics
            num_elites = metrics["num_elites"]
            del metrics["num_elites"]

            # Update early stopping tracking
            if metrics["found_new_best"]:
                self.steps_since_best = 0
                self.best_score_iteration = iteration
            else:
                self.steps_since_best += 1

            # Log metrics
            if iteration % self.log_frequency == 0:
                self._log_metrics(metrics, iteration)
            else:
                # Still update progress bar with minimal info
                minimal_metrics = {
                    "best_score": metrics["best_score"],
                    "samples_seen": metrics["samples_seen"],
                }
                self._log_metrics(minimal_metrics, iteration)

            # Log best construction when it improves
            if metrics["found_new_best"] and self.best_construction is not None and self.wandb_run:
                # Log construction score as metric
                construction_metrics = {"construction_score": self.best_score}
                if metrics.get("iteration"):
                    construction_metrics["construction_iteration"] = iteration
                self.wandb_run.log(construction_metrics, step=iteration)

            # Save checkpoint
            if iteration > 0 and iteration % self.model_save_frequency == 0:
                self._save_checkpoint(iteration, metrics)

            # Check if problem wants to stop early
            should_stop, stop_reason = self.problem.should_stop_early(metrics["best_score"])
            if should_stop:
                print(f"Early stopping at iteration {iteration}")
                print(stop_reason)
                early_stopped = True
                break

            # Early stopping condition: no improvement for patience steps
            if self.steps_since_best >= self.early_stopping_patience:
                print(
                    f"Early stopping at iteration {iteration}: "
                    f"no improvement for {self.early_stopping_patience} steps"
                )
                print(
                    f"Best score {self.best_score} was achieved at "
                    f"iteration {self.best_score_iteration}"
                )
                early_stopped = True
                break

            # Early stopping condition (optional)
            if num_elites == 0:
                print(f"Warning: No elites selected at iteration {iteration}")
                break

        # Save final checkpoint
        if final_metrics:
            self._save_checkpoint(iteration, final_metrics)

        print()
        print("=" * 50)
        print("Training complete!")
        print(f"Best score: {self.best_score:.2f}")
        print(f"Checkpoints saved to: {self.run_checkpoint_dir}")
        print()

        return {
            "best_score": self.best_score,
            # Move to cpu because Hydra cannot deserialize a CUDA object on the login node
            "best_construction": self.best_construction.cpu()
            if self.best_construction is not None
            else None,
            "num_elites": num_elites,
            "final_metrics": final_metrics,
            "early_stopped": early_stopped,
            "iterations": iteration,
            "samples_seen": self.samples_seen,
            "best_score_iteration": self.best_score_iteration,
            "steps_since_best": self.steps_since_best,
        }

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics.

        Returns:
            Training metrics
        """
        constructions = self.model.sample(self.batch_size)
        batch_scores = self.problem.reward(constructions)

        self.samples_seen += self.batch_size

        current_best = torch.max(batch_scores).item()
        avg_score = torch.mean(batch_scores).item()

        found_new_best = False
        if current_best > self.best_score:
            self.best_score = current_best
            best_idx = torch.argmax(batch_scores)
            self.best_construction = constructions[best_idx].clone()
            found_new_best = True

        elites = self.select_elites(constructions, batch_scores)

        if len(elites) == 0:
            print("Zero elites passed. This should not occur. Stopping run.")
            return {
                "best_score": self.best_score,
                "avg_score": avg_score,
                "num_elites": 0,
                "found_new_best": found_new_best,
                "loss": float("nan"),
                "accuracy": float("nan"),
                "samples_seen": self.samples_seen,
            }

        # Extract training examples and train
        dataloader = self.extract_examples(
            elites,
            output_batch_size=min(self.batch_size, len(elites) * elites.shape[1]),
        )
        train_metrics = self.supervised_train_step(dataloader)

        return {
            "best_score": self.best_score,
            "avg_score": avg_score,
            "num_elites": len(elites),
            "found_new_best": found_new_best,
            "samples_seen": self.samples_seen,
            **train_metrics,
        }

    def select_elites(
        self,
        constructions: torch.Tensor,
        batch_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select top elite constructions based on scores (highest for maximization).

        Returns:
            Top self.elite_proportion of constructions
        """
        batch_size = len(batch_scores)
        return_count = int(batch_size * self.elite_proportion)
        # descending for maximization
        elite_indices = torch.argsort(batch_scores, descending=True)[:return_count]
        return constructions[elite_indices]

    def extract_examples(
        self, constructions: torch.Tensor, output_batch_size: int | None = None
    ) -> DataLoader:
        """Create training examples from constructions.

        For each construction, mask it past i with 0s. Then the state is the masked
        construction, the position is i + 1, and the action is what was at i + 1
        before the mask.

        Returns:
            Input constructions converted to training examples and packaged into a dataloader
        """
        if output_batch_size is None:
            output_batch_size = self.batch_size

        num_constructions, num_edges = constructions.shape
        target_device = constructions.device

        # Create mask to hide future positions during training
        pos_tensor = torch.arange(num_edges).repeat(num_constructions).to(target_device)
        edge_indices = torch.arange(num_edges).to(target_device)
        mask = (edge_indices.unsqueeze(0) < pos_tensor.unsqueeze(1)).to(target_device)

        obs_tensor = constructions.repeat_interleave(num_edges, dim=0) * mask
        obs_tensor = obs_tensor.to(target_device)
        construction_indices = (
            torch.arange(num_constructions)
            .repeat_interleave(num_edges)
            .to(target_device)  # Defined elementwise: T[i] = elite_constructions[i][pos_tensor[i]]
        )
        actions_tensor = constructions[construction_indices, pos_tensor].to(
            target_device, dtype=torch.long
        )

        dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

        return dataloader

    def supervised_train_step(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """Perform one training step and return metrics.

        Returns:
            Training metrics
        """
        self.model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (obs, pos, target_actions) in enumerate(train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(obs, pos)
            loss = self.criterion(outputs, target_actions)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target_actions.size(0)
            train_correct += (predicted == target_actions).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            accuracy = train_correct / train_total if train_total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy, "total_samples": train_total}
