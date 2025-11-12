"""Deep Cross-Entropy Algorithm."""

from collections.abc import Callable
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from .models import MLP
from .strongly_regular_graphs_problem import StronglyRegularGraphs


class WagnerDeepCrossEntropy:
    """Deep Cross Entropy Algorithm.

    Implementation of Deep Cross Entropy from [Wagner 2021](http://arxiv.org/abs/2104.14516).
    """

    model: MLP

    def __init__(
        self,
        model: MLP,
        problem: StronglyRegularGraphs,
        iterations: int = 10_000,
        batch_size: int = 512,
        learning_rate: float = 0.0001,
        elite_proportion: float = 0.1,
        device: str = "cpu",
        early_stopping_patience: int = 300,
        use_wandb: bool = True,
        experiment_name: str | None = None,
        save_dir: str = "runs",
        checkpoint_frequency: int = 100,
        save_best_constructions: bool = True,
        hydra_cfg: Any | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        survivor_proportion: float = 0.0,
        torch_compile: bool = False,
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
            early_stopping_patience: Iterations to wait before early stopping
            use_wandb: Whether to use Weights & Biases logging
            experiment_name: Custom experiment name (overrides auto-naming)
            save_dir: Base directory for saving experiments
            checkpoint_frequency: How often to save model checkpoints
            save_best_constructions: Whether to save best constructions
            hydra_cfg: Hydra configuration (if using Hydra)
            scheduler: Learning rate scheduler instance (optional)
            survivor_proportion: Fraction of constructions to keep as survivors each iteration
            torch_compile: Whether to compile the iteration step with torch.compile
        """
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.problem = problem

        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_frequency = checkpoint_frequency
        self.save_best_constructions = save_best_constructions
        self.hydra_cfg = hydra_cfg

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = scheduler

        # Optimization State (we want to maximize)
        self.best_score = float("-inf")
        self.best_construction = torch.ones(self.problem.n, device=self.device, dtype=torch.long)
        self.samples_seen = 0

        # Early stopping tracking
        self.steps_since_best = 0
        self.best_score_iteration = 0

        # Survivors
        self.survivor_proportion = survivor_proportion
        self.survivor_count = 0
        self.survivors = None
        self.survivor_scores = None

        self._run_iteration: Callable[[], dict[str, float]]
        if torch_compile:
            self._run_iteration = torch.compile(self._run_iteration_impl)
        else:
            self._run_iteration = self._run_iteration_impl

        # Weights and Biases - lazy import to avoid slow startup
        if use_wandb:
            import wandb  # noqa: PLC0415

            print("initialized")
            self.wandb_run = wandb.init(
                project="combo-dl",
                name=experiment_name,
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "elite_proportion": self.elite_proportion,
                    "early_stopping_patience": self.early_stopping_patience,
                    "max_iterations": self.iterations,
                    "srg_parameters": {
                        "n": self.problem.n,
                        "k": self.problem.k,
                        "lambda": self.problem.lambda_param,
                        "mu": self.problem.mu,
                    },
                },
            )
        else:
            print("didnt")
            self.wandb_run = None

        # Setup experiment directory
        self.experiment_name = self._get_experiment_name(experiment_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(save_dir) / f"{timestamp}_{self.experiment_name}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "best_constructions").mkdir(exist_ok=True)

        # Save initial config
        self._save_config()

    def _edges(self) -> int:
        if hasattr(self.problem, "edges"):
            return self.problem.edges()
        return 0

    def _get_experiment_name(self, custom_name: str | None) -> str:
        """Get experiment name with priority: custom > hydra > wandb > auto.

        Returns:
            Base experiment name (timestamp added to directory name)
        """
        if custom_name:
            return custom_name

        if self.hydra_cfg and hasattr(self.hydra_cfg, "experiment_name"):
            return self.hydra_cfg.experiment_name

        if self.wandb_run and self.wandb_run.name:
            return self.wandb_run.name

        return "experiment"

    def _save_config(self) -> None:
        """Save configuration files."""
        if self.hydra_cfg:
            with (self.experiment_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.dump(self.hydra_cfg, f, default_flow_style=False)

        config = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "elite_proportion": self.elite_proportion,
            "early_stopping_patience": self.early_stopping_patience,
            "max_iterations": self.iterations,
            "srg_parameters": {
                "n": self.problem.n,
                "k": self.problem.k,
                "lambda": self.problem.lambda_param,
                "mu": self.problem.mu,
            },
        }

        with (self.experiment_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.experiment_dir / "checkpoints" / f"checkpoint_{iteration:06d}.pt"
        checkpoint_dict = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "best_construction": (
                self.best_construction.cpu() if self.best_construction is not None else None
            ),
            "metrics": metrics,
            "config": {
                "n": self.problem.n,
                "k": self.problem.k,
                "lambda": self.problem.lambda_param,
                "mu": self.problem.mu,
            },
        }
        if self.scheduler is not None:
            checkpoint_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"  → Saved checkpoint: {checkpoint_path.name}")

    def _save_construction(self, iteration: int) -> None:
        """Save construction."""
        construction_path = (
            self.experiment_dir
            / "best_constructions"
            / f"best_iter_{iteration:06d}_score_{self.best_score:.4f}.pt"
        )
        torch.save(
            {
                "construction": self.best_construction.cpu(),
                "score": self.best_score,
                "iteration": iteration,
                "config": {
                    "n": self.problem.n,
                    "k": self.problem.k,
                    "lambda": self.problem.lambda_param,
                    "mu": self.problem.mu,
                },
            },
            construction_path,
        )
        print(f"  → Saved best construction: {construction_path.name}")

    def optimize(self) -> dict[str, Any]:
        """Run the full Deep Cross-Entropy optimization.

        Returns:
            Dictionary of optimization results
        """
        stop_early = False
        final_metrics = None
        p_bar = tqdm(total=self.iterations)

        for iteration in range(self.iterations):
            metrics = self.run_iteration()
            final_metrics = metrics
            num_elites = metrics["num_elites"]

            if metrics["found_new_best"]:
                self.steps_since_best = 0
                self.best_score_iteration = iteration
                # Save best construction when found
                if self.save_best_constructions:
                    self._save_construction(iteration)
            else:
                self.steps_since_best += 1

            should_stop, reason = self.problem.should_stop_early(metrics["best_score"])
            if should_stop:
                print(f"Stopping early: {reason}")
                stop_early = True
                break
            if num_elites == 0:
                print("No elites returned. This should not happen. Stopping")
                stop_early = True

            if self.steps_since_best >= self.early_stopping_patience:
                print(f"No improvement for {self.steps_since_best} steps. Stopping")
                stop_early = True
                break

            # Progress bar
            p_bar.update(1)
            p_bar.set_postfix({
                "best": metrics["best_score"],
                "avg": metrics["avg_score"],
                "accuracy": f"{metrics['accuracy']: .3f}",
                "loss": metrics["loss"],
                "graphs_seen": metrics["samples_seen"],
                "steps_since_best": self.steps_since_best,
            })

            # Step learning rate scheduler
            if self.scheduler is not None:
                # ReduceLROnPlateau needs a metric value, others just need step()
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use avg_score (current batch performance) for ReduceLROnPlateau
                    # Allows scheduler to react to current performance, not just historical best
                    self.scheduler.step(metrics["avg_score"])
                else:
                    self.scheduler.step()

            # Log to WandB
            if self.wandb_run:
                self.wandb_run.log(metrics)

            # Save checkpoint if needed
            if iteration % self.checkpoint_frequency == 0:
                self._save_checkpoint(iteration, metrics)

            if stop_early:
                break

        p_bar.close()

        final_results = {
            "best_score": self.best_score,
            "best_construction": (
                self.best_construction.cpu() if self.best_construction is not None else None
            ),
            "num_elites": num_elites,
            "final_metrics": final_metrics,
            "early_stopped": stop_early,
            "iterations": iteration,
            "samples_seen": self.samples_seen,
            "best_score_iteration": self.best_score_iteration,
            "steps_since_best": self.steps_since_best,
            "config": {
                "n": self.problem.n,
                "k": self.problem.k,
                "lambda": self.problem.lambda_param,
                "mu": self.problem.mu,
            },
        }

        with (self.experiment_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, default=str)

        print()
        print("=" * 50)
        print("Training complete!")
        print(f"Best score: {self.best_score:.2f}")
        print(f"Experiment directory: {self.experiment_dir}")
        print()

        return final_results

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration (compiled if configured)."""
        return self._run_iteration()

    def _run_iteration_impl(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics.

        Returns:
            Training metrics
        """
        current_survivor_count = self.survivors.shape[0] if self.survivors is not None else 0
        samples_to_generate = self.batch_size - current_survivor_count
        new_samples = self.model.sample(samples_to_generate)
        new_sample_scores = self.problem.reward(new_samples)

        if self.survivors is not None:
            constructions = torch.cat([self.survivors, new_samples])
            # scores is None iff survivors is None
            construction_scores = torch.cat([self.survivor_scores, new_sample_scores])  # pyright: ignore[reportArgumentType]
        else:
            constructions = new_samples
            construction_scores = new_sample_scores

        self.samples_seen += constructions.shape[0]

        current_best = torch.max(construction_scores).item()
        avg_score = torch.mean(construction_scores).item()

        found_new_best = False
        if current_best > self.best_score:
            self.best_score = current_best
            best_idx = torch.argmax(construction_scores)
            self.best_construction = constructions[best_idx].clone()
            found_new_best = True

        elites, elite_indices = self.select_elites(constructions, construction_scores)

        if self.survivor_proportion != 0.0:
            survivor_count = int(len(constructions) * self.survivor_proportion)
            survivor_indices = elite_indices[:survivor_count]
            self.survivors = constructions[survivor_indices]
            self.survivor_scores = construction_scores[survivor_indices]
            self.survivor_count = survivor_count
        else:
            self.survivor_count = 0

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]

        if len(elites) == 0:
            return {
                "best_score": self.best_score,
                "avg_score": avg_score,
                "num_elites": 0,
                "found_new_best": found_new_best,
                "loss": float("nan"),
                "accuracy": float("nan"),
                "samples_seen": self.samples_seen,
                "learning_rate": current_lr,
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
            "learning_rate": current_lr,
            **train_metrics,
        }

    def select_elites(
        self,
        constructions: torch.Tensor,
        batch_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top elite constructions based on scores (highest for maximization).

        Returns:
            Top self.elite_proportion of constructions, elite indices in construction tensor
        """
        batch_size = len(batch_scores)
        return_count = int(batch_size * self.elite_proportion)
        # descending for maximization
        elite_indices = torch.argsort(batch_scores, descending=True)[:return_count]
        return constructions[elite_indices], elite_indices

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
