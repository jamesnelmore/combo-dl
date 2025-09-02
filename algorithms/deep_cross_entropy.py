from datetime import datetime
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_logger import ExperimentLogger
from models.protocols import SamplingModel
from problems.base_problem import BaseProblem

from .algorithm import BaseAlgorithm


class WagnerDeepCrossEntropy(BaseAlgorithm):
    model: SamplingModel

    def __init__(
        self,
        model: SamplingModel,
        problem: BaseProblem,
        iterations: int = 10_000,
        batch_size: int = 512,
        learning_rate: float = 0.0001,
        elite_proportion: float = 0.1,
        goal_score: float | None = None,
        device: str | None = None,
        logger: ExperimentLogger | None = None,
    ):
        super().__init__(model, problem, logger)
        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.goal_score = goal_score
        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )
        self.model.to(self.device)

        if logger is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_name = f"Deep Cross Entropy {date} {type(problem).__name__}"
            self.logger: ExperimentLogger = ExperimentLogger(experiment_name, use_wandb=False)
        else:
            self.logger = logger
        metrics = ["best_score", "avg_score", "loss", "accuracy"]
        self.logger.setup(metrics, self.iterations)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Track optimization state (we want to minimize, so start with +inf)
        self.best_score = float("inf")
        self.best_construction = None

    def select_elites(
        self,
        constructions: torch.Tensor,
        batch_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select top elite constructions based on scores (lowest for minimization).
        """
        batch_size = len(batch_scores)
        return_count = int(batch_size * self.elite_proportion)
        elite_indices = torch.argsort(batch_scores, descending=False)[
            :return_count
        ]  # ascending for minimization
        return constructions[elite_indices]

    def extract_examples(
        self, elite_constructions: torch.Tensor, output_batch_size: int | None = None
    ) -> DataLoader:
        """
        For each construction, mask it past i with 0s. Then the state is the masked
        construction, the position is i + 1, and the action is what was at i + 1
        before the mask.
        """
        if output_batch_size is None:
            output_batch_size = self.batch_size

        num_constructions, num_edges = elite_constructions.shape
        target_device = elite_constructions.device

        # Create mask to hide future positions during training
        pos_tensor = torch.arange(num_edges).repeat(num_constructions).to(target_device)
        edge_indices = torch.arange(num_edges).to(target_device)
        mask = (edge_indices.unsqueeze(0) < pos_tensor.unsqueeze(1)).to(target_device)

        obs_tensor = elite_constructions.repeat_interleave(num_edges, dim=0) * mask
        obs_tensor = obs_tensor.to(target_device)
        construction_indices = (
            torch.arange(num_constructions)
            .repeat_interleave(num_edges)
            .to(target_device)  # Defined elementwise: T[i] = elite_constructions[i][pos_tensor[i]]
        )
        actions_tensor = elite_constructions[construction_indices, pos_tensor].to(target_device)

        dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

        return dataloader

    def supervised_train_step(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """Perform one training step and return metrics."""

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

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics."""
        # Generate constructions and score them
        constructions = self.model.sample(self.batch_size)
        batch_scores = self.problem.score(constructions)

        # Track best score and construction (minimization)
        current_best = torch.min(batch_scores).item()
        avg_score = torch.mean(batch_scores).item()
        found_new_best = False

        if current_best < self.best_score:
            self.best_score = current_best
            best_idx = torch.argmin(batch_scores)
            self.best_construction = constructions[best_idx].clone()
            found_new_best = True

        # Select elites
        elites = self.select_elites(constructions, batch_scores)

        if len(elites) == 0:
            return {
                "best_score": self.best_score,
                "avg_score": avg_score,
                "num_elites": 0,
                "found_new_best": found_new_best,
                "loss": float("nan"),
                "accuracy": float("nan"),
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
            **train_metrics,
        }

    def optimize(self, **kwargs) -> dict[str, Any]:
        """
        Run the full Deep Cross-Entropy optimization.

        Args:
            progress_callback: Optional callback function that receives
                (iteration, metrics)
            goal_score: Optional goal score for early stopping

        Returns:
            Dictionary with optimization results
        """
        early_stopped = False
        iterations_completed = 0
        final_metrics = None

        # Update progress bar with total iterations if not already set
        if self.logger.progress_bar is not None and self.logger.progress_bar.total is None:
            self.logger.progress_bar.total = self.iterations
            self.logger.progress_bar.refresh()  # TODO This should not be necessary

        for iteration in range(self.iterations):
            metrics = self.run_iteration()
            final_metrics = metrics
            iterations_completed = iteration + 1

            # Log progress with experiment logger
            self.logger.log_metrics(metrics, iteration)

            # Log best construction when it improves
            if metrics["found_new_best"] and self.best_construction is not None:
                self.logger.log_construction(
                    self.best_construction,
                    self.best_score,
                    step=iteration,
                    metadata={"iteration": iteration},
                )

            # Early stopping if goal is reached
            if self.goal_score is not None and metrics["best_score"] < self.goal_score:
                self.logger.log_info(f"Goal achieved! Stopping early at iteration {iteration}")
                self.logger.log_info(
                    f"Best score: {metrics['best_score']:.6f} < goal: {self.goal_score:.6f}"
                )
                self.logger.log_info("Early Stopping")
                early_stopped = True
                break

            # Early stopping condition (optional)
            if metrics["num_elites"] == 0:
                self.logger.log_info(f"Warning: No elites selected at iteration {iteration}")
                break

        return {
            "best_score": self.best_score,
            "best_construction": self.best_construction,
            "final_metrics": final_metrics,
            "early_stopped": early_stopped,
            "iterations_completed": iterations_completed,
        }
