from datetime import datetime
from typing import Any, override

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_logger import ExperimentLogger
from models.protocols import SamplingModel
from problems.base_problem import BaseProblem

from .base_algorithm import BaseAlgorithm


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
        device: str = "cpu",
        logger: ExperimentLogger | None = None,  # pyright: ignore[reportRedeclaration]
        log_frequency: int = 1,
        model_save_frequency: int = 1000,
    ):
        if logger is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_name = f"Deep Cross Entropy {date} {type(problem).__name__}"
            logger: ExperimentLogger = ExperimentLogger(experiment_name, use_wandb=False)

        super().__init__(model, problem, logger)

        postfix_metrics = ["best_score", "avg_score", "loss", "accuracy"]
        self.logger.configure_progress_bar(postfix_metrics, total_iterations=self.iterations)

        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.log_frequency = log_frequency
        self.model_save_frequency = model_save_frequency
        self.device = device
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Optimization State (we want to maximize)
        self.best_score = float("-inf")
        self.best_construction = None

    @override
    def optimize(self, **kwargs) -> dict[str, Any]:
        """Run the full Deep Cross-Entropy optimization.
        Args:
            None

        Kwargs:
            Not used, only for API compatability
        Returns
        -------
            Dictionary of optimization results
        """
        if kwargs != {}:
            self.logger.log_info(
                f"Deep Cross Entropy passed keyword arguments but does not use any.\n{kwargs}"
            )

        early_stopped = False
        final_metrics = None

        for iteration in range(self.iterations):
            metrics = self.run_iteration()
            final_metrics = metrics

            # Always update progress bar, but only log detailed metrics at log_frequency
            if iteration % self.log_frequency == 0:
                self.logger.log_metrics(metrics, iteration)
            else:
                # Still update progress bar with minimal info
                minimal_metrics = {"best_score": metrics["best_score"]}
                self.logger.log_metrics(minimal_metrics, iteration)

            # Log best construction when it improves
            if metrics["found_new_best"] and self.best_construction is not None:
                self.logger.log_construction(
                    self.best_construction,
                    self.best_score,
                    step=iteration,
                    metadata={"iteration": iteration},
                )

            if iteration % self.model_save_frequency:
                self.logger.log_model(
                    self.model, model_name=f"{self.logger.experiment_name} step {iteration}"
                )

            # Check if problem wants to stop early
            should_stop, stop_reason = self.problem.should_stop_early(metrics["best_score"])
            if should_stop:
                self.logger.log_info(f"Early stopping at iteration {iteration}")
                self.logger.log_info(stop_reason)
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
            "iterations": iteration,
        }

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics."""
        constructions = self.model.sample(self.batch_size)
        batch_scores = self.problem.reward(constructions)

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

    def select_elites(
        self,
        constructions: torch.Tensor,
        batch_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select top elite constructions based on scores (highest for maximization)."""
        batch_size = len(batch_scores)
        return_count = int(batch_size * self.elite_proportion)
        # descending for maximization
        elite_indices = torch.argsort(batch_scores, descending=True)[:return_count]
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
