"""Deep Cross Entropy Algorithm."""

from datetime import datetime
from typing import Any, override

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..experiment_logger import ExperimentLogger
from ..models.protocols import SamplingModel
from ..problems.base_problem import BaseProblem
from .base_algorithm import BaseAlgorithm


class WagnerDeepCrossEntropy(BaseAlgorithm):
    """Deep Cross Entropy Algorithm.

    Implementation of Deep Cross Entropy from [Wagner 2021](http://arxiv.org/abs/2104.14516).
    """

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
        early_stopping_patience: int = 300,
    ):
        if logger is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_name = f"Deep Cross Entropy {date} {type(problem).__name__}"
            logger: ExperimentLogger = ExperimentLogger(experiment_name, use_wandb=False)

        super().__init__(model, problem, logger)

        self.iterations = iterations
        postfix_metrics = ["best_score", "avg_score", "loss", "accuracy"]
        self.logger.configure_progress_bar(postfix_metrics, total_iterations=self.iterations)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.log_frequency = log_frequency
        self.model_save_frequency = model_save_frequency
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Optimization State (we want to maximize)
        self.best_score = float("-inf")
        self.best_construction = None
        self.samples_seen = 0
        
        # Early stopping tracking
        self.steps_since_best = 0
        self.best_score_iteration = 0

    @override
    def optimize(self, **kwargs) -> dict[str, Any]:
        """Run the full Deep Cross-Entropy optimization.

        Args:
            None.

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
            num_elites = metrics["num_elites"]
            del metrics["num_elites"]
            
            # Update early stopping tracking
            if metrics["found_new_best"]:
                self.steps_since_best = 0
                self.best_score_iteration = iteration
            else:
                self.steps_since_best += 1
            
            # Always update progress bar, but only log detailed metrics at log_frequency
            if iteration % self.log_frequency == 0:
                self.logger.log_metrics(metrics, iteration)
            else:
                # Still update progress bar with minimal info
                minimal_metrics = {
                    "best_score": metrics["best_score"],
                    "samples_seen": metrics["samples_seen"],
                }
                self.logger.log_metrics(minimal_metrics, iteration)

            # Log best construction when it improves
            if metrics["found_new_best"] and self.best_construction is not None:
                self.logger.log_construction(
                    self.best_construction,
                    self.best_score,
                    step=iteration,
                    metadata={"iteration": iteration},
                )

            if iteration > 0 and iteration % self.model_save_frequency == 0:
                self.logger.log_model(
                    self.model, model_name=f"{self.logger.experiment_name}_step_{iteration}"
                )

            # Check if problem wants to stop early
            should_stop, stop_reason = self.problem.should_stop_early(metrics["best_score"])
            if should_stop:
                self.logger.log_info(f"Early stopping at iteration {iteration}")
                self.logger.log_info(stop_reason)
                early_stopped = True
                break

            # Early stopping condition: no improvement for patience steps
            if self.steps_since_best >= self.early_stopping_patience:
                self.logger.log_info(f"Early stopping at iteration {iteration}: no improvement for {self.early_stopping_patience} steps")
                self.logger.log_info(f"Best score {self.best_score} was achieved at iteration {self.best_score_iteration}")
                early_stopped = True
                break

            # Early stopping condition (optional)
            if num_elites == 0:
                self.logger.log_info(f"Warning: No elites selected at iteration {iteration}")
                break

        return {
            "best_score": self.best_score,
            "best_construction": self.best_construction,
            "num_elites": num_elites,
            "final_metrics": final_metrics,
            "early_stopped": early_stopped,
            "iterations": iteration,
            "samples_seen": self.samples_seen,
            "best_score_iteration": self.best_score_iteration,
            "steps_since_best": self.steps_since_best,
        }

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics."""
        constructions = self.model.sample(self.batch_size)
        batch_scores = self.problem.reward(constructions)

        # Update samples seen counter
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
            self.logger.log_info("Zero elites passed. This should not occur. Stopping run.")
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
        """Select top elite constructions based on scores (highest for maximization)."""
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
        actions_tensor = constructions[construction_indices, pos_tensor].to(target_device, dtype=torch.long)

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
