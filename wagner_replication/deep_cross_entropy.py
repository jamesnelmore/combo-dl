import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Callable

from wagner_model import WagnerModel, generate_sampled_constructions


class WagnerDeepCrossEntropy:
    def __init__(
        self, 
        model: WagnerModel,
        iterations: int = 10_000,
        batch_size: int = 512,
        learning_rate: float = 0.0001,
        elite_proportion: float = 0.1,
        device: str | None = None
    ):
        self.model = model
        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.elite_proportion = elite_proportion
        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        )
        
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
        
        # Track optimization state
        self.best_score = float('-inf')
        self.best_construction = None
        self.history = []
        
        assert hasattr(model, "score") and callable(getattr(model, "score")), "model.score must be a function"

    def select_elites(
        self,
        constructions: torch.Tensor,
        batch_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select top elite constructions based on scores."""
        batch_size = len(batch_scores)
        return_count = int(batch_size * self.elite_proportion)
        elite_indices = torch.argsort(batch_scores, descending=True)[:return_count]
        return constructions[elite_indices]

    def extract_examples(
        self, 
        elite_constructions: torch.Tensor, 
        output_batch_size: int | None = None
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
        pos_tensor = (
            torch.arange(num_edges).repeat(num_constructions).to(target_device)
        )
        edge_indices = torch.arange(num_edges).to(target_device)
        mask = (edge_indices.unsqueeze(0) < pos_tensor.unsqueeze(1)).to(target_device)

        obs_tensor = elite_constructions.repeat_interleave(num_edges, dim=0) * mask
        obs_tensor = obs_tensor.to(target_device)
        construction_indices = (
            torch.arange(num_constructions).repeat_interleave(num_edges).to(target_device) # Defined elementwise: T[i] = elite_constructions[i][pos_tensor[i]]
        ) 
        actions_tensor = elite_constructions[construction_indices, pos_tensor].to(
            target_device
        )

        dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

        return dataloader

    def train_step(
        self,
        train_loader: DataLoader,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> dict[str, float]:
        """Perform one training step and return metrics."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (obs, pos, target_actions) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self.model(obs, pos)
            loss = criterion(outputs, target_actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target_actions.size(0)
            train_correct += (predicted == target_actions).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
            
            if progress_callback is not None:
                progress_callback(avg_loss, accuracy)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": train_total
        }

    def run_iteration(self) -> dict[str, float]:
        """Run one DCE iteration and return metrics."""
        # Generate constructions and score them
        constructions = generate_sampled_constructions(self.model, batch_size=self.batch_size)
        batch_scores = self.model.score(constructions)
        
        # Track best score and construction
        current_best = torch.max(batch_scores).item()
        avg_score = torch.mean(batch_scores).item()
        
        if current_best > self.best_score:
            self.best_score = current_best
            best_idx = torch.argmax(batch_scores)
            self.best_construction = constructions[best_idx].clone()
        
        # Select elites
        elites = self.select_elites(constructions, batch_scores)
        
        if len(elites) == 0:
            return {
                "best_score": self.best_score,
                "avg_score": avg_score,
                "num_elites": 0,
                "loss": float('nan'),
                "accuracy": float('nan')
            }
        
        # Extract training examples and train
        dataloader = self.extract_examples(
            elites, 
            output_batch_size=min(self.batch_size, len(elites) * self.model.edges)
        )
        train_metrics = self.train_step(dataloader)
        
        return {
            "best_score": self.best_score,
            "avg_score": avg_score,
            "num_elites": len(elites),
            **train_metrics
        }

    def optimize(self, progress_callback: Callable[[int, dict], None] | None = None) -> dict:
        """
        Run the full Deep Cross-Entropy optimization.
        
        Args:
            progress_callback: Optional callback function that receives (iteration, metrics)
        
        Returns:
            Dictionary with optimization results and history
        """
        progress_bar = tqdm(range(self.iterations), desc="DCE Iterations")
        
        for iteration in progress_bar:
            metrics = self.run_iteration()
            self.history.append(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}" if not torch.isnan(torch.tensor(metrics['loss'])) else "N/A",
                "acc": f"{metrics['accuracy']:.2f}%" if not torch.isnan(torch.tensor(metrics['accuracy'])) else "N/A",
                "best": f"{metrics['best_score']:.4f}",
                "avg": f"{metrics['avg_score']:.4f}",
                "elites": metrics['num_elites']
            })
            
            # Call external progress callback if provided
            if progress_callback is not None:
                progress_callback(iteration, metrics)
            
            # Early stopping condition (optional)
            if metrics['num_elites'] == 0:
                print(f"Warning: No elites selected at iteration {iteration}")
        
        return {
            "best_score": self.best_score,
            "best_construction": self.best_construction,
            "history": self.history,
            "final_metrics": self.history[-1] if self.history else None
        }

    def get_best_construction(self) -> torch.Tensor | None:
        """Return the best construction found so far."""
        return self.best_construction

    def get_optimization_history(self) -> list[dict[str, float]]:
        """Return the full optimization history."""
        return self.history


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = WagnerModel(4)
    
    # Create optimizer instance
    optimizer = WagnerDeepCrossEntropy(
        model=model,
        batch_size=4096,
        iterations=10_000,
        learning_rate=0.0001,
        elite_proportion=0.1,
        device=device
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    print(f"Best score achieved: {results['best_score']:.4f}")
    print(f"Best construction: {results['best_construction']}")


if __name__ == "__main__":
    main()
