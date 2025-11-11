from collections.abc import Callable

import torch

from combo_dl import WagnerDeepCrossEntropy
from combo_dl.graph_utils import edge_vec_to_adj
from combo_dl.models.mlp import MLP


class LaplacianSpectralBound:
    def __init__(self, bound: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], n: int):
        self.bound = bound
        self.n = n
        self.k = -1
        self.lambda_param = -1
        self.mu = -1

    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early.

        Stop when we find a graph that violates the bound (reward > 0 means max_eigval > bound).

        Args:
            best_score: The current best score (reward = max_eigval - bound).

        Returns:
            Tuple of (should_stop, reason_message)
        """
        if best_score > 0.01:
            return (
                True,
                f"Bound violated: max_eigval > bound (reward = {best_score:.6f} > 0)",
            )
        return False, ""

    def reward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        adj = edge_vec_to_adj(edge_vec, self.n)  # (batch, row, col)
        degrees = adj.sum(dim=1)  # (batch, n)
        # Create batched diagonal degree matrix
        degree_mat = torch.diag_embed(degrees, dim1=-2, dim2=-1)  # (batch, n, n)
        laplacian = degree_mat - adj  # L = D - A

        if laplacian.device.type == "mps":
            laplacian_cpu = laplacian.cpu()
            eigvals = torch.linalg.eigvals(laplacian_cpu).real
            eigvals = eigvals.to(laplacian.device)
        else:
            eigvals = torch.linalg.eigvals(laplacian).real
        max_eigval = torch.max(eigvals, dim=1, keepdim=True)[0]

        neighbor_degree_sums = (adj @ degrees.unsqueeze(-1)).squeeze(-1)  # (batch_size, n)
        # Avoid division by zero on isolated vertices
        avg_neighbor_degrees = neighbor_degree_sums / (degrees + 1e-8)
        vertex_scores = self.bound(degrees, avg_neighbor_degrees)
        bound, _indices = torch.max(vertex_scores, dim=1, keepdim=True)

        return (max_eigval - bound).squeeze(-1)  # Return shape (batch_size,)


@torch.compile
def bound1(degrees: torch.Tensor, avg_degrees: torch.Tensor) -> torch.Tensor:
    """Computes the bound from Ghebleh et al.

    Args:
        degrees: Tensor of shape (batch_size, n) containing the degree of each vertex
        avg_degrees: Tensor of shape (batch_size, n) containing the average degree
            of the neighbors of each vertex.

    Returns:
        conjectured eigenvalue bound.
    """

    numerator = 4 * degrees**3
    return torch.sqrt(numerator / avg_degrees)


def bound4(degrees: torch.Tensor, avg_degrees: torch.Tensor) -> torch.Tensor:
    return (2 * degrees**2) / avg_degrees


def bound5(degrees: torch.Tensor, avg_degrees: torch.Tensor) -> torch.Tensor:
    numerator = 2 * degrees**2
    return (numerator / avg_degrees) + avg_degrees


def bound31(degrees: torch.Tensor, avg_degrees: torch.Tensor) -> torch.Tensor:
    numerator = 4 * avg_degrees**2
    # Avoid division by zero when both avg_degrees and degrees are zero (isolated vertices)
    return numerator / (avg_degrees + degrees + 1e-8)


class TestLaplacianSpectralBound:
    def test_initialization(self):
        lsb = LaplacianSpectralBound(bound1, n=4)
        assert lsb.bound == bound1
        assert lsb.n == 4

    def test_reward_shape(self):
        torch.manual_seed(42)
        lsb = LaplacianSpectralBound(bound1, n=4)
        edge_vec = torch.rand(3, 6)  # batch_size=3, edges=6 for n=4
        reward = lsb.reward(edge_vec)
        assert reward.shape == (3,)

    def test_reward_with_bound1(self):
        torch.manual_seed(42)
        lsb = LaplacianSpectralBound(bound1, n=3)
        # Complete graph on 3 vertices
        edge_vec = torch.ones(1, 3)  # n=3 has 3 edges
        reward = lsb.reward(edge_vec)
        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()

    def test_batch_processing(self):
        torch.manual_seed(42)
        lsb = LaplacianSpectralBound(bound1, n=4)
        edge_vec = torch.rand(5, 6)  # batch_size=5
        reward = lsb.reward(edge_vec)
        assert reward.shape == (5,)


def main():
    """Main function to run Laplacian spectral bound experiment with MLP DCE."""
    print("=" * 60)
    print("Laplacian Spectral Bound Experiment")
    print("=" * 60)

    n = 15
    bound = bound31
    iterations = 10000
    batch_size = 2048
    learning_rate = 1e-3
    elite_proportion = 0.1
    early_stopping_patience = iterations

    # Model hyperparameters
    hidden_layer_sizes = [64, 32, 16, 8, 4]
    output_size = 2
    dropout_probability = 0.1
    layernorm = True
    activation_function = "relu"

    # Device setup
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Graph: n={n}, bound={bound.__name__}")
    print(f"Training: iterations={iterations}, batch_size={batch_size}")
    print(f"Device: {device}")
    print("=" * 60)

    # Initialize problem
    problem = LaplacianSpectralBound(bound, n)

    # Initialize model
    model = MLP(
        n=problem.n,
        hidden_layer_sizes=hidden_layer_sizes,
        output_size=output_size,
        dropout_probability=dropout_probability,
        layernorm=layernorm,
        activation_function=activation_function,
    )

    # Initialize Deep Cross Entropy optimizer
    experiment_name = f"laplacian_spectral_bound_n{n}_bound1"

    dce = WagnerDeepCrossEntropy(
        model,
        problem,  # type: ignore[arg-type]
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        elite_proportion=elite_proportion,
        early_stopping_patience=early_stopping_patience,
        device=device,
        hydra_cfg=None,
        checkpoint_frequency=100,
        save_best_constructions=True,
        experiment_name=experiment_name,
    )

    # Create ReduceLROnPlateau scheduler using DCE's optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dce.optimizer,
        mode="max",
        factor=0.5,
        patience=300,
        min_lr=1e-6,
    )
    dce.scheduler = scheduler

    # Run optimization
    dce.optimize()


if __name__ == "__main__":
    main()
