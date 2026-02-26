from collections.abc import Callable

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from combo_dl import WagnerDeepCrossEntropy
from combo_dl.graph_utils import edge_vec_to_adj


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic_cudnn: If True, enables deterministic cuDNN operations.
            This ensures full reproducibility but may cause ~10% performance slowdown.
            For MLPs (linear layers), the impact is typically minimal.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set deterministic mode for CUDA operations (may impact performance)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LaplacianSpectralBound:
    def __init__(self, bound: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], n: int):
        self.bound = bound
        self.n = n
        self.k = -1
        self.lambda_param = -1
        self.mu = -1

    def should_stop_early(self, best_score: float | None = None, best_construction: torch.Tensor) -> tuple[bool, str]:
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

        # Check connectivity: graph is connected iff algebraic connectivity (second smallest eigenval) > 0
        sorted_eigvals = torch.sort(eigvals, dim=1)[0]
        algebraic_connectivity = sorted_eigvals[:, 1]  # Second smallest eigenvalue
        is_connected = algebraic_connectivity > 1e-6  # (batch_size,)

        max_eigval = torch.max(eigvals, dim=1, keepdim=True)[0]

        neighbor_degree_sums = (adj @ degrees.unsqueeze(-1)).squeeze(-1)  # (batch_size, n)
        isolated_vertex_penalty = 100 * torch.sum(degrees == 0, dim=1)
        # Avoid division by zero on isolated vertices
        avg_neighbor_degrees = neighbor_degree_sums / (degrees + 1e-8)
        vertex_scores = self.bound(degrees, avg_neighbor_degrees)
        bound, _indices = torch.max(vertex_scores, dim=1, keepdim=True)

        reward = (max_eigval - bound).squeeze(-1)  # (batch_size,)
        # Heavily penalize disconnected graphs
        reward = torch.where(is_connected, reward, torch.full_like(reward, -1000.0))
        return reward


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


def bound3(
    degrees: torch.Tensor, avg_degrees: torch.Tensor
) -> torch.Tensor:  # Broken in Ghebleh with n=21
    return (avg_degrees**2 / (degrees + 1e-8)) + avg_degrees


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


BOUND_FUNCTIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "bound1": bound1,
    "bound4": bound4,
    "bound5": bound5,
    "bound31": bound31,
    "bound3": bound3,
}

class LaplacianDCE(WagnerDeepCrossEntropy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def should_stop_early(self) -> tuple[bool, str]:
        return self.problem.

@hydra.main(config_path="../configs", config_name="laplacian_spectral_bounds", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run Laplacian spectral bound experiment with MLP DCE."""
    # Set random seed for reproducibility
    seed: int = cfg.get("seed", 42)
    deterministic_cudnn: bool = cfg.get("deterministic_cudnn", True)
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)

    print("=" * 60)
    print("Laplacian Spectral Bound Experiment")
    print("=" * 60)
    print(f"Random seed: {seed}")
    print(f"Deterministic cuDNN: {deterministic_cudnn}")

    bound_key: str = cfg.problem.bound
    if bound_key not in BOUND_FUNCTIONS:
        raise ValueError(
            f"Unknown bound function '{bound_key}'. Available: {list(BOUND_FUNCTIONS)}"
        )
    bound_fn = BOUND_FUNCTIONS[bound_key]

    n: int = cfg.problem.n

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Graph: n={n}, bound={bound_key}")
    print(
        f"Training: iterations={cfg.training.iterations}, "
        f"batch_size={cfg.training.batch_size}, "
        f"learning_rate={cfg.training.learning_rate}"
    )
    print(f"Device: {device}")
    print("=" * 60)

    problem = LaplacianSpectralBound(bound_fn, n)
    model = instantiate(cfg.model, n=problem.n)

    experiment_name: str | None = cfg.get("experiment_name", None)
    torch_compile: bool = bool(cfg.get("torch_compile", False))



    dce = WagnerDeepCrossEntropy(
        model,
        problem,  # type: ignore[arg-type]
        cfg.training.iterations,
        cfg.training.batch_size,
        cfg.training.learning_rate,
        cfg.training.elite_proportion,
        early_stopping_patience=cfg.training.early_stopping_patience,
        device=device,
        hydra_cfg=cfg,
        checkpoint_frequency=cfg.training.checkpoint_frequency,
        save_best_constructions=cfg.training.save_best_constructions,
        experiment_name=experiment_name,
        survivor_proportion=cfg.training.survivor_proportion,
        torch_compile=torch_compile,
    )

    if hasattr(cfg.training, "scheduler") and cfg.training.scheduler is not None:
        OmegaConf.set_struct(cfg.training.scheduler, False)
        dce.scheduler = instantiate(cfg.training.scheduler, optimizer=dce.optimizer)


    dce.optimize()


if __name__ == "__main__":
    main()
