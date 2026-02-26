import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from combo_dl import WagnerDeepCrossEntropy


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

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)


@hydra.main(config_path="../configs", config_name="mlp_dce", version_base=None)
def main(cfg: DictConfig) -> None:
    # Set random seed for reproducibility
    seed: int = cfg.get("seed", 42)
    deterministic_cudnn: bool = cfg.get("deterministic_cudnn", True)
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)
    
    print("Configuration loaded:")
    print(cfg)
    print(f"Random seed: {seed}")
    print(f"Deterministic cuDNN: {deterministic_cudnn}")

    problem = instantiate(cfg.graph)
    model = instantiate(cfg.model, n=problem.n)

    experiment_name: str | None = cfg.get("experiment_name", None)
    torch_compile: bool = bool(cfg.get("torch_compile", False))

    dce = WagnerDeepCrossEntropy(
        model,
        problem,
        cfg.training.iterations,
        cfg.training.batch_size,
        cfg.training.learning_rate,
        cfg.training.elite_proportion,
        early_stopping_patience=cfg.training.early_stopping_patience,
        device=device,
        hydra_cfg=cfg,
        checkpoint_frequency=100,
        save_best_constructions=True,
        survivor_proportion=cfg.training.survivor_proportion,
        experiment_name=experiment_name,
        torch_compile=torch_compile,
    )

    # Instantiate scheduler if configured, using DCE's optimizer
    if hasattr(cfg.training, "scheduler") and cfg.training.scheduler is not None:
        OmegaConf.set_struct(cfg.training.scheduler, False)
        dce.scheduler = instantiate(cfg.training.scheduler, optimizer=dce.optimizer)
    dce.optimize()


if __name__ == "__main__":
    main()
