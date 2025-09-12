"""Simple main entry point using minimal Hydra configuration.

Usage:
    python main.py                              # Use default config
    python main.py model.n=25                   # Override model size
    python main.py algorithm.learning_rate=0.01 # Override learning rate
    python main.py seed=123                     # Change random seed
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

# TODO add slurm defaults for each HPC job submission


def get_device(device_config: str) -> str:
    """Get the appropriate device based on config and availability.

    Args:
        device_config: Device configuration string ("auto", "cpu", "cuda", "mps")

    Returns:
        The appropriate device string based on availability
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_config


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> dict:
    """Entrypoint for Hydra.

    This function:
    1. Sets up the device (CPU/GPU/MPS)
    2. Creates model, problem, algorithm, and logger from config
    3. Runs the optimization
    4. Returns results

    Args:
        cfg: Hydra configuration dictionary

    Returns:
        Dictionary containing optimization results
    """
    # Logger
    logger = instantiate(cfg.logger)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_experiment_start(config_dict)

    # Seed
    torch.manual_seed(cfg.seed)
    logger.log_info(f"Set random seed: {cfg.seed}")

    # Device
    device = get_device(cfg.device)
    logger.log_info(f"Set device: {device}")

    # Model
    model = instantiate(cfg.model).to(device)

    # Problem
    problem = instantiate(cfg.problem)

    # Algorithm
    algorithm = instantiate(
        cfg.algorithm, model=model, problem=problem, logger=logger, device=device
    )

    logger.log_info(f"Experiment: {cfg.experiment_name}")

    # Run the optimization
    results = algorithm.optimize()

    # Log final model and results
    logger.log_model(model, model_name="trained_model")
    logger.log_experiment_end(results, success=True)

    logger.log_info("Experiment completed!")
    logger.log_info(f"Final results: {results}")

    return results


if __name__ == "__main__":
    main()
