"""
Main entry point for thesis experiments using Hydra configuration.

Usage:
    python -m thesis.main                    # Default config
    python -m thesis.main model.n=25         # Override model size
    python -m thesis.main algorithm=ppo      # Use different algorithm
    python -m thesis.main --multirun seed=1,2,3  # Run multiple seeds
"""

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from algorithms import BaseAlgorithm
from experiment_logger.logger import BaseExperimentLogger

log = logging.getLogger(__name__)


def get_device(device_config: str) -> str:
    """Get the appropriate device based on config and availability."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_config


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> dict:
    """
    Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary with experiment results
    """

    # Set up device
    device = get_device(cfg.device)
    log.info(f"Using device: {device}")

    # Instantiate components with proper device handling
    model: nn.Module = instantiate(cfg.model)
    model = model.to(device)

    problem = instantiate(cfg.problem)

    logger: BaseExperimentLogger = instantiate(cfg.logger)

    # Convert config to container for logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_experiment_start(config_dict)

    logger.log_info(f"Working directory: {HydraConfig.get().runtime.output_dir}")
    logger.log_info(f"Using device: {device}")
    logger.log_info("Starting main experiment...")

    # Instantiate algorithm with all required parameters
    algorithm: BaseAlgorithm = instantiate(
        cfg.algorithm, model=model, problem=problem, logger=logger, device=device
    )

    results = algorithm.optimize()

    # Log experiment completion
    logger.log_model(model, model_name="model")
    logger.log_experiment_end(results, success=True)

    return results


if __name__ == "__main__":
    main()
