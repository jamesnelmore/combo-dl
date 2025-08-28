"""
Main entry point for thesis experiments using Hydra configuration.

Usage:
    python -m thesis.main                    # Default config
    python -m thesis.main model.n=25         # Override model size
    python -m thesis.main algorithm=ppo      # Use different algorithm
    python -m thesis.main --multirun seed=1,2,3  # Run multiple seeds
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def get_device(device_config: str) -> str:
    """Get the appropriate device based on config and availability."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_config

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> dict:
    """
    Main training function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary with experiment results
    """
    
    # Setup logging
    log.info("Starting experiment with configuration:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        log.info(f"Set random seed to {cfg.seed}")
    
    # Device management
    device = get_device(getattr(cfg, 'device', 'auto'))
    log.info(f"Using device: {device}")
    
    # Instantiate problem
    log.info("Instantiating problem...")
    problem = hydra.utils.instantiate(cfg.problem)
    log.info(f"Created problem: {type(problem).__name__}")
    
    # Instantiate model
    log.info("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    log.info(f"Created model: {type(model).__name__} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Instantiate data (optional)
    data = None
    if hasattr(cfg, 'data') and cfg.data is not None:
        log.info("Instantiating data...")
        data = hydra.utils.instantiate(cfg.data)
        log.info(f"Created data: {type(data).__name__}")
    
    # Instantiate algorithm
    log.info("Instantiating algorithm...")
    algorithm = hydra.utils.instantiate(
        cfg.algorithm, 
        model=model, 
        problem=problem,
        device=device
    )
    log.info(f"Created algorithm: {type(algorithm).__name__}")
    
    # Run optimization
    log.info("Starting optimization...")
    try:
        # Create progress callback for logging
        def progress_callback(iteration: int, metrics: dict):
            if iteration % 100 == 0:  # Log every 100 iterations
                log.info(f"Iteration {iteration}: best_score={metrics.get('best_score', 'N/A'):.6f}")
        
        # Check if algorithm accepts progress callback
        if hasattr(algorithm, 'optimize'):
            if 'progress_callback' in algorithm.optimize.__code__.co_varnames:
                results = algorithm.optimize(progress_callback=progress_callback)
            else:
                results = algorithm.optimize()
        else:
            raise AttributeError(f"Algorithm {type(algorithm).__name__} must have an 'optimize' method")
            
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        raise
    
    # Log final results
    log.info("=== FINAL RESULTS ===")
    log.info(f"Best score achieved: {results.get('best_score', 'N/A')}")
    
    if hasattr(problem, 'goal_score'):
        goal_score = problem.goal_score
        best_score = results.get('best_score', float('inf'))
        success = best_score < goal_score if best_score != float('inf') else False
        
        log.info(f"Goal score: {goal_score:.6f}")
        log.info(f"Success: {'YES' if success else 'NO'}")
    
    if 'early_stopped' in results:
        log.info(f"Early stopped: {'YES' if results['early_stopped'] else 'NO'}")
    
    if 'iterations_completed' in results:
        log.info(f"Iterations completed: {results['iterations_completed']}")
    
    # Save results to working directory (exclude tensors for YAML compatibility)
    serializable_results = {}
    for key, value in results.items():
        if hasattr(value, 'cpu'):  # PyTorch tensor
            if key == 'best_construction':
                # Save tensor info but not the actual tensor
                serializable_results[key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'sum': float(value.sum().item()),
                    'note': 'Tensor data not saved in YAML (too large)'
                }
            else:
                serializable_results[key] = float(value.item()) if value.numel() == 1 else 'tensor_data'
        else:
            serializable_results[key] = value
    
    results_with_config = {
        'results': serializable_results,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'device_used': device
    }
    
    # Save results as YAML
    results_path = Path("results.yaml")
    OmegaConf.save(results_with_config, results_path)
    log.info(f"Results saved to {results_path.absolute()}")
    
    # Save the actual best construction tensor separately if it exists
    if 'best_construction' in results and results['best_construction'] is not None:
        import torch
        tensor_path = Path("best_construction.pt")
        torch.save(results['best_construction'], tensor_path)
        log.info(f"Best construction tensor saved to {tensor_path.absolute()}")
    
    return results