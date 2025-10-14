import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from combo_dl import WagnerDeepCrossEntropy

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)


@hydra.main(config_path="../configs", config_name="mlp_dce", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration loaded:")
    print(cfg)

    problem = instantiate(cfg.graph)
    model = instantiate(cfg.model, n=problem.n)

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
    )
    dce.optimize()


if __name__ == "__main__":
    main()
