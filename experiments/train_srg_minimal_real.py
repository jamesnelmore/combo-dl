from pathlib import Path

import torch

from combo_dl.algorithms import WagnerDeepCrossEntropy
from combo_dl.models import FFModel
from combo_dl.problems import StronglyRegularGraphs

N = 17
K = 8
LAMBDA = 3
MU = 4

ITERATIONS = 10_000
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
ELITE_PROPORTION = 0.1
EARLY_STOPPING_PATIENCE = ITERATIONS

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    problem = StronglyRegularGraphs(N, K, LAMBDA, MU)
    model = FFModel(N, hidden_layer_sizes=[64, 32, 16, 8, 4])
    dce = WagnerDeepCrossEntropy(
        model,
        problem,
        ITERATIONS,
        BATCH_SIZE,
        LEARNING_RATE,
        ELITE_PROPORTION,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        wandb_mode="online",
        checkpoint_dir=Path("checkpoints"),
        device=device,
    )
    dce.optimize()


if __name__ == "__main__":
    main()
