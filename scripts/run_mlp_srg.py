#!/usr/bin/env python3
"""Run Deep Cross-Entropy on a single SRG parameter set.

Usage:
    uv run python3 scripts/run_mlp_srg.py <n> <k> <lambda> <mu> [options]

Example (run for 1 hour on (13,6,2,3)):
    uv run python3 scripts/run_mlp_srg.py 13 6 2 3 --seconds 3600
"""

import argparse
import sys

import torch

from combo_dl import StronglyRegularGraphs, WagnerDeepCrossEntropy
from combo_dl.models import MLP


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("k", type=int)
    p.add_argument("lam", type=int)
    p.add_argument("mu", type=int)
    p.add_argument("--seconds", type=int, default=3600, help="wall-time budget")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--elite", type=float, default=0.1)
    p.add_argument("--survivor", type=float, default=0.02)
    p.add_argument("--hidden", type=str, default="256,128,64",
                   help="comma-separated hidden layer sizes")
    p.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="enable torch.compile (broken on MPS in current torch)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler-patience", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="runs/mlp_single")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    hidden = [int(x) for x in args.hidden.split(",")]
    print(f"device={device} seed={args.seed} budget={args.seconds}s "
          f"batch={args.batch_size} lr={args.lr} hidden={hidden}")

    problem = StronglyRegularGraphs(args.n, args.k, args.lam, args.mu)
    model = MLP(
        n=args.n,
        hidden_layer_sizes=hidden,
        output_size=2,
        dropout_probability=args.dropout,
        layernorm=not args.no_layernorm,
        activation_function=args.activation,
    )
    dce = WagnerDeepCrossEntropy(
        model=model,
        problem=problem,
        iterations=10**9,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        elite_proportion=args.elite,
        survivor_proportion=args.survivor,
        device=device,
        early_stopping_patience=10**9,
        use_wandb=args.wandb,
        save_best_constructions=True,
        save_dir=args.save_dir,
        experiment_name=f"srg_{args.n}_{args.k}_{args.lam}_{args.mu}",
        torch_compile=args.compile,
        max_wall_seconds=args.seconds,
    )
    dce.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dce.optimizer, mode="max", factor=0.5,
        patience=args.scheduler_patience, min_lr=1e-5,
    )
    dce.optimize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
