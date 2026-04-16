#!/usr/bin/env python3
"""Smoke-test the MLP on the 10 smallest known SRG parameter sets.

For each case, trains an MLP via Deep Cross-Entropy with a 2-minute
wall-time limit. A perfect SRG has reward 0 (the negative squared
Frobenius norm of the SRG defining equation's residual); any score >= 0
means a graph matching the parameters was found.

Usage:
    uv run python3 scripts/smoke_test_mlp_srg.py
"""

import sys
import time

import torch

from combo_dl import StronglyRegularGraphs, WagnerDeepCrossEntropy
from combo_dl.models import MLP

# 10 smallest known (n,k,lambda,mu) SRG parameter sets.
CASES: list[tuple[int, int, int, int]] = [
    (5, 2, 0, 1),
    (9, 4, 1, 2),
    (10, 3, 0, 1),
    (13, 6, 2, 3),
    (15, 6, 1, 3),
    (16, 5, 0, 2),
    (16, 6, 2, 2),
    (17, 8, 3, 4),
    (21, 10, 3, 6),
    (21, 10, 4, 5),
]

TIME_LIMIT_SECONDS = 120
SEED = 42


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_case(n: int, k: int, lam: int, mu: int, device: str) -> dict:
    problem = StronglyRegularGraphs(n, k, lam, mu)
    width = int(min(n**2 / 2, 256))
    model = MLP(
        n=n,
        hidden_layer_sizes=[width, width, width, 64],
        output_size=2,
        dropout_probability=0.1,
        layernorm=True,
        activation_function="gelu",
    )
    dce = WagnerDeepCrossEntropy(
        model=model,
        problem=problem,
        iterations=10**5,
        batch_size=2048,
        learning_rate=1e-3,
        elite_proportion=0.1,
        device=device,
        early_stopping_patience=10**6,
        use_wandb=False,
        save_best_constructions=False,
        checkpoint_frequency=10**9,
        save_dir="runs/mlp_srg_smoke",
        experiment_name=f"srg_{n}_{k}_{lam}_{mu}",
        torch_compile=False,
        max_wall_seconds=TIME_LIMIT_SECONDS,
    )
    return dce.optimize()


def main() -> int:
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}, seed: {SEED}, per-case budget: {TIME_LIMIT_SECONDS}s\n")

    results = []
    for n, k, lam, mu in CASES:
        header = f"SRG({n},{k},{lam},{mu})"
        print("=" * 60)
        print(header)
        print("=" * 60)
        t0 = time.monotonic()
        out = run_case(n, k, lam, mu, device)
        elapsed = time.monotonic() - t0
        results.append({
            "params": (n, k, lam, mu),
            "best_score": out["best_score"],
            "wall_seconds": out.get("wall_seconds", elapsed),
            "iterations": out.get("iterations"),
            "found": out["best_score"] >= 0.0,
        })

    print("\nSUMMARY")
    print(f"{'params':<20}{'score':>14}{'time':>10}{'iters':>10}  status")
    for r in results:
        n, k, lam, mu = r["params"]
        params = f"({n},{k},{lam},{mu})"
        status = "FOUND" if r["found"] else "none"
        iters = r["iterations"] if r["iterations"] is not None else "?"
        print(
            f"{params:<20}{r['best_score']:>14.4f}{r['wall_seconds']:>9.1f}s{iters:>10}  {status}"
        )

    n_found = sum(1 for r in results if r["found"])
    print(f"\n{n_found}/{len(results)} SRGs found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
