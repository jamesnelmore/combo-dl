"""Typer commands for the Cayley DSRG search.

Thin wrappers — they only convert CLI-friendly types into the dataclasses
used by `orchestration` and pick a device. All real work lives one layer
down.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .orchestration import run_one_group, run_single
from .subset_enumeration import DSRGParams

app = typer.Typer(help="Cayley-graph DSRG exhaustive search.")


@app.command()
def single(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    output_dir: Path = Path("cayley_data"),
    batch_size: int = 100_000,
    device: str | None = None,
    noninteractive: bool = False,
) -> None:
    """Search every group of order n for DSRG(n, k, t, λ, μ)."""
    run_single(
        DSRGParams(n=n, k=k, t=t, lambda_=lambda_, mu=mu),
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        noninteractive=noninteractive,
    )


@app.command(name="one-group")
def one_group(
    n: int,
    k: int,
    t: int,
    lambda_: int,
    mu: int,
    group_id: int,
    output_dir: Path = Path("cayley_data"),
    batch_size: int = 100_000,
    device: str | None = None,
    noninteractive: bool = False,
) -> None:
    """Search one specific group (by SmallGroups library_id) for DSRG(n, k, t, λ, μ)."""
    run_one_group(
        DSRGParams(n=n, k=k, t=t, lambda_=lambda_, mu=mu),
        group_id=group_id,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        noninteractive=noninteractive,
    )


@app.command()
def cleanup() -> None:
    """Aggregate per-task results and dedup. (Not yet wired up.)"""
    raise NotImplementedError("Wire to aggregate.py / dedup.py")


if __name__ == "__main__":
    app()
