"""Visualisation utilities for ILP sweep results.

The primary chart is **wall time vs n** (number of vertices) with one
series per ``(model, config)`` combination.  Instances that hit the
time limit are drawn as triangles pinned to the timeout line so they
are visually distinct from completed solves.

Usage from the CLI::

    python -m ilp plot results.json --timeout 300

Usage from Python::

    from ilp.bench.viz import plot_walltime_vs_n
    plot_walltime_vs_n("sweep_results.json", timeout=300)

All plotting is done with matplotlib so there are no extra dependencies
beyond what the project already has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load a sweep results JSON file."""
    return json.loads(Path(path).read_text())


def _series_label(row: dict[str, Any]) -> str:
    """Derive a human-readable series label from model name + config fields.

    Config fields are stored with a ``cfg_`` prefix in the result dict.
    We strip the prefix and format them as ``key=value`` pairs.
    """
    model = row.get("model", "unknown")
    cfg_parts = sorted(
        (k.removeprefix("cfg_"), v)
        for k, v in row.items()
        if k.startswith("cfg_")
    )
    if cfg_parts:
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg_parts)
        return f"{model} [{cfg_str}]"
    return model


def _group_by_series(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group result rows by their series label."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        label = _series_label(row)
        groups.setdefault(label, []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

# A selection of colours that are distinguishable in both light and dark
# themes and under common colour-vision deficiencies.
_PALETTE = [
    "#4C72B0",  # steel blue
    "#DD8452",  # sandy brown
    "#55A868",  # medium green
    "#C44E52",  # indian red
    "#8172B3",  # muted purple
    "#937860",  # tan
    "#DA8BC3",  # orchid pink
    "#8C8C8C",  # grey
    "#CCB974",  # dark khaki
    "#64B5CD",  # sky blue
]


def plot_walltime_vs_n(
    *result_paths: str | Path,
    timeout: float | None = None,
    title: str | None = None,
    output: str | Path | None = None,
    log_y: bool = False,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """Plot wall time vs n for one or more sweep result files.

    Multiple result files are overlaid so you can visually compare
    different formulations / configurations.

    Args:
        result_paths: One or more paths to sweep result JSON files.
        timeout: If provided, draws a horizontal line at this value and
            renders ``TimeLimit`` results as upward-pointing triangles
            pinned to the timeout line.
        title: Custom plot title. Defaults to
            ``"ILP solve time vs graph size"``.
        output: If provided, save the figure to this path instead of
            calling ``plt.show()``.
        log_y: Use a log scale for the y-axis.
        figsize: Figure size in inches ``(width, height)``.
    """
    # ── Collect results from all files ────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    for p in result_paths:
        all_results.extend(_load_results(p))

    if not all_results:
        print("No results to plot.")
        return

    groups = _group_by_series(all_results)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (label, rows) in enumerate(sorted(groups.items())):
        colour = _PALETTE[idx % len(_PALETTE)]

        # Split into completed vs timed-out
        completed = [r for r in rows if r["status"] != "TimeLimit"]
        timed_out = [r for r in rows if r["status"] == "TimeLimit"]

        # Completed: circles
        if completed:
            ns = [r["n"] for r in completed]
            ts = [r["wall_seconds"] for r in completed]
            ax.scatter(
                ns,
                ts,
                color=colour,
                marker="o",
                s=60,
                label=label,
                zorder=3,
                alpha=0.85,
            )

        # Timed-out: triangles pinned to the timeout line (or their own time)
        if timed_out:
            ns_to = [r["n"] for r in timed_out]
            ts_to = [
                timeout if timeout is not None else r["wall_seconds"]
                for r in timed_out
            ]
            ax.scatter(
                ns_to,
                ts_to,
                color=colour,
                marker="^",
                s=80,
                edgecolors="black",
                linewidths=0.6,
                label=f"{label} (timeout)",
                zorder=4,
                alpha=0.85,
            )

    # ── Timeout line ──────────────────────────────────────────────────────
    if timeout is not None:
        ax.axhline(
            timeout,
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"timeout ({timeout}s)",
        )

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel("n  (number of vertices)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title(title or "ILP solve time vs graph size")

    if log_y:
        ax.set_yscale("log")

    # Integer ticks on the x-axis (n is always integral).
    all_ns = sorted({r["n"] for r in all_results})
    if len(all_ns) <= 30:
        ax.set_xticks(all_ns)
    else:
        # Too many distinct n values — let matplotlib auto-tick.
        pass

    ax.legend(fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output is not None:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: summary table
# ---------------------------------------------------------------------------


def print_summary(
    *result_paths: str | Path,
) -> None:
    """Print a compact summary table of sweep results to stdout.

    Columns: model, (params), status, wall_seconds.
    """
    all_results: list[dict[str, Any]] = []
    for p in result_paths:
        all_results.extend(_load_results(p))

    if not all_results:
        print("No results.")
        return

    # Header
    print(f"{'model':<20s} {'params':<20s} {'status':<12s} {'wall_s':>8s}")
    print("-" * 64)

    for r in all_results:
        model = r.get("model", "?")
        keys_ordered = ["n", "k", "t", "lambda", "mu"]
        vals = [str(r[k]) for k in keys_ordered if k in r]
        params_str = "(" + ",".join(vals) + ")"
        status = r.get("status", "?")
        wall = r.get("wall_seconds", 0.0)
        print(f"{model:<20s} {params_str:<20s} {status:<12s} {wall:>8.2f}")
