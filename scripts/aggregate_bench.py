#!/usr/bin/env python3
"""Aggregate bench-single results into a summary CSV.

Walks a benchmark output directory (one subdirectory per array task),
reads each ``result.json``, and combines them into a single CSV sorted
by parameter set index.

Usage::

    python scripts/aggregate_bench.py bench_output/<JOB_ID>
    python scripts/aggregate_bench.py bench_output/<JOB_ID> -o summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def aggregate(bench_dir: Path, output: Path | None = None) -> pd.DataFrame:
    """Read all result.json files and return a combined DataFrame."""
    results = []

    for result_file in sorted(bench_dir.glob("*/result.json")):
        try:
            data = json.loads(result_file.read_text())
            results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {result_file}: {exc}", file=sys.stderr)

    if not results:
        print(f"No result.json files found in {bench_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(results)

    # Sort by index (array task ID) if present.
    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)

    # Reorder columns: index, params, status/timing first, then metrics.
    param_cols = ["index", "n", "k", "lambda", "mu", "t"]
    status_cols = ["model", "status", "wall_seconds", "runtime"]
    metric_cols = [
        "node_count", "iter_count", "sol_count", "mip_gap", "obj_val",
        "obj_bound", "num_vars", "num_constrs", "num_gen_constrs",
    ]
    cfg_cols = sorted(c for c in df.columns if c.startswith("cfg_"))
    other_cols = [
        c for c in df.columns
        if c not in param_cols + status_cols + metric_cols + cfg_cols
    ]
    ordered = [c for c in param_cols + status_cols + metric_cols + cfg_cols + other_cols if c in df.columns]
    df = df[ordered]

    out_path = output or (bench_dir / "summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} results to {out_path}")

    # Print summary table.
    print()
    display_cols = [c for c in ["index", "n", "k", "lambda", "mu", "status", "wall_seconds", "node_count"] if c in df.columns]
    print(df[display_cols].to_string(index=False))

    # Status counts.
    print(f"\nStatus counts:")
    for status, count in df["status"].value_counts().items():
        print(f"  {status}: {count}")

    return df


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: aggregate_bench.py bench_output/<JOB_ID> [-o summary.csv]")
        sys.exit(1)

    bench_dir = Path(sys.argv[1])
    output = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "-o" and i + 1 < len(sys.argv):
            output = Path(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    aggregate(bench_dir, output)


if __name__ == "__main__":
    main()
