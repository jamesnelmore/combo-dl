"""Sequential sweep runner for ILP benchmarking.

This module is **Gurobi-aware but model-agnostic**: it uses the model
registry in :mod:`ilp.models` to obtain a ``gp.Model``, then centrally
controls thread count, time limit, output suppression, and timing.  It
never inspects the constraints or variables of the model.

Key entry points:

* :func:`run_instance` — build + solve a single instance, return a result dict.
* :func:`run_sweep` — run a list of ``(model_name, params, config)`` tuples
  sequentially, saving results incrementally to a JSON file.

Design notes
------------
* All instances run **sequentially** — they're multicore solves so
  parallelism at the Python level would distort timings.
* Results are saved after every instance so a crash / Ctrl-C loses at most
  one data point.
* Already-completed instances (keyed by ``(model, params, config)``) are
  skipped on resume.
* The result schema is a flat dict suitable for loading into ``pandas``.
  Adjacency matrices are **not** stored (sweeps target known instances
  for speed benchmarking).

Hydra compatibility
-------------------
The interface is plain dicts + primitives so wrapping with ``@hydra.main``
later is straightforward: each CLI flag maps 1-to-1 to a Hydra config key,
and :func:`run_sweep` accepts the same ``list[dict]`` that Hydra's
``ListConfig`` resolves to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from ilp.models import get_builder


# ---------------------------------------------------------------------------
# Status extraction
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    GRB.OPTIMAL: "Optimal",
    GRB.INFEASIBLE: "Infeasible",
    GRB.TIME_LIMIT: "TimeLimit",
    GRB.SUBOPTIMAL: "Suboptimal",
    GRB.INF_OR_UNBD: "InfOrUnbd",
    GRB.UNBOUNDED: "Unbounded",
    GRB.NODE_LIMIT: "NodeLimit",
    GRB.SOLUTION_LIMIT: "SolutionLimit",
}


def _extract_status(model: gp.Model) -> str:
    return _STATUS_MAP.get(model.Status, f"Unknown({model.Status})")


# ---------------------------------------------------------------------------
# Instance key — used to detect already-completed work on resume
# ---------------------------------------------------------------------------

def _instance_key(model_name: str, params: dict, config: dict) -> str:
    """Deterministic string key for deduplication.

    Sorts dict keys so that ``{"n": 10, "k": 3}`` and ``{"k": 3, "n": 10}``
    produce the same key.
    """
    def _sorted_repr(d: dict) -> str:
        return str(sorted(d.items()))
    return f"{model_name}|{_sorted_repr(params)}|{_sorted_repr(config)}"


# ---------------------------------------------------------------------------
# Single instance
# ---------------------------------------------------------------------------

def run_instance(
    model_name: str,
    params: dict[str, Any],
    config: dict[str, Any],
    *,
    threads: int = -1,
    time_limit: float | None = None,
    heuristics: float | None = None,
    log_file: str | Path | None = None,
    seed: int | None = None,
    return_model: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], "gp.Model"]:
    """Build and solve a single model instance, returning a result dict.

    Args:
        model_name: Registered model name (e.g. ``"srg_exact"``).
        params: Graph parameters (``n``, ``k``, ``lambda``, ``mu``,
            optionally ``t``).
        config: Model configuration (``fix_neighbors``, ``lex_order``, etc.).
        threads: Gurobi thread count (-1 = solver default / all cores).
        time_limit: Wall-clock limit in seconds (``None`` = unlimited).
        heuristics: Fraction of solve time on MIP heuristics (0.0–1.0).
            ``None`` uses the Gurobi default (0.05).
        log_file: Path for Gurobi solver log (works independently of
            ``OutputFlag``).  ``None`` disables logging to file.
        seed: Gurobi random seed for reproducibility.  ``None`` uses
            the solver default.
        return_model: If ``True``, return ``(result, model)`` so the caller
            can extract solution data (e.g. adjacency matrices).

    Returns:
        Flat dict with at least ``model``, ``status``, ``wall_seconds``,
        plus all entries from *params* and *config*, and extended solver
        metrics (``node_count``, ``iter_count``, ``num_vars``, etc.).
        If *return_model* is ``True``, returns ``(result_dict, gp.Model)``.
    """
    builder = get_builder(model_name)
    model = builder(params, config)

    # ── Centralised Gurobi parameter control ──────────────────────────────
    # When logging to a file, keep OutputFlag=1 so Gurobi writes its full
    # solve log (node table, cuts, presolve, etc.) to LogFile.  Console
    # output goes to the SLURM .out file which is fine for HPC runs.
    # When no log file is requested, suppress console output.
    if log_file is not None:
        model.setParam("LogFile", str(log_file))
        model.setParam("OutputFlag", 1)
    else:
        model.setParam("OutputFlag", 0)
    if threads >= 0:
        model.setParam("Threads", threads)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if heuristics is not None:
        model.setParam("Heuristics", heuristics)
    if model_name.endswith("_exact"):
        model.setParam("MIPFocus", 1)
    if seed is not None:
        model.setParam("Seed", seed)

    # ── Solve + time ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - t0

    status = _extract_status(model)

    obj_val = None
    if model.SolCount > 0:
        try:
            obj_val = model.ObjVal
        except Exception:  # noqa: BLE001
            pass

    # ── Extended solver metrics ────────────────────────────────────────────
    def _safe_attr(attr: str) -> Any:
        try:
            return getattr(model, attr)
        except Exception:  # noqa: BLE001
            return None

    # ── Assemble result ───────────────────────────────────────────────────
    result: dict[str, Any] = {
        "model": model_name,
        "status": status,
        "wall_seconds": round(wall, 4),
        "obj_val": obj_val,
        "node_count": _safe_attr("NodeCount"),
        "iter_count": _safe_attr("IterCount"),
        "num_vars": _safe_attr("NumVars"),
        "num_constrs": _safe_attr("NumConstrs"),
        "num_gen_constrs": _safe_attr("NumGenConstrs"),
        "mip_gap": _safe_attr("MIPGap"),
        "obj_bound": _safe_attr("ObjBound"),
        "sol_count": _safe_attr("SolCount"),
        "runtime": _safe_attr("Runtime"),
    }
    # Flatten params and config into the result for easy DataFrame use.
    result.update(params)
    for k_cfg, v_cfg in config.items():
        result[f"cfg_{k_cfg}"] = v_cfg

    if return_model:
        return result, model
    return result


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    instances: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    threads: int = -1,
    time_limit: float | None = None,
    heuristics: float | None = None,
    seed: int | None = None,
    log_dir: str | Path | None = None,
    output_path: str | Path = "sweep_results.json",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run a batch of instances sequentially with incremental save.

    Args:
        instances: List of ``(model_name, params, config)`` tuples.
        threads: Gurobi thread count for every instance.
        time_limit: Per-instance wall-clock limit in seconds.
        heuristics: Fraction of solve time on MIP heuristics (0.0–1.0).
            ``None`` uses the Gurobi default (0.05).
        seed: Gurobi random seed for reproducibility.
        log_dir: Directory for per-instance Gurobi log files.  Each
            instance gets ``{log_dir}/{model}_{params}.log``.
        output_path: JSON file to write results to.  If the file already
            exists, completed instances are loaded and skipped.
        verbose: Print progress to stdout.

    Returns:
        The full list of result dicts (including any previously saved).
    """
    output_path = Path(output_path)

    # ── Load existing results for resume ──────────────────────────────────
    results: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
        except (json.JSONDecodeError, OSError):
            results = []

    done_keys: set[str] = set()
    for r in results:
        # Reconstruct the key from saved result.  We need to separate
        # params from config fields (cfg_* prefix).
        saved_model = r.get("model", "")
        _meta_keys = {
            "model", "status", "wall_seconds", "obj_val",
            "node_count", "iter_count", "num_vars", "num_constrs",
            "num_gen_constrs", "mip_gap", "obj_bound", "sol_count",
            "runtime",
        }
        saved_params = {
            k: v for k, v in r.items()
            if k not in _meta_keys
            and not k.startswith("cfg_")
        }
        saved_config = {
            k.removeprefix("cfg_"): v for k, v in r.items()
            if k.startswith("cfg_")
        }
        done_keys.add(_instance_key(saved_model, saved_params, saved_config))

    total = len(instances)
    skipped = 0

    for idx, (model_name, params, config) in enumerate(instances, 1):
        key = _instance_key(model_name, params, config)
        if key in done_keys:
            skipped += 1
            if verbose:
                label = _format_params(params)
                print(f"  [{idx}/{total}] skip {model_name} {label} (done)")
            continue

        label = _format_params(params)
        if verbose:
            print(
                f"  [{idx}/{total}] {model_name} {label} ...",
                end=" ",
                flush=True,
            )

        # Per-instance log file path.
        log_file = None
        if log_dir is not None:
            ld = Path(log_dir)
            ld.mkdir(parents=True, exist_ok=True)
            log_file = ld / f"{model_name}_{_format_params(params)}.log"

        result = run_instance(
            model_name,
            params,
            config,
            threads=threads,
            time_limit=time_limit,
            heuristics=heuristics,
            log_file=log_file,
            seed=seed,
        )

        results.append(result)
        done_keys.add(key)

        # Incremental save.
        output_path.write_text(json.dumps(results, indent=2))

        if verbose:
            print(f"{result['status']} in {result['wall_seconds']:.2f}s")

    if verbose:
        print(
            f"\nDone. {total - skipped} solved, {skipped} skipped. "
            f"Results in {output_path}"
        )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_params(params: dict) -> str:
    """Compact label string for a parameter set, e.g. ``(10,3,0,1)``."""
    keys_ordered = ["n", "k", "t", "lambda", "mu"]
    vals = [str(params[k]) for k in keys_ordered if k in params]
    return "(" + ",".join(vals) + ")"
