"""Model registry for SRG / DSRG ILP formulations.

Every builder in the registry conforms to a uniform interface::

    build(params: dict, config: dict) -> gp.Model

where *params* holds the graph parameters (``n``, ``k``, ``lambda``, ``mu``,
and optionally ``t`` for directed graphs) and *config* holds solver/model
options (``fix_neighbors``, ``lex_order``, etc.).

The runner in :mod:`ilp.bench.runner` uses this registry so it never needs
to know what constraints the model contains — it just calls the builder,
sets Gurobi parameters, and times the solve.

To add a new formulation, write a raw builder in ``srg.py`` or ``dsrg.py``
(or a new module), then register it here with :func:`register`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gurobipy as gp

# Type alias for the uniform builder protocol.
# params: {"n": int, "k": int, "lambda": int, "mu": int, optionally "t": int}
# config: {"fix_neighbors": bool, "lex_order": str, ...}
# Returns: a fully-built gp.Model ready for optimize().
BuilderFn = Callable[[dict[str, Any], dict[str, Any]], Any]

_REGISTRY: dict[str, BuilderFn] = {}


def register(name: str, fn: BuilderFn) -> None:
    """Register a builder function under *name*."""
    _REGISTRY[name] = fn


def get_builder(name: str) -> BuilderFn:
    """Look up a registered builder by *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model {name!r}. Available: {available}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Adapter helpers — wrap the raw module builders into the uniform interface.
# ---------------------------------------------------------------------------


def _build_srg_exact(params: dict, config: dict) -> "gp.Model":
    from .srg import build_srg_exact

    model, _edges, _e = build_srg_exact(
        n=params["n"],
        k=params["k"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        fix_neighbors=config.get("fix_neighbors", True),
        fix_v1=config.get("fix_v1", False),
        lex_order=config.get("lex_order", "none"),
        lex_block_size=config.get("lex_block_size", 20),
        quiet=True,
    )
    return model


def _build_srg_relaxed(params: dict, config: dict) -> "gp.Model":
    from .srg import build_srg_relaxed

    model, _edges, _e = build_srg_relaxed(
        n=params["n"],
        k=params["k"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        fix_neighbors=config.get("fix_neighbors", True),
        fix_v1=config.get("fix_v1", False),
        lex_order=config.get("lex_order", "none"),
        lex_block_size=config.get("lex_block_size", 20),
        quiet=True,
    )
    return model


def _build_srg_quadratic(params: dict, config: dict) -> "gp.Model":
    from .srg import build_srg_quadratic

    model, _edges, _e = build_srg_quadratic(
        n=params["n"],
        k=params["k"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        fix_neighbors=config.get("fix_neighbors", True),
        fix_v1=config.get("fix_v1", False),
        lex_order=config.get("lex_order", "none"),
        lex_block_size=config.get("lex_block_size", 20),
        quiet=True,
    )
    return model


def _build_dsrg_exact(params: dict, config: dict) -> "gp.Model":
    from .dsrg import build_dsrg_exact

    model, _edges = build_dsrg_exact(
        n=params["n"],
        k=params["k"],
        t=params["t"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        fix_neighbors=config.get("fix_neighbors", True),
        quiet=True,
    )
    return model


def _build_dsrg_relaxed(params: dict, config: dict) -> "gp.Model":
    from .dsrg import build_dsrg_relaxed

    model, _edges = build_dsrg_relaxed(
        n=params["n"],
        k=params["k"],
        t=params["t"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        fix_neighbors=config.get("fix_neighbors", True),
        quiet=True,
    )
    return model


def _build_cayley_dsrg(params: dict, config: dict) -> "gp.Model":
    from .cayley_dsrg import build_cayley_dsrg, load_cayley_data

    group_data = load_cayley_data(params["n"], config["lib_id"])
    model, _x = build_cayley_dsrg(
        n=params["n"],
        k=params["k"],
        t=params["t"],
        lambda_param=params["lambda"],
        mu=params["mu"],
        group_data=group_data,
        use_aut_pruning=config.get("use_aut_pruning", True),
        quiet=True,
    )
    return model


# ---------------------------------------------------------------------------
# Register the built-in formulations.
# ---------------------------------------------------------------------------

register("srg_exact", _build_srg_exact)
register("srg_relaxed", _build_srg_relaxed)
register("srg_quadratic", _build_srg_quadratic)
register("dsrg_exact", _build_dsrg_exact)
register("dsrg_relaxed", _build_dsrg_relaxed)
register("cayley_dsrg", _build_cayley_dsrg)
