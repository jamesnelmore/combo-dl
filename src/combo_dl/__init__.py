"""Combo DL: Graph optimization using deep learning."""

from . import (
    graph_utils,
    models,
)
from .deep_cross_entropy import WagnerDeepCrossEntropy
from .strongly_regular_graphs_problem import StronglyRegularGraphs
from .wagner_corollary_2_1_problem import WagnerCorollary21

__all__ = [
    "StronglyRegularGraphs",
    "WagnerCorollary21",
    "WagnerDeepCrossEntropy",
    "graph_utils",
    "models",
]

__version__ = "0.1.0"
