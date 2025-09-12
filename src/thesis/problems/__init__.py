"""Problems to be solved by algorithms."""

from .base_problem import BaseProblem
from .strongly_regular_graphs import StronglyRegularGraphs
from .wagner_corollary_2_1 import WagnerCorollary21

__all__ = ["BaseProblem", "StronglyRegularGraphs", "WagnerCorollary21"]

# TODO: Add comprehensive unit tests for all problem classes
# TODO: Add smoke tests to verify basic functionality without full optimization runs
