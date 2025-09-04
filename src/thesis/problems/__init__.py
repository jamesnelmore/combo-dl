from .base_problem import BaseProblem
from .strongly_regular_graphs import StronglyRegularGraphs
from .wagner_corollary_2_1 import WagnerCorollary2_1

__all__ = ["BaseProblem", "StronglyRegularGraphs", "WagnerCorollary2_1"]

# TODO wandb doesn't appear to be working when running the srg_config experiment
# TODO Walk through entire codebase
# TODO add testing
# TODO add smoke testing
# TODO set up CI
