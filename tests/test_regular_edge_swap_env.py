"""Tests for the EdgeSwapEnv environment."""

from pathlib import Path
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from combo_dl.environments.edge_swap_env import (
    RegularEdgeSwapEnv,
    _mask_actions,
    _perform_cross_swap_inplace,
    _perform_parallel_swap_inplace,
)
from combo_dl.problems import StronglyRegularGraphs


def assert_arrays_equal_with_debug(actual, expected, name="array"):
    """Assert arrays are equal with detailed debug output."""
    if not np.array_equal(actual, expected):
        print(f"\n{name} mismatch:")
        print(f"Expected:\n{expected}")
        print(f"Actual:\n{actual}")
        print(f"Diff:\n{expected - actual}")
        print(f"Shape - Expected: {expected.shape}, Actual: {actual.shape}")
        print(f"Data types - Expected: {expected.dtype}, Actual: {actual.dtype}")
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def peterson_env():
    """Create a RegularEdgeSwapEnv configured to find the Petersen graph.

    Returns:
        RegularEdgeSwapEnv: Environment set up to find the Petersen graph (n=10, k=3, λ=0, μ=1)
    """
    n = 10
    k = 3
    lambda_param = 0
    mu = 1
    problem = StronglyRegularGraphs(n, k, lambda_param, mu)
    return RegularEdgeSwapEnv(problem, n, k)


class TestRegularEdgeSwapEnv:
    """Test cases for RegularEdgeSwapEnv."""

    def test_environment_initialization_and_reset(self, peterson_env):
        """Test that the environment initializes correctly and can be reset."""
        # Test initialization
        assert peterson_env.n == 10
        assert peterson_env.k == 3
        assert peterson_env.num_edges == (10 * 3) // 2  # nk/2
        assert isinstance(peterson_env.problem, StronglyRegularGraphs)

        # Test observation space structure TODO figure out gym best practice
        # assert "edge_list" in env.observation_space.spaces
        # assert "node_features" in env.observation_space.spaces

        # Test reset
        obs, _ = peterson_env.reset(seed=42)

        # Test observation structure
        assert isinstance(obs, dict)
        assert "edge_list" in obs
        assert "node_features" in obs

        # Test edge_list shape and values
        edge_list = obs["edge_list"]
        assert edge_list.shape == (15, 2), f"Edge List: {edge_list}"
        assert edge_list.dtype == np.int32
        assert np.all(edge_list >= 0)
        assert np.all(edge_list < 10)  # n=10 nodes

        # Test node_features
        node_features = obs["node_features"]
        assert node_features.shape == (10,)  # n=10 nodes
        assert np.array_equal(node_features, np.arange(10))

    def test_mask_action_manual(self):
        edge_list = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=int)
        # fmt: off
        adj = np.array(
            [[0, 1, 0, 0],
             [1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0]]
        )
        correct_actions = np.array(
            [[0, 0, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        # fmt: off

        # Unsqueeze to add batch dimension
        adj = np.expand_dims(adj, axis=0)  # shape: (1, 4, 4)
        correct_actions = np.expand_dims(correct_actions, axis=0)  # shape: (1, 4, 4)
        test_actions = np.ones_like(adj)
        test_actions = _mask_actions(test_actions, edge_list, adj)

        # Debug output if assertion fails
        if not np.array_equal(test_actions, correct_actions):
            print("\nMask mismatch:")
            print(f"Expected:\n{correct_actions[0]}")
            print(f"Actual:\n{test_actions[0]}")
            print(f"Diff:\n{correct_actions[0] - test_actions[0]}")

        np.testing.assert_array_equal(test_actions, correct_actions)

    def test_mask_action_ai(self, peterson_env):
        """Test that action masking correctly identifies invalid edge swaps."""
        # Reset environment to get initial state
        peterson_env.reset(seed=42)

        # Create a simple action matrix (batch_size=1, num_edges=15, num_edges=15)
        batch_size = 1
        num_edges = peterson_env.num_edges
        action_scores = np.ones((batch_size, num_edges, num_edges))

        # Apply masking
        masked_actions = peterson_env._mask_actions(action_scores)

        # Check that diagonal is masked (can't swap edge with itself)
        for b in range(batch_size):
            assert np.all(masked_actions[b].diagonal() == 0)

        # Check that some actions are still valid (not all masked)
        assert np.any(masked_actions > 0), "All actions were masked, which shouldn't happen"

        # Check that diagonal is fully masked
        for b in range(batch_size):
            assert np.all(masked_actions[b].diagonal() == 0)

        # Note: Upper and lower triangles represent different swap types:
        # - Upper (i<j): parallel swap (x,y),(u,v) -> (x,u),(y,v)
        # - Lower (i>j): cross swap (x,y),(u,v) -> (x,v),(y,u)
        # So symmetry doesn't hold - they're different operations

    def test_parallel_edge_swap(self):
        # fmt: off
        two_cycle_edge_list = np.array(
            [[0, 1],
             [1, 2],
             [2, 3],
             [0, 3]])
        two_cycle_adj = np.array(
            [[0, 1, 0, 1],
             [1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 0]])

        expected_edge_list = np.array(
            [[0, 2],
             [1, 2],
             [1, 3],
             [0, 3]])
        expected_adj = np.array(
            [[0, 0, 1, 1],
             [0, 0, 1, 1],
             [1, 1, 0, 0],
             [1, 1, 0, 0]])
        # fmt: on

        i, j = 0, 2  # Parallel swap (0,1), (2,3) -> (0,2), (1,3)
        _perform_parallel_swap_inplace(i, j, two_cycle_adj, two_cycle_edge_list)

        np.testing.assert_array_equal(two_cycle_edge_list, expected_edge_list)
        print("Edge lists are identical")
        np.testing.assert_array_equal(two_cycle_adj, expected_adj)

    def test_cross_edge_swap(self):
        # fmt: off
        two_cycle_edge_list = np.array(
            [[0, 1],
             [1, 2],
             [2, 3],
             [0, 3]])
        two_cycle_adj = np.array(
            [[0, 1, 0, 1],
             [1, 0, 1, 0],
             [0, 1, 0, 1],
             [1, 0, 1, 0]])

        expected_edge_list = np.array(
            [[0, 1],
             [1, 3],
             [2, 3],
             [0, 2]])
        expected_adj = np.array(
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]])
        # fmt: on

        i, j = 1, 3  # Cross swap (1, 2), (0, 3) -> (1, 3), (0, 2)
        _perform_cross_swap_inplace(i, j, two_cycle_adj, two_cycle_edge_list)

        np.testing.assert_array_equal(two_cycle_edge_list, expected_edge_list)
        print("Edge lists are identical")
        np.testing.assert_array_equal(two_cycle_adj, expected_adj)
