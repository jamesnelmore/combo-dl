"""Tests for the EdgeSwapEnv environment."""

from pathlib import Path
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from combo_dl.environments.edge_swap_env import RegularEdgeSwapEnv, _mask_actions
from combo_dl.problems import StronglyRegularGraphs


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

        # Pretty print the mask for debugging
        print("Mask after _mask_actions:")
        print(test_actions[0])  # mask is (1, 4, 4), print the first batch

        print("Expected mask")
        print(correct_actions[0])

        assert np.array_equal(test_actions, correct_actions)

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
