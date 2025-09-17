"""Tests for the EdgeSwapEnv environment."""

from pathlib import Path
import sys

import gymnasium as gym
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

# Import for Stable-Baselines3 integration test
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False


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


@pytest.fixture
def srg_env():
    """Create a RegularEdgeSwapEnv configured for SRG problem.

    Returns:
        RegularEdgeSwapEnv: Environment set up for SRG problem (n=8, k=3)
    """
    n = 8
    k = 3
    problem = StronglyRegularGraphs(n, k, lambda_param=1, mu=2)
    return RegularEdgeSwapEnv(problem, n, k, max_steps=50)  # Short episodes for testing


@pytest.fixture
def small_env():
    """Create a small environment for quick testing.

    Returns:
        RegularEdgeSwapEnv: Small environment (n=6, k=2) with short episodes
    """
    n = 6
    k = 2
    problem = StronglyRegularGraphs(n, k, lambda_param=0, mu=1)
    return RegularEdgeSwapEnv(problem, n, k, max_steps=20)


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


class TestEpisodeCompletion:
    """Test cases for episode completion scenarios."""

    def test_environment_spin_up(self, small_env):
        """Test that environment can be initialized and reset properly."""
        # Test initialization
        assert small_env.n == 6
        assert small_env.k == 2
        assert small_env.max_steps == 20
        assert small_env.step_count == 0

        # Test reset
        obs, info = small_env.reset(seed=42)

        # Check observation structure
        assert isinstance(obs, dict)
        assert "edge_list" in obs
        assert "node_features" in obs
        assert obs["edge_list"].shape == (6, 2)  # 6 edges for n=6, k=2
        assert obs["node_features"].shape == (6,)

        # Check info structure
        assert isinstance(info, dict)
        assert "step_count" in info
        assert "best_reward_seen" in info
        assert "last_improvement_step" in info
        assert info["step_count"] == 0
        assert info["best_reward_seen"] == float("-inf")

    def test_single_step_execution(self, small_env):
        """Test that a single step can be executed properly."""
        obs, info = small_env.reset(seed=42)
        initial_reward = small_env._calculate_reward()

        # Take a valid action (swap first two edges)
        action = np.array([0, 1])
        obs, reward, terminated, truncated, info = small_env.step(action)

        # Check step execution
        assert small_env.step_count == 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        # Check info tracking
        assert info["step_count"] == 1
        assert info["best_reward_seen"] >= initial_reward or info["best_reward_seen"] == reward
        assert info["last_improvement_step"] >= 0

    def test_invalid_action_handling(self, small_env):
        """Test that invalid actions (self-swap) are handled correctly."""
        obs, info = small_env.reset(seed=42)
        initial_step_count = small_env.step_count

        # Take invalid action (swap edge with itself)
        action = np.array([0, 0])
        obs, reward, terminated, truncated, info = small_env.step(action)

        # Check that step count still increments
        assert small_env.step_count == initial_step_count + 1
        assert reward == -1.0  # Invalid action penalty
        assert not terminated  # Should not terminate on invalid action
        assert not truncated  # Should not truncate on invalid action

    def test_episode_truncation_by_max_steps(self, small_env):
        """Test that episodes are truncated when max_steps is reached."""
        # Temporarily disable stagnation detection to test max_steps truncation
        original_stagnation_threshold = small_env.stagnation_threshold
        small_env.stagnation_threshold = small_env.max_steps + 10  # Much higher than max_steps

        try:
            obs, info = small_env.reset(seed=42)

            # Take actions until max_steps is reached
            for step in range(small_env.max_steps):
                # Use a simple action pattern
                action = np.array([step % small_env.num_edges, (step + 1) % small_env.num_edges])
                obs, reward, terminated, truncated, info = small_env.step(action)

                if step < small_env.max_steps - 1:
                    # Should not be truncated until max_steps is reached
                    assert not truncated, f"Episode truncated early at step {step}"
                else:
                    # Should be truncated at max_steps
                    assert truncated, f"Episode not truncated at max_steps {small_env.max_steps}"
                    assert info["step_count"] == small_env.max_steps
        finally:
            # Restore original stagnation threshold
            small_env.stagnation_threshold = original_stagnation_threshold

    def test_episode_termination_by_goal_achievement(self, srg_env):
        """Test that episodes terminate when goal is achieved."""
        obs, info = srg_env.reset(seed=42)

        # Get the goal score
        goal_score = srg_env.problem.get_goal_score()
        assert goal_score is not None, "SRG problem should have a goal score"

        # Simulate achieving the goal by manually setting a high reward
        # This is a bit of a hack, but necessary for testing termination
        srg_env._best_reward_seen = goal_score + 0.1

        # Check that should_stop_early returns True
        should_stop, reason = srg_env.problem.should_stop_early(goal_score + 0.1)
        assert should_stop, "should_stop_early should return True when goal is achieved"
        assert "Goal achieved" in reason

        # Check that _should_terminate returns True
        assert srg_env._should_terminate(), "Episode should terminate when goal is achieved"

    def test_episode_termination_by_convergence(self, small_env):
        """Test that episodes terminate when rewards converge."""
        obs, info = small_env.reset(seed=42)

        # Fill reward history with identical rewards to simulate convergence
        identical_reward = 0.5
        for _ in range(small_env.convergence_window):
            small_env._reward_history.append(identical_reward)

        # Check that convergence is detected
        assert len(small_env._reward_history) >= small_env.convergence_window
        recent_rewards = small_env._reward_history[-small_env.convergence_window :]
        reward_variance = np.var(recent_rewards)
        assert reward_variance < small_env.convergence_threshold

        # Check that _should_terminate returns True
        assert small_env._should_terminate(), "Episode should terminate when rewards converge"

    def test_stagnation_detection(self, small_env):
        """Test that episodes are truncated when no improvement occurs."""
        obs, info = small_env.reset(seed=42)

        # Simulate stagnation by setting last improvement far in the past
        small_env._last_improvement_step = 0
        small_env.step_count = small_env.stagnation_threshold + 1

        # Check that _should_truncate returns True
        assert small_env._should_truncate(), "Episode should be truncated due to stagnation"

    def test_full_episode_lifecycle(self, small_env):
        """Test a complete episode from reset to completion."""
        # Reset environment
        obs, info = small_env.reset(seed=42)
        initial_reward = small_env._calculate_reward()

        episode_rewards = []
        episode_length = 0

        # Run episode until completion
        while True:
            # Take a random valid action
            edge1 = episode_length % small_env.num_edges
            edge2 = (episode_length + 1) % small_env.num_edges
            if edge1 == edge2:
                edge2 = (edge2 + 1) % small_env.num_edges

            action = np.array([edge1, edge2])
            obs, reward, terminated, truncated, info = small_env.step(action)

            episode_rewards.append(reward)
            episode_length += 1

            # Episode should end at some point
            assert episode_length <= small_env.max_steps + 10, "Episode ran too long"

            if terminated or truncated:
                break

        # Check episode completion
        assert episode_length > 0, "Episode should have at least one step"
        assert terminated or truncated, "Episode should have ended"
        assert info["step_count"] == episode_length
        assert len(episode_rewards) == episode_length

        # Check that some progress was made (not all rewards are identical)
        if len(set(episode_rewards)) > 1:
            assert info["best_reward_seen"] > float("-inf"), "Should have seen some improvement"

    def test_multiple_episode_resets(self, small_env):
        """Test that environment can be reset multiple times correctly."""
        # Run first episode
        obs1, info1 = small_env.reset(seed=42)
        action = np.array([0, 1])
        obs1, reward1, term1, trunc1, info1 = small_env.step(action)

        # Reset and run second episode
        obs2, info2 = small_env.reset(seed=123)
        action = np.array([0, 1])
        obs2, reward2, term2, trunc2, info2 = small_env.step(action)

        # Check that episodes are independent
        assert small_env.step_count == 1, "Step count should reset"
        assert info2["step_count"] == 1, "Info should reset"
        assert info2["best_reward_seen"] > float("-inf"), "Reward tracking should reset"

        # Observations might be different due to different seeds
        assert obs1["edge_list"].shape == obs2["edge_list"].shape
        assert obs1["node_features"].shape == obs2["node_features"].shape

    def test_termination_reason_tracking(self, small_env):
        """Test that termination reasons are properly tracked."""
        obs, info = small_env.reset(seed=42)

        # Test convergence termination reason
        for _ in range(small_env.convergence_window):
            small_env._reward_history.append(0.5)

        # Force termination by convergence
        terminated = small_env._should_terminate()
        assert terminated, "Should terminate due to convergence"

        # Check that termination reason is set
        assert hasattr(small_env, "_termination_reason")
        assert "Converged" in small_env._termination_reason
        assert "reward variance" in small_env._termination_reason

        # Check that info includes termination reason
        info = small_env._get_info()
        assert "termination_reason" in info
        assert info["termination_reason"] == small_env._termination_reason

    def test_episode_info_completeness(self, small_env):
        """Test that episode info contains all expected fields."""
        obs, info = small_env.reset(seed=42)

        # Take a few steps
        for i in range(3):
            action = np.array([i % small_env.num_edges, (i + 1) % small_env.num_edges])
            obs, reward, terminated, truncated, info = small_env.step(action)

        # Check all expected info fields
        expected_fields = ["step_count", "best_reward_seen", "last_improvement_step"]
        for field in expected_fields:
            assert field in info, f"Missing field: {field}"

        # Check field types and values
        assert isinstance(info["step_count"], int)
        assert info["step_count"] == 3
        assert isinstance(info["best_reward_seen"], float)
        assert isinstance(info["last_improvement_step"], int)
        assert info["last_improvement_step"] >= 0


@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable-Baselines3 not available")
class TestStableBaselinesIntegration:
    """Integration tests with Stable-Baselines3 to ensure the environment works end-to-end."""

    def test_ppo_training_run(self, small_env):
        """Test that PPO can train on the environment without crashing."""
        # Create a vectorized environment for faster training
        vec_env = make_vec_env(
            lambda: RegularEdgeSwapEnv(
                StronglyRegularGraphs(4, 2, lambda_param=0, mu=1), n=4, k=2, max_steps=10
            ),
            n_envs=2,  # Small number for quick test
            seed=42,
        )

        # Create PPO model with minimal settings for quick test
        model = PPO(
            "MultiInputPolicy",  # Use Multi-Input policy for dict observation space
            vec_env,
            verbose=0,  # Suppress output
            learning_rate=3e-4,
            n_steps=8,  # Small number of steps per update
            batch_size=4,  # Small batch size
            n_epochs=2,  # Minimal epochs
            ent_coef=0.01,
            device="cpu",  # Use CPU for consistency
        )

        # Train for a very small number of timesteps
        try:
            model.learn(total_timesteps=50)  # Very short training run
            print("PPO training completed successfully!")
        except Exception as e:
            pytest.fail(f"PPO training failed with error: {e}")

        # Test that the trained model can make predictions
        try:
            obs = vec_env.reset()
            action, _states = model.predict(obs, deterministic=True)

            # Verify action shape is correct
            assert action.shape == (2, 2), f"Expected action shape (2, 2), got {action.shape}"
            assert action.dtype == np.int32 or action.dtype == np.int64, (
                f"Expected integer action type, got {action.dtype}"
            )

            # Test stepping with the predicted action
            obs, rewards, dones, infos = vec_env.step(action)

            # Verify step results
            assert obs.keys() == {"edge_list", "node_features"}
            assert rewards.shape == (2,)
            assert dones.shape == (2,)
            assert len(infos) == 2

            print("PPO prediction and stepping completed successfully!")

        except Exception as e:
            pytest.fail(f"PPO prediction/stepping failed with error: {e}")

        # Clean up
        vec_env.close()

    def test_environment_compatibility_with_sb3(self, small_env):
        """Test that the environment is fully compatible with Stable-Baselines3."""
        # Test basic environment properties that SB3 expects
        assert hasattr(small_env, "observation_space")
        assert hasattr(small_env, "action_space")
        assert hasattr(small_env, "reset")
        assert hasattr(small_env, "step")

        # Test observation space compatibility
        obs_space = small_env.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "edge_list" in obs_space.spaces
        assert "node_features" in obs_space.spaces

        # Test action space compatibility
        action_space = small_env.action_space
        assert isinstance(action_space, gym.spaces.MultiDiscrete)
        assert action_space.shape == (2,)

        # Test that reset returns correct types
        obs, info = small_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        # Test that step returns correct types
        action = np.array([0, 1])  # Valid action
        obs, reward, terminated, truncated, info = small_env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        print("Environment compatibility with Stable-Baselines3 verified!")

    def test_multiple_episodes_with_sb3(self, small_env):
        """Test that multiple episodes can be run with SB3 without issues."""

        # Create a simple environment wrapper for SB3
        def make_env():
            return RegularEdgeSwapEnv(
                StronglyRegularGraphs(4, 2, lambda_param=0, mu=1), n=4, k=2, max_steps=5
            )

        vec_env = make_vec_env(make_env, n_envs=1, seed=123)

        # Create a minimal PPO model
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=0,
            learning_rate=1e-3,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            device="cpu",
        )

        # Run multiple episodes
        total_rewards = []
        for episode in range(3):
            obs = vec_env.reset()
            episode_reward = 0

            for step in range(10):  # Max 10 steps per episode
                action, _ = model.predict(obs, deterministic=False)
                obs, rewards, dones, infos = vec_env.step(action)
                episode_reward += rewards[0]

                if dones[0]:
                    break

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} completed with reward: {episode_reward}")

        # Verify that episodes completed successfully
        assert len(total_rewards) == 3, "Should have completed 3 episodes"

        # Debug reward types
        for i, reward in enumerate(total_rewards):
            print(f"Episode {i + 1} reward: {reward} (type: {type(reward)})")

        assert all(isinstance(r, (int, float, np.number)) for r in total_rewards), (
            f"All rewards should be numeric, got types: {[type(r) for r in total_rewards]}"
        )

        print(f"Multiple episodes completed successfully! Rewards: {total_rewards}")

        # Clean up
        vec_env.close()
