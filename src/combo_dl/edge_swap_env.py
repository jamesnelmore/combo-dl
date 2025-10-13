from typing import override

import gymnasium as gym
import numpy as np
import torch

from combo_dl import StronglyRegularGraphs
from combo_dl.graph_utils import gen_random_regular_graph

# RegularEdgeSwapEnv for k-regular graphs. Takes a problem and degree k, not an SRG parameter set


# TODO Reimplement as a VecEnv, including with GPU optimization for torch and numpy operations
# All batchable operations should be implemented as batch ops using unsqueeze for now


class RegularEdgeSwapEnv(gym.Env):
    def __init__(
        self, problem: StronglyRegularGraphs, n: int, k: int, max_steps: int | None = None
    ):
        """Initialize RegularEdgeSwapEnv for k-regular graphs.

        Args:
            problem: The optimization problem to solve
            n: Number of vertices in the graph
            k: Degree of each vertex (regularity)
            max_steps: Maximum number of steps per episode. Defaults to 2*n*k

        Raises:
            ValueError: If n or k are invalid, or if n*k is odd, or if n < k+1
        """
        super().__init__()
        self.problem = problem

        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        if k <= 0:
            raise ValueError("k must be positive")
        if not (n * k % 2 == 0):
            raise ValueError("n * k must be even")
        if not (n >= k + 1):
            raise ValueError("n must be greater than or equal to k + 1")
        self.k = k
        self.num_edges = (n * k) // 2

        # Termination and truncation parameters
        self.max_steps = max_steps or (n * k * 2)  # Default: 2x the number of edges
        self.convergence_window = min(50, self.max_steps // 10)
        self.convergence_threshold = 1e-6
        self.stagnation_threshold = self.max_steps // 2

        # Episode state tracking
        self.step_count = 0
        self._reward_history = []
        self._best_reward_seen = float("-inf")
        self._last_improvement_step = 0
        self._termination_reason = None

        self.observation_space = gym.spaces.Dict({
            "edge_list": gym.spaces.Box(
                low=0,
                high=n - 1,  # Node indices range from 0 to n-1
                shape=(self.num_edges, 2),
                dtype=np.int32,
            ),
            "node_features": gym.spaces.MultiDiscrete(
                [n] * n
            ),  # Each node has its own index (0 to n-1)
            # TODO add more meaningful features
        })
        self.node_indices = torch.arange(n)

        # Conceptually the action space is 2 edges and whether to cross swap or parallel swap them.
        # Each swap type is commutative, so this could be represented by the upper and lower
        # triangles of a num_edges x num_edges matrix, where a self swap is always illegal so the
        # diagonal doesn't matter.
        self.action_space = gym.spaces.MultiDiscrete([self.num_edges, self.num_edges])

    @override
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:  # pyright: ignore[reportIncompatibleMethodOverride]
        super().reset(seed=seed)

        self.adj, self.edge_list = gen_random_regular_graph(self.n, self.k, seed=seed)

        # Reset episode state
        self.step_count = 0
        self._reward_history = []
        self._best_reward_seen = float("-inf")
        self._last_improvement_step = 0
        self._termination_reason = None

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> dict:
        return {"edge_list": self.edge_list, "node_features": self.node_indices}

    # TODO add logging

    def _get_info(self) -> dict:
        """Get episode information including termination reason if available.

        Returns:
            Dictionary containing episode information
        """
        info = {
            "step_count": self.step_count,
            "best_reward_seen": self._best_reward_seen,
            "last_improvement_step": self._last_improvement_step,
        }

        # Add termination reason if available
        if hasattr(self, "_termination_reason"):
            info["termination_reason"] = self._termination_reason

        return info

    def _should_terminate(self) -> bool:
        """Check if episode should terminate due to goal achievement or convergence.

        Returns:
            True if episode should terminate naturally, False otherwise
        """
        # Check goal achievement using should_stop_early
        # Use the best reward seen rather than current reward for termination
        should_stop, reason = self.problem.should_stop_early(self._best_reward_seen)
        if should_stop:
            self._termination_reason = reason
            return True

        # Check convergence (reward stability)
        if len(self._reward_history) >= self.convergence_window:
            recent_rewards = self._reward_history[-self.convergence_window :]
            if len(recent_rewards) > 1:  # Need at least 2 values to compute variance
                reward_variance = np.var(recent_rewards)
                if reward_variance < self.convergence_threshold:
                    self._termination_reason = (
                        f"Converged: reward variance {reward_variance:.6f} < "
                        f"{self.convergence_threshold}"
                    )
                    return True

        return False

    def _should_truncate(self) -> bool:
        """Check if episode should be truncated due to length limits or stagnation.

        Returns:
            True if episode should be truncated, False otherwise
        """
        # Maximum episode length
        if self.step_count >= self.max_steps:
            return True

        # Stagnation detection (no improvement for too long)
        return self.step_count - self._last_improvement_step >= self.stagnation_threshold

    def _calculate_reward(self) -> float:
        adj_torch = torch.from_numpy(self.adj).unsqueeze(0)
        return float(self.problem.reward(adj_torch).item())

    # TODO add shape and dtype to numpy type signatures

    def _mask_actions(self, actions: np.ndarray) -> np.ndarray:
        return _mask_actions(actions, self.edge_list, self.adj)

    @override
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one swap on the current graph.

        Args:
            action: Integer of shape (2,) containing [i,j] where:
                - i, j are edge indices
                - If i > j, perform parallel swap
                - If i < j, perform cross swap
                - If i == j, invalid action/null swap

        Returns:
            observation, reward, terminated, truncated, info
        """
        i, j = int(action[0]), int(action[1])  # TODO handle invalid shape

        # Track step count
        self.step_count += 1

        if i == j:
            reward = -1.0
            # TODO info about failed action
        else:
            # Perform the swap
            if i > j:
                _perform_parallel_swap_inplace(i, j, self.adj, self.edge_list)
            else:
                _perform_cross_swap_inplace(i, j, self.adj, self.edge_list)

            # Calculate reward
            reward = self._calculate_reward()

            # Track reward history and improvements
            self._reward_history.append(reward)
            if reward > self._best_reward_seen:
                self._best_reward_seen = reward
                self._last_improvement_step = self.step_count

            # Keep reward history bounded
            if len(self._reward_history) > self.convergence_window * 2:
                self._reward_history = self._reward_history[-self.convergence_window :]

        terminated = self._should_terminate()
        truncated = self._should_truncate()

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    @classmethod
    def from_srg_problem(
        cls, problem: StronglyRegularGraphs, max_steps: int | None = None
    ) -> "RegularEdgeSwapEnv":
        """Builds a RegularEdgeSwapEnv compatible with a given SRG problem.

        Returns:
            Compatible edge swap env
        """
        return RegularEdgeSwapEnv(problem, problem.n, problem.k, max_steps=max_steps)


def _perform_parallel_swap_inplace(i: int, j: int, adj: np.ndarray, edges: np.ndarray) -> None:
    r"""Performs a parallel edge swap on the adjacency matrix adj and edge list edges.

    A parallel edge swap is defined as
    .. math::
        (x, y), (u, v) \\rightarrow (x, u), (y, v)

    This function does no checks to ensure the swap does not create a loop or duplicate edge.

    Args:
        i: index of first edge to swap
        j: index of second edge to swap
        adj: adjacency matrix to mutate
        edges: edge list to mutate
    """
    # Swap edge list
    # (x,y), (u,v) -> (x, u), (y,v)

    x, y = edges[i, 0], edges[i, 1]
    u, v = edges[j, 0], edges[j, 1]

    edges[i] = np.array([min(x, u), max(x, u)])  # (x,y) -> (x,y) [properly sorted]
    edges[j] = np.array([min(y, v), max(y, v)])  # (u,v) -> (y,v) [properly sorted]

    # Swap Adjacency Matrix
    adj[x, y] = adj[y, x] = 0
    adj[u, v] = adj[v, u] = 0
    adj[x, u] = adj[u, x] = 1
    adj[y, v] = adj[v, y] = 1


def _perform_cross_swap_inplace(i: int, j: int, adj: np.ndarray, edges: np.ndarray) -> None:
    r"""Performs a cross edge swap on the adjacency matrix adj and edge list edges.

    A cross edge swap is defined as
    .. math::
        (x, y), (u, v) \\rightarrow (x, v), (y, u)

    This function does no checks to ensure the swap does not create a loop or duplicate edge.

    Args:
        i: index of first edge to swap
        j: index of second edge to swap
        adj: adjacency matrix to mutate
        edges: edge list to mutate
    """
    x, y = edges[i, 0], edges[i, 1]
    u, v = edges[j, 0], edges[j, 1]

    edges[i] = np.array([min(x, v), max(x, v)])  # (x,y) -> (x,v) [properly sorted]
    edges[j] = np.array([min(y, u), max(y, u)])  # (u,v) -> (y,u) [properly sorted]

    # Swap Adjacency Matrix
    adj[x, y] = adj[y, x] = 0
    adj[u, v] = adj[v, u] = 0
    adj[x, v] = adj[v, x] = 1
    adj[y, u] = adj[u, y] = 1


def _edge_exists(adj: np.ndarray, edge: np.ndarray) -> bool:
    """Check if an edge exists using the adjacency matrix.

    Args:
        adj: Adjacency matrix to check against
        edge: Edge to check [u, v]

    Returns:
        True if edge exists, False otherwise
    """
    u, v = edge[0], edge[1]
    return adj[u, v] == 1


def _mask_actions(actions: np.ndarray, edge_list: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Mask invalid edge swap actions.

    Args:
        actions: Action scores of shape (batch_size, num_edges, num_edges)
        edge_list: Edge list of graph being manipulated, where u < v for all edges (u,v)
        adj: Adjacency matrix of graph being manipulated

    Returns:
        Masked action scores where invalid actions are set to 0
    """
    batch_size, num_edges, _ = actions.shape
    assert batch_size == 1, "Vectorization not supported yet."
    mask = np.ones_like(actions, dtype=bool)

    # Not on diagonal
    diag_indices = np.arange(num_edges)
    mask[np.arange(batch_size), diag_indices, diag_indices] = False

    for b in range(batch_size):
        current_edges = edge_list
        for i in range(num_edges):
            for j in range(i, num_edges):
                if i == j:
                    continue
                edge1 = current_edges[i]  # [x, y]
                edge2 = current_edges[j]  # [u, v]

                x = edge1[0]
                y = edge1[1]

                u = edge2[0]
                v = edge2[1]

                # If any indices overlap, this causes a loop, so mask out parallel and cross swap
                if x in {u, v} or y in {u, v}:
                    mask[b, i, j] = 0
                    mask[b, j, i] = 0
                    continue  # Skip further checks if indices overlap

                # See if new edges already exist

                # Parallel edges: (x,y),(u,v) -> (x,u),(y,v)
                parallel_edge_1 = np.sort(np.array([x, u]))
                parallel_edge_2 = np.sort(np.array([y, v]))

                # Cross edges: (x,y),(u,v) -> (x,v),(y,u)
                cross_edge_1 = np.sort(np.array([x, v]))
                cross_edge_2 = np.sort(np.array([y, u]))

                # Check if parallel edges already exist
                parallel_exists_1 = np.any(np.all(current_edges == parallel_edge_1, axis=1))
                parallel_exists_2 = np.any(np.all(current_edges == parallel_edge_2, axis=1))
                if parallel_exists_1 or parallel_exists_2:
                    mask[b, i, j] = 0

                # Check if cross edges already exist
                cross_exists_1 = np.any(np.all(current_edges == cross_edge_1, axis=1))
                cross_exists_2 = np.any(np.all(current_edges == cross_edge_2, axis=1))
                if cross_exists_1 or cross_exists_2:
                    mask[b, j, i] = 0

    return actions * mask
