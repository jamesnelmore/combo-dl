from typing import override

import gymnasium as gym
import numpy as np
import torch

from combo_dl.graph_tools import gen_random_regular_graph
from combo_dl.problems import BaseProblem

# RegularEdgeSwapEnv for k-regular graphs. Takes a problem and degree k, not an SRG parameter set


# TODO Reimplement as a VecEnv, including with GPU optimization for torch and numpy operations
# All batchable operations should be implemented as batch ops using unsqueeze for now


class RegularEdgeSwapEnv(gym.Env):
    def __init__(self, problem: BaseProblem, n: int, k: int):
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

    @override
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:  # pyright: ignore[reportIncompatibleMethodOverride]
        super().reset(seed=seed)

        self.adj, self.edge_list = gen_random_regular_graph(self.n, self.k, seed=seed)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self) -> dict:
        return {"edge_list": self.edge_list, "node_features": self.node_indices}

    def _get_info(self) -> dict:  # noqa: PLR6301 | Will get real info eventually
        return {}

    def _calculate_reward(self) -> float:
        adj_torch = torch.from_numpy(self.adj).unsqueeze(0)
        return float(self.problem.reward(adj_torch).item())

    # TODO add shape and dtype to numpy type signatures

    def _mask_actions(self, actions: np.ndarray) -> np.ndarray:
        return _mask_actions(actions, self.edge_list, self.adj)


def _edge_exists(adj: np.ndarray, edge: np.ndarray) -> bool:
    """Check if an edge exists using the adjacency matrix.

    Args:
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
