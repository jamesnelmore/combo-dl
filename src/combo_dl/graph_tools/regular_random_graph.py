import networkx as nx
import numpy as np


def _swap_edges(edge_list: np.ndarray, num_swaps: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # TODO add shape checks
    # TODO add batching

    for _ in range(num_swaps):
        # Select 2 edges: (u,v) and (x,y)
        edges = rng.choice(edge_list, size=2, replace=False)
        assert edges.shape == (4, 4)

        # Create new edges

        # Ensure new edges do not exist

        # Add new edges and remove old one

    return edge_list


def generate_random_regular_graph(n: int, k: int, seed: int | None = None) -> np.ndarray:
    if not (n * k % 2 == 0 and n >= k + 1):
        raise ValueError("n * k must be even and n >= k + 1.")

    offsets = []
    if k % 2 == 0:
        r = k // 2
        offsets = list(range(1, r + 1))
    else:
        # n must be even by the initial check
        m = n // 2
        r = (k - 1) // 2
        offsets = [*list(range(1, r + 1)), m]

    circulant: nx.Graph = nx.circulant_graph(n, offsets)
    edge_list = np.array(list(circulant.edges()))

    # Apply edge swapping if seed is provided

    # final_edge_list = _swap_edges(edge_list, n**2, seed)
    final_edge_list = edge_list
    A = np.zeros(shape=(n, n), dtype=int)

    # Convert edge list to adjacency matrix (vectorized)
    # Set both directions of each edge
    A[final_edge_list[:, 0], final_edge_list[:, 1]] = 1
    A[final_edge_list[:, 1], final_edge_list[:, 0]] = 1

    return A
