import networkx as nx
import numpy as np


def _swap_edges(edge_list: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Randomly swap edges O(n^2) times. Swaps occur in place.

    Args:
        edge_list: edege list of shape (*, 2).
        n: Will swap between .5*n^2 and 2*n^2 edges. Intended to be number of vertices.
        seed: random seed.

    Returns:
        `edge_list` with swaps applied

    Raises:
        ValueError: if edge_list is incorrect shape.
    """
    if not (isinstance(edge_list, np.ndarray) and edge_list.ndim == 2 and edge_list.shape[1] == 2):
        raise ValueError("edge_list must be a 2D numpy array of shape (num_edges, 2)")
    num_edges = edge_list.shape[0]  # TODO change to 1 if batching

    rng = np.random.default_rng(seed)

    order_of_magnitude = n**2
    min_swaps = max(1, int(0.5 * order_of_magnitude))
    max_swaps = int(2 * order_of_magnitude)
    num_swaps = rng.integers(min_swaps, max_swaps + 1)

    successful_swaps = 0
    max_attempts = num_swaps * 10  # Prevent infinite loops

    for _ in range(max_attempts):
        if successful_swaps >= num_swaps:
            break

        i_arr: np.ndarray = rng.choice(num_edges, size=2, replace=False)
        i: int = int(i_arr[0])
        j: int = int(i_arr[1])
        new_edge_1: np.ndarray = np.array([edge_list[i, 0], edge_list[j, 0]])
        new_edge_2: np.ndarray = np.array([edge_list[i, 1], edge_list[j, 1]])

        # Skip if either new edge would be a self-loop
        if new_edge_1[0] == new_edge_1[1] or new_edge_2[0] == new_edge_2[1]:
            continue

        # Check if edges already exist (check both orientations for undirected graph)
        edge_1_exists = np.any(np.all(edge_list == new_edge_1, axis=1)) or np.any(
            np.all(edge_list == new_edge_1[::-1], axis=1)
        )

        edge_2_exists = np.any(np.all(edge_list == new_edge_2, axis=1)) or np.any(
            np.all(edge_list == new_edge_2[::-1], axis=1)
        )

        if edge_1_exists or edge_2_exists:
            continue  # Skip this swap

        # Perform the edge swap
        edge_list[i] = new_edge_1
        edge_list[j] = new_edge_2
        successful_swaps += 1

    return edge_list


def gen_random_regular_graph(
    n: int, k: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a k-regular random graph by generating a circulant graph and swapping edges.

    Args:
        n: Number of nodes in the graph
        k: Degree of each node (must satisfy n * k is even and n >= k + 1)
        seed: Random seed for reproducibility (optional)

    Returns:
        Adjacency matrix, edge list

    Raises:
        ValueError: If k-regularity is infeasible for the given parameter set
    """
    if not (n * k % 2 == 0 and n >= k + 1):
        raise ValueError("n * k must be even and n >= k + 1.")

    offsets = []
    if k % 2 == 0:
        r = k // 2
        offsets = list(range(1, r + 1))
    else:
        m = n // 2  # n must be even by the initial check
        r = (k - 1) // 2
        offsets = [*list(range(1, r + 1)), m]

    circulant: nx.Graph = nx.circulant_graph(n, offsets)
    edge_list = np.array(list(circulant.edges()), dtype=np.int32)

    final_edge_list = _swap_edges(edge_list, n, seed)
    A = np.zeros(shape=(n, n), dtype=int)

    # Convert edge list to adjacency matrix (vectorized)
    # Set both directions of each edge
    A[final_edge_list[:, 0], final_edge_list[:, 1]] = 1
    A[final_edge_list[:, 1], final_edge_list[:, 0]] = 1

    return A, final_edge_list
