"""Tests for Wagner Corollary 2.1 maximum matching function."""

import networkx as nx
import torch

from thesis.problems.canonical_edge_mapping import edge_vector_to_adjacency_matrix
from thesis.problems.graph_utils import compute_maximum_matching
from thesis.problems.wagner_corollary_2_1 import WagnerCorollary21


class TestMaximumMatching:
    """Test cases for the maximum matching computation function."""

    def test_empty_graph(self) -> None:
        """Test maximum matching for empty graph (no edges)."""
        n = 4
        # Empty graph: all zeros
        adj_matrix = torch.zeros(1, n, n)
        matching_number = compute_maximum_matching(adj_matrix)

        assert matching_number.shape == (1,)
        assert matching_number[0] == 0

    def test_complete_graph(self) -> None:
        """Test maximum matching for complete graph."""
        n = 4
        # Complete graph: all ones except diagonal
        adj_matrix = torch.ones(1, n, n)
        adj_matrix[0].fill_diagonal_(0)
        matching_number = compute_maximum_matching(adj_matrix)

        # For complete graph K_n, maximum matching is floor(n/2)
        expected = n // 2
        assert matching_number[0] == expected

    def test_path_graph(self) -> None:
        """Test maximum matching for path graph (linear chain)."""
        n = 5
        # Path graph: 0-1-2-3-4
        adj_matrix = torch.zeros(1, n, n)
        for i in range(n - 1):
            adj_matrix[0, i, i + 1] = 1
            adj_matrix[0, i + 1, i] = 1

        matching_number = compute_maximum_matching(adj_matrix)

        # For path P_n, maximum matching is floor(n/2)
        expected = n // 2
        assert matching_number[0] == expected

    def test_cycle_graph(self) -> None:
        """Test maximum matching for cycle graph."""
        n = 6
        # Cycle graph: 0-1-2-3-4-5-0
        adj_matrix = torch.zeros(1, n, n)
        for i in range(n):
            adj_matrix[0, i, (i + 1) % n] = 1
            adj_matrix[0, (i + 1) % n, i] = 1

        matching_number = compute_maximum_matching(adj_matrix)

        # For cycle C_n, maximum matching is floor(n/2)
        expected = n // 2
        assert matching_number[0] == expected

    def test_star_graph(self) -> None:
        """Test maximum matching for star graph (one central vertex)."""
        n = 5
        # Star graph: center vertex 0 connected to all others
        adj_matrix = torch.zeros(1, n, n)
        for i in range(1, n):
            adj_matrix[0, 0, i] = 1
            adj_matrix[0, i, 0] = 1

        matching_number = compute_maximum_matching(adj_matrix)

        # For star graph, maximum matching is 1 (can only match one edge)
        assert matching_number[0] == 1

    def test_disconnected_components(self) -> None:
        """Test maximum matching for graph with disconnected components."""
        n = 6
        # Two disjoint triangles: 0-1-2 and 3-4-5
        adj_matrix = torch.zeros(1, n, n)

        # First triangle: 0-1-2
        adj_matrix[0, 0, 1] = adj_matrix[0, 1, 0] = 1
        adj_matrix[0, 1, 2] = adj_matrix[0, 2, 1] = 1
        adj_matrix[0, 2, 0] = adj_matrix[0, 0, 2] = 1

        # Second triangle: 3-4-5
        adj_matrix[0, 3, 4] = adj_matrix[0, 4, 3] = 1
        adj_matrix[0, 4, 5] = adj_matrix[0, 5, 4] = 1
        adj_matrix[0, 5, 3] = adj_matrix[0, 3, 5] = 1

        matching_number = compute_maximum_matching(adj_matrix)

        # Each triangle has maximum matching of 1, total is 2
        assert matching_number[0] == 2

    def test_single_vertex(self) -> None:
        """Test maximum matching for single vertex graph."""
        n = 1
        adj_matrix = torch.zeros(1, n, n)
        matching_number = compute_maximum_matching(adj_matrix)

        # Single vertex has no edges, so matching is 0
        assert matching_number[0] == 0

    def test_two_vertices_no_edge(self) -> None:
        """Test maximum matching for two disconnected vertices."""
        n = 2
        adj_matrix = torch.zeros(1, n, n)
        matching_number = compute_maximum_matching(adj_matrix)

        # No edges, so matching is 0
        assert matching_number[0] == 0

    def test_two_vertices_with_edge(self) -> None:
        """Test maximum matching for two connected vertices."""
        n = 2
        adj_matrix = torch.zeros(1, n, n)
        adj_matrix[0, 0, 1] = adj_matrix[0, 1, 0] = 1
        matching_number = compute_maximum_matching(adj_matrix)

        # One edge, so matching is 1
        assert matching_number[0] == 1

    def test_batch_processing(self) -> None:
        """Test maximum matching for batch of different graphs."""
        batch_size = 3
        n = 4

        # Create batch with different graphs
        adj_matrices = torch.zeros(batch_size, n, n)

        # Graph 0: Empty
        # Graph 1: Complete graph
        adj_matrices[1] = torch.ones(n, n)
        adj_matrices[1].fill_diagonal_(0)

        # Graph 2: Path graph
        for i in range(n - 1):
            adj_matrices[2, i, i + 1] = 1
            adj_matrices[2, i + 1, i] = 1

        matching_numbers = compute_maximum_matching(adj_matrices)

        assert matching_numbers.shape == (batch_size,)
        assert matching_numbers[0] == 0  # Empty graph
        assert matching_numbers[1] == 2  # Complete graph K_4
        assert matching_numbers[2] == 2  # Path P_4

    def test_tensor_shapes_and_dtypes(self) -> None:
        """Test that function handles different tensor shapes and dtypes correctly."""
        n = 3
        batch_size = 2

        # Test with different dtypes
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            adj_matrix = torch.zeros(batch_size, n, n, dtype=dtype)
            # Add some edges
            adj_matrix[0, 0, 1] = adj_matrix[0, 1, 0] = 1
            adj_matrix[1, 1, 2] = adj_matrix[1, 2, 1] = 1

            matching_number = compute_maximum_matching(adj_matrix)

            assert matching_number.shape == (batch_size,)
            assert matching_number.dtype == torch.int32

    def test_device_handling(self) -> None:
        """Test that function handles different devices correctly."""
        n = 3
        adj_matrix = torch.zeros(1, n, n)
        adj_matrix[0, 0, 1] = adj_matrix[0, 1, 0] = 1

        # Test on CPU
        cpu_matrix = adj_matrix.cpu()
        cpu_matching = compute_maximum_matching(cpu_matrix)
        assert cpu_matching.device.type == "cpu"

        # Test on MPS if available
        if torch.backends.mps.is_available():
            mps_matrix = adj_matrix.to("mps")
            mps_matching = compute_maximum_matching(mps_matrix)
            assert mps_matching.device.type == "mps"

    def test_networkx_consistency(self) -> None:
        """Test that our implementation matches NetworkX results directly."""
        n = 5
        adj_matrix = torch.zeros(1, n, n)

        # Create a specific graph: 0-1-2-3-4 (path)
        for i in range(n - 1):
            adj_matrix[0, i, i + 1] = 1
            adj_matrix[0, i + 1, i] = 1

        # Our implementation
        our_result = compute_maximum_matching(adj_matrix)

        # Direct NetworkX computation
        G = nx.from_numpy_array(adj_matrix[0].numpy())
        nx_matching = nx.max_weight_matching(G, maxcardinality=True)
        nx_result = len(nx_matching)

        assert our_result[0] == nx_result

    def test_edge_vector_integration(self) -> None:
        """Test maximum matching with edge vector input (integration test)."""
        n = 4
        # Create edge vector for complete graph
        edges = (n * (n - 1)) // 2
        edge_vector = torch.ones(1, edges)  # All edges present

        # Convert to adjacency matrix
        adj_matrix = edge_vector_to_adjacency_matrix(edge_vector, n)

        # Compute maximum matching
        matching_number = compute_maximum_matching(adj_matrix)

        # Complete graph K_4 should have maximum matching of 2
        assert matching_number[0] == 2

    def test_known_theoretical_results(self) -> None:
        """Test against known theoretical results for maximum matching."""
        test_cases = [
            (3, "complete", 1),  # K_3 has matching number 1
            (4, "complete", 2),  # K_4 has matching number 2
            (5, "complete", 2),  # K_5 has matching number 2
            (6, "complete", 3),  # K_6 has matching number 3
            (4, "path", 2),  # P_4 has matching number 2
            (5, "path", 2),  # P_5 has matching number 2
            (6, "path", 3),  # P_6 has matching number 3
        ]

        for n, graph_type, expected in test_cases:
            adj_matrix = torch.zeros(1, n, n)

            if graph_type == "complete":
                adj_matrix[0] = torch.ones(n, n)
                adj_matrix[0].fill_diagonal_(0)
            elif graph_type == "path":
                for i in range(n - 1):
                    adj_matrix[0, i, i + 1] = 1
                    adj_matrix[0, i + 1, i] = 1

            matching_number = compute_maximum_matching(adj_matrix)
            assert matching_number[0] == expected, f"Failed for {graph_type} graph with n={n}"


class TestWagnerCorollary21Integration:
    """Integration tests for Wagner Corollary 2.1 problem class."""

    def test_reward_function_with_known_graphs(self) -> None:
        """Test that the reward function works correctly with known graph types."""
        n = 4
        problem = WagnerCorollary21(n=n)

        # Test with complete graph
        edges = (n * (n - 1)) // 2
        complete_edge_vector = torch.ones(1, edges)
        complete_reward = problem.reward(complete_edge_vector)

        # Test with empty graph
        empty_edge_vector = torch.zeros(1, edges)
        empty_reward = problem.reward(empty_edge_vector)

        # Complete graph should have lower reward (worse score) due to higher eigenvalue + matching
        assert complete_reward[0] < empty_reward[0]

        # Both should be negative (since we negate eigenvalue + matching)
        assert complete_reward[0] < 0
        assert empty_reward[0] <= 0  # Empty graph can have exactly 0

    def test_batch_processing_in_reward(self) -> None:
        """Test that reward function handles batches correctly."""
        n = 3
        problem = WagnerCorollary21(n=n)

        batch_size = 3
        edges = (n * (n - 1)) // 2
        edge_vectors = torch.zeros(batch_size, edges)

        # Different graphs in batch
        edge_vectors[0] = 0  # Empty graph
        edge_vectors[1, :2] = 1  # Partial graph
        edge_vectors[2] = 1  # Complete graph

        rewards = problem.reward(edge_vectors)

        assert rewards.shape == (batch_size,)
        assert torch.all(rewards <= 0)  # All should be negative or zero
