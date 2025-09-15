"""Tests for the graph_tools module."""

from pathlib import Path
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from combo_dl.graph_tools.regular_random_graph import generate_random_regular_graph


def assert_k_regular(adjacency_matrix: np.ndarray, k: int) -> None:
    """Assert that an adjacency matrix represents a k-regular graph.

    Args:
        adjacency_matrix: The adjacency matrix to check
        k: The expected degree for each node
    """
    row_sums = adjacency_matrix.sum(axis=1)
    assert np.all(row_sums == k), f"Graph is not {k}-regular. Row sums: {row_sums}"


def assert_symmetric(adjacency_matrix: np.ndarray) -> None:
    """Assert that an adjacency matrix is symmetric (for undirected graphs).

    Args:
        adjacency_matrix: The adjacency matrix to check
    """
    assert np.array_equal(adjacency_matrix, adjacency_matrix.T), (
        "Adjacency matrix is not symmetric"
    )


class TestRegularRandomGraph:
    """Test cases for the generate_random_regular_graph function."""

    def test_small_graph_generation(self) -> None:
        """Test generation of a small regular graph."""
        mat = generate_random_regular_graph(5, 4)

        # Check basic properties
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (5, 5)
        assert mat.dtype == int

        assert_symmetric(mat)

        # Check that diagonal is zero (no self-loops)
        assert np.all(np.diag(mat) == 0)

        # Check that all values are 0 or 1
        assert np.all((mat == 0) | (mat == 1))

    @pytest.mark.parametrize(
        "n, k",
        [
            (100, 30),
            (101, 100),
        ],
    )
    def test_large_graph_regularity(self, n: int, k: int) -> None:
        """Test that large graphs maintain k-regularity for various n and k."""
        mat = generate_random_regular_graph(n, k)

        assert_k_regular(mat, k)

        assert_symmetric(mat)

        # Check that diagonal is zero
        assert np.all(np.diag(mat) == 0)

    def test_even_degree_graph(self) -> None:
        """Test generation of graphs with even degree."""
        mat = generate_random_regular_graph(6, 4)

        assert_k_regular(mat, 4)

        # Check symmetry
        assert_symmetric(mat)

    def test_odd_degree_graph(self) -> None:
        """Test generation of graphs with odd degree."""
        mat = generate_random_regular_graph(8, 3)  # n must be even for odd k

        assert_k_regular(mat, 3)

        assert_symmetric(mat)

    def test_minimal_valid_graph(self) -> None:
        """Test the minimal valid parameters."""
        mat = generate_random_regular_graph(3, 2)

        assert mat.shape == (3, 3)
        assert_k_regular(mat, 2)
        assert_symmetric(mat)

    def test_invalid_parameters_odd_nk(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="n \\* k must be even"):
            generate_random_regular_graph(3, 3)  # n*k = 9 (odd)

    def test_invalid_parameters_n_too_small(self) -> None:
        """Test that n < k+1 raises ValueError."""
        with pytest.raises(ValueError, match="n >= k \\+ 1"):
            generate_random_regular_graph(3, 4)  # n=3, k=4, but 3 < 4+1

    def test_invalid_parameters_odd_n_odd_k(self) -> None:
        """Test that odd n with odd k raises ValueError."""
        with pytest.raises(ValueError, match="n \\* k must be even"):
            generate_random_regular_graph(5, 3)  # n=5 (odd), k=3 (odd), n*k=15 (odd)

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            (10, 4, 42),
            (8, 3, 123),
            (12, 6, 7),
            (6, 2, 99),
            (14, 5, 2024),
        ],
    )
    def test_deterministic_with_seed(self, n: int, k: int, seed: int) -> None:
        """Test that the same seed produces the same graph for various parameters."""
        mat1 = generate_random_regular_graph(n, k, seed=seed)
        mat2 = generate_random_regular_graph(n, k, seed=seed)

        assert np.array_equal(mat1, mat2)

    def test_different_seeds_produce_different_graphs(self) -> None:
        """Test that different seeds can produce different graphs."""
        mat1 = generate_random_regular_graph(10, 4, seed=42)
        mat2 = generate_random_regular_graph(10, 4, seed=123)

        # Note: This test might occasionally fail if both seeds happen to produce
        # the same graph, but it's very unlikely for larger graphs
        assert not np.array_equal(mat1, mat2)
        assert_symmetric(mat1)
        assert_symmetric(mat2)

    def test_adjacency_matrix_properties(self) -> None:
        """Test that the adjacency matrix has correct properties."""
        mat = generate_random_regular_graph(8, 3)

        # Check that it's a valid adjacency matrix
        assert np.all((mat == 0) | (mat == 1))  # Binary values only
        assert np.all(np.diag(mat) == 0)  # No self-loops
        assert_symmetric(mat)

        assert_k_regular(mat, 3)
        col_sums = mat.sum(axis=0)
        assert np.all(col_sums == 3)
        assert np.array_equal(mat.sum(axis=1), col_sums)
