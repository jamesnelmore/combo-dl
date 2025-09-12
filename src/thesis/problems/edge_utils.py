"""Utilities for converting between edge vectors and adjacency matrices.

This module provides canonical mappings used throughout the project for converting
between different graph representations.
"""

import torch


def edge_vector_to_adjacency_matrix(edge_vector: torch.Tensor, n: int) -> torch.Tensor:
    """Convert edge vector to symmetric adjacency matrix using canonical mapping.

    This function provides the standard conversion used throughout the project.
    The edge vector represents the upper triangular part of the adjacency matrix
    (excluding the diagonal) in row-major order.

    Args:
        edge_vector: Tensor of shape (batch_size, edges) where edges = n*(n-1)/2
                    Each element should be 0 or 1 representing edge presence
        n: Number of vertices in the graph

    Returns:
        Tensor of shape (batch_size, n, n) representing symmetric adjacency matrices
        with zeros on the diagonal

    Example:
        For n=4, the edge vector [e01, e02, e03, e12, e13, e23] maps to:
        [[0,   e01, e02, e03],
         [e01, 0,   e12, e13],
         [e02, e12, 0,   e23],
         [e03, e13, e23, 0  ]]
    """
    batch_size = edge_vector.shape[0]
    edges = (n * (n - 1)) // 2
    device = edge_vector.device
    dtype = edge_vector.dtype

    # Validate input dimensions
    assert edge_vector.shape[1] == edges, (
        f"Expected edge vector with {edges} elements for n={n}, got {edge_vector.shape[1]}"
    )

    # Create empty adjacency matrices
    adj_matrices = torch.zeros(batch_size, n, n, device=device, dtype=dtype)

    # Get upper triangular indices (excluding diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)

    # Fill upper triangular part
    adj_matrices[:, triu_indices[0], triu_indices[1]] = edge_vector

    # Make symmetric by copying upper triangular to lower triangular
    adj_matrices = adj_matrices + adj_matrices.transpose(-1, -2)

    return adj_matrices


def adjacency_matrix_to_edge_vector(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Convert symmetric adjacency matrix to edge vector using canonical mapping.

    This function provides the inverse of edge_vector_to_adjacency_matrix.
    It extracts the upper triangular part of the adjacency matrix (excluding diagonal)
    in row-major order.

    Args:
        adj_matrix: Tensor of shape (batch_size, n, n) representing symmetric adjacency matrices

    Returns:
        Tensor of shape (batch_size, edges) where edges = n*(n-1)/2

    Example:
        For the adjacency matrix:
        [[0,   e01, e02, e03],
         [e01, 0,   e12, e13],
         [e02, e12, 0,   e23],
         [e03, e13, e23, 0  ]]
        Returns edge vector [e01, e02, e03, e12, e13, e23]
    """
    batch_size, n, _ = adj_matrix.shape
    device = adj_matrix.device

    # Validate input is square
    assert adj_matrix.shape[1] == adj_matrix.shape[2], (
        f"Expected square adjacency matrix, got shape {adj_matrix.shape}"
    )

    # Get upper triangular indices (excluding diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)

    # Extract upper triangular part
    edge_vector = adj_matrix[:, triu_indices[0], triu_indices[1]]

    return edge_vector


def get_edge_count(n: int) -> int:
    """Get the number of edges in the upper triangular part of an nxn adjacency matrix.

    Args:
        n: Number of vertices

    Returns:
        Number of edges = n*(n-1)/2
    """
    return (n * (n - 1)) // 2
