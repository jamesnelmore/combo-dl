"""Utilities for graph computations including eigenvalues and matching numbers.

This module provides graph-theoretic computations used throughout the project,
including eigenvalue calculations and maximum matching computations.
"""

import multiprocessing as mp

import networkx as nx
import torch


def compute_largest_eigenvalue(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the largest eigenvalue of the adjacency matrix.

    Args:
        adj_matrix: Adjacency matrices of shape (batch_size, n, n)

    Returns:
        Largest eigenvalue as tensor of shape (batch_size,)
    """
    # TODO: Find faster binary eigenvalue algorithm, such as power iteration.
    #  Current implementation converts to float and moves to CPU, which is inefficient.
    original_device = adj_matrix.device

    # Convert to float and move to CPU (eigenval computation not supported on MPS)
    adj_float = adj_matrix.float()
    if adj_float.device.type == "mps":
        adj_float = adj_float.cpu()

    # Batch compute eigenvalues
    eigenvalues = torch.linalg.eigvals(adj_float).real  # Shape: (batch_size, n)

    # Get the largest eigenvalue for each matrix
    largest_eigenvalue = torch.max(eigenvalues, dim=1)[0]  # Shape: (batch_size,)

    return largest_eigenvalue.to(original_device)


def _compute_matching_for_adjacency(adj_matrix: torch.Tensor) -> int:
    """Helper function to compute matching number for a single adjacency matrix.

    This function is designed to be used with multiprocessing.

    Args:
        adj_matrix: Single adjacency matrix of shape (n, n)

    Returns:
        Maximum matching number
    """
    G = nx.from_numpy_array(adj_matrix.numpy())
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)


def compute_maximum_matching(adj_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the exact maximum matching number using NetworkX with multiprocessing.

    Args:
        adj_matrix: Batch of adjacency matrices of shape (batch_size, n, n)

    Returns:
        Maximum matching number as tensor of shape (batch_size,)
    """
    batch_size = adj_matrix.shape[0]
    device = adj_matrix.device

    # For small batches, use sequential processing to avoid multiprocessing overhead
    if batch_size <= 2:
        adj_cpu = adj_matrix.detach().cpu()
        matching_numbers = []

        for i in range(batch_size):
            matching_num = _compute_matching_for_adjacency(adj_cpu[i])
            matching_numbers.append(matching_num)

        matching_number = torch.tensor(matching_numbers, dtype=torch.int32, device=device)
        return matching_number

    # For larger batches, use multiprocessing
    adj_cpu = adj_matrix.detach().cpu()

    # Determine number of processes (use min of CPU count and batch size)
    num_processes = min(mp.cpu_count(), batch_size)

    # Use multiprocessing to compute matching numbers in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Convert each adjacency matrix to a separate tensor for multiprocessing
        adj_matrices = [adj_cpu[i] for i in range(batch_size)]
        matching_numbers = pool.map(_compute_matching_for_adjacency, adj_matrices)

    # Convert results back to tensor on the original device
    matching_number = torch.tensor(matching_numbers, dtype=torch.int32, device=device)

    return matching_number
