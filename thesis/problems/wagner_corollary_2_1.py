import torch
import math
import networkx as nx

from thesis.core.problem import BaseProblem

# TODO figuure out what this vibe coded scoring function does

class WagnerCorollary2_1(BaseProblem):
    def __init__(self, n: int):
        self.n = n
        self.edges = (n**2 - n) // 2
        self.goal_score = math.sqrt(n - 1) + 1
        print(f"Goal score (sqrt({n-1}) + 1): {self.goal_score:.6f}")
        print(f"Searching for graphs with eigenvalue + matching < {self.goal_score:.6f}")
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for each construction in the batch.
        Score = largest eigenvalue + matching number (to be minimized)
        
        Args:
            x: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph
        
        Returns:
            Tensor of shape (batch_size,) with scores for each construction
        """
        batch_size = x.shape[0]
        
        # Batch convert all edge vectors to adjacency matrices
        adj_matrices = self._batch_edge_vector_to_adjacency(x)
        
        # Batch compute eigenvalues (CPU operation)
        largest_eigenvalues = self._batch_compute_largest_eigenvalue(adj_matrices)
        
        # Batch compute matching numbers using NetworkX
        matching_numbers = self._batch_compute_maximum_matching(adj_matrices)
        
        # Combine scores
        scores = largest_eigenvalues + matching_numbers
        return scores
    
    def _edge_vector_to_adjacency(self, edge_vector: torch.Tensor) -> torch.Tensor:
        """
        Convert edge vector to adjacency matrix using vectorized operations.
        
        Args:
            edge_vector: Tensor of shape (edges,) with 0s and 1s
            
        Returns:
            Adjacency matrix of shape (n, n)
        """
        n = self.n
        device = edge_vector.device
        dtype = edge_vector.dtype
        
        # Create adjacency matrix
        adj_matrix = torch.zeros((n, n), device=device, dtype=dtype)
        
        # Pre-compute indices for upper triangle (excluding diagonal)
        # This creates coordinate pairs (i, j) for i < j
        if not hasattr(self, '_upper_triangle_indices'):
            # Cache the indices for reuse - create on CPU first if MPS device
            if device.type == 'mps':
                triu_indices = torch.triu_indices(n, n, offset=1, device='cpu').to(device)
            else:
                triu_indices = torch.triu_indices(n, n, offset=1, device=device)
            self._upper_triangle_indices = triu_indices
        else:
            triu_indices = self._upper_triangle_indices
            # Move to correct device if needed
            if triu_indices.device != device:
                if device.type == 'mps':
                    # Create on CPU first then move to MPS
                    triu_indices = torch.triu_indices(n, n, offset=1, device='cpu').to(device)
                else:
                    triu_indices = triu_indices.to(device)
                self._upper_triangle_indices = triu_indices
        
        # Vectorized assignment to upper triangle
        adj_matrix[triu_indices[0], triu_indices[1]] = edge_vector
        
        # Make symmetric by copying upper triangle to lower triangle
        adj_matrix = adj_matrix + adj_matrix.t()
        
        return adj_matrix
    
    def _compute_largest_eigenvalue(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the largest eigenvalue of the adjacency matrix.
        
        Args:
            adj_matrix: Symmetric adjacency matrix of shape (n, n)
            
        Returns:
            Largest eigenvalue as a scalar tensor
        """
        # Convert to float for eigenvalue computation
        adj_float = adj_matrix.float()
        original_device = adj_float.device
        
        # Move to CPU if on MPS device (eigenvals not supported on MPS)
        if adj_float.device.type == 'mps':
            adj_float = adj_float.cpu()
        
        # Compute eigenvalues (real since matrix is symmetric)
        eigenvalues = torch.linalg.eigvals(adj_float).real
        
        # Return the largest eigenvalue, moved back to original device
        largest_eigenvalue = torch.max(eigenvalues)
        return largest_eigenvalue.to(original_device)
    
    def _compute_maximum_matching(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the exact maximum matching number using NetworkX.
        
        Args:
            adj_matrix: Adjacency matrix of shape (n, n)
            
        Returns:
            Exact maximum matching number as a scalar tensor
        """
        n = self.n
        
        # Convert adjacency matrix to NetworkX graph
        # First convert to numpy for NetworkX compatibility
        adj_np = adj_matrix.detach().cpu().numpy()
        
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adj_np)
        
        # Compute maximum matching using NetworkX's exact algorithm
        # This uses Edmonds' blossom algorithm for general graphs
        matching = nx.max_weight_matching(G, maxcardinality=True)
        
        # Return the size of the matching
        matching_size = len(matching)
        
        return torch.tensor(matching_size, dtype=torch.float32, device=adj_matrix.device)
    
    def _batch_edge_vector_to_adjacency(self, edge_vectors: torch.Tensor) -> torch.Tensor:
        """
        Batch convert edge vectors to adjacency matrices.
        
        Args:
            edge_vectors: Tensor of shape (batch_size, edges) with 0s and 1s
            
        Returns:
            Batch of adjacency matrices of shape (batch_size, n, n)
        """
        batch_size = edge_vectors.shape[0]
        n = self.n
        device = edge_vectors.device
        dtype = edge_vectors.dtype
        
        # Create batch of adjacency matrices
        adj_matrices = torch.zeros((batch_size, n, n), device=device, dtype=dtype)
        
        # Use cached indices or create them
        if not hasattr(self, '_upper_triangle_indices'):
            if device.type == 'mps':
                triu_indices = torch.triu_indices(n, n, offset=1, device='cpu').to(device)
            else:
                triu_indices = torch.triu_indices(n, n, offset=1, device=device)
            self._upper_triangle_indices = triu_indices
        else:
            triu_indices = self._upper_triangle_indices
            if triu_indices.device != device:
                if device.type == 'mps':
                    triu_indices = torch.triu_indices(n, n, offset=1, device='cpu').to(device)
                else:
                    triu_indices = triu_indices.to(device)
                self._upper_triangle_indices = triu_indices
        
        # Vectorized assignment to upper triangle for all matrices
        adj_matrices[:, triu_indices[0], triu_indices[1]] = edge_vectors
        
        # Make symmetric by copying upper triangle to lower triangle
        adj_matrices = adj_matrices + adj_matrices.transpose(-1, -2)
        
        return adj_matrices
    
    def _batch_compute_largest_eigenvalue(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """
        Batch compute the largest eigenvalue for multiple adjacency matrices.
        
        Args:
            adj_matrices: Batch of symmetric adjacency matrices of shape (batch_size, n, n)
            
        Returns:
            Largest eigenvalues as tensor of shape (batch_size,)
        """
        batch_size = adj_matrices.shape[0]
        original_device = adj_matrices.device
        
        # Convert to float and move to CPU for eigenvalue computation
        adj_float = adj_matrices.float()
        if adj_float.device.type == 'mps':
            adj_float = adj_float.cpu()
        
        # Batch compute eigenvalues
        eigenvalues = torch.linalg.eigvals(adj_float).real  # Shape: (batch_size, n)
        
        # Get the largest eigenvalue for each matrix
        largest_eigenvalues = torch.max(eigenvalues, dim=1)[0]  # Shape: (batch_size,)
        
        return largest_eigenvalues.to(original_device)
    
    def _batch_compute_maximum_matching(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """
        Batch compute maximum matching numbers using NetworkX.
        
        Args:
            adj_matrices: Batch of adjacency matrices of shape (batch_size, n, n)
            
        Returns:
            Maximum matching numbers as tensor of shape (batch_size,)
        """
        batch_size = adj_matrices.shape[0]
        device = adj_matrices.device
        matching_numbers = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Convert to numpy for NetworkX
        adj_np = adj_matrices.detach().cpu().numpy()
        
        # Process each graph (this is still sequential but we batch the data prep)
        for i in range(batch_size):
            # Create graph from adjacency matrix
            G = nx.from_numpy_array(adj_np[i])
            
            # Compute maximum matching
            matching = nx.max_weight_matching(G, maxcardinality=True)
            matching_numbers[i] = len(matching)
        
        return matching_numbers