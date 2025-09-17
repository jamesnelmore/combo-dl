from typing import override

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from torch.nn import functional
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GCNN(nn.Module):
    """Graph Convolutional Neural Network using PyG Sequential model."""

    def __init__(
        self, channel_sizes: list[int], num_classes: int = 2, input_dim: int | None = None
    ):
        """Initialize GCNN with specified channel sizes.

        Args:
            channel_sizes: List of feature dimensions for each layer
            num_classes: Number of output classes
            input_dim: Input feature dimension. If None, uses first element of channel_sizes
        """
        super().__init__()

        # Use input_dim if provided, otherwise use first channel size
        if input_dim is not None:
            actual_channel_sizes = [input_dim, *channel_sizes]
        else:
            actual_channel_sizes = channel_sizes

        # Build PyG graph layers only
        graph_layers = []
        for i in range(len(actual_channel_sizes) - 1):
            graph_layers.append((
                GATConv(actual_channel_sizes[i], actual_channel_sizes[i + 1]),
                "x, edge_index -> x",
            ))
            graph_layers.append((torch.nn.ReLU(), "x -> x"))
            graph_layers.append((torch.nn.Dropout(0.1), "x -> x"))

        # PyG Sequential for graph layers only
        self.graph_model = gnn.Sequential("x, edge_index", graph_layers)

        # Separate PyTorch layer for final classification
        self.classifier = torch.nn.Linear(actual_channel_sizes[-1], num_classes)

    @override
    def forward(self, data) -> torch.Tensor:
        """Forward pass through the GCNN.

        Args:
            data: PyG Data object containing x (node features) and edge_index

        Returns:
            Log probabilities for each node
        """
        # Pass through PyG graph layers
        x = self.graph_model(data.x, data.edge_index)

        # Pass through PyTorch linear layer
        x = self.classifier(x)

        return functional.log_softmax(x, dim=1)


class GCNNFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor that converts environment observations to PyG Data."""

    def __init__(
        self, observation_space, features_dim: int, channel_sizes: list[int], num_classes: int = 2
    ):
        """Initialize the GCNN feature extractor.

        Args:
            observation_space: The observation space from the environment
            features_dim: The dimension of the output features
            channel_sizes: List of feature dimensions for each GCNN layer
            num_classes: Number of output classes for the GCNN
        """
        super().__init__(observation_space, features_dim)
        # Input dimension is 1 since we convert node indices to 1D features
        self.network = GCNN(channel_sizes, num_classes, input_dim=1)

    @override
    def forward(self, observations: dict) -> torch.Tensor:
        """Convert environment observations to PyG Data and pass through GCNN.

        Args:
            observations: Dictionary containing 'edge_list' and 'node_features' TODO add better type hints to dict

        Returns:
            Graph features extracted by the GCNN
        """
        # Extract components from observation dictionary
        edge_list = observations["edge_list"]  # Shape: (num_edges, 2)
        node_features = observations["node_features"]  # Shape: (num_nodes,)

        # Convert edge_list to edge_index format (PyG expects shape [2, num_edges])
        edge_index = (
            edge_list.t().contiguous()
        )  # TODO just use [2, num_edges] in the observation space

        # Convert node_features to proper format for PyG
        # Add feature dimension if needed (node_features might be 1D indices) TODO features should be real valued eventually and so not one hot encoded
        if node_features.dim() == 1:
            # One-hot encode node indices or use as features directly
            x = node_features.unsqueeze(1).float()  # Shape: (num_nodes, 1)
        else:
            x = node_features.float()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        # Pass through GCNN
        return self.network(data)
