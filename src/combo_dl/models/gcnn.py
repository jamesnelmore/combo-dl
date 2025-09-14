from typing import override

import torch
from torch import nn
from torch.nn import functional
from torch_geometric import nn as gnn
from torch_geometric.nn import GATConv


class GCNN(nn.Module):
    """Graph Convolutional Neural Network using PyG Sequential model."""

    def __init__(self, channel_sizes: list[int], num_classes: int = 2):
        """Initialize GCNN with specified channel sizes.

        Args:
            channel_sizes: List of feature dimensions for each layer
            num_classes: Number of output classes
        """
        super().__init__()

        # Build PyG graph layers only
        graph_layers = []
        for i in range(len(channel_sizes) - 1):
            graph_layers.append((
                GATConv(channel_sizes[i], channel_sizes[i + 1]),
                "x, edge_index -> x",
            ))
            graph_layers.append((torch.nn.ReLU(), "x -> x"))
            graph_layers.append((torch.nn.Dropout(0.1), "x -> x"))

        # PyG Sequential for graph layers only
        self.graph_model = gnn.Sequential("x, edge_index", graph_layers)

        # Separate PyTorch layer for final classification
        self.classifier = torch.nn.Linear(channel_sizes[-1], num_classes)

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
