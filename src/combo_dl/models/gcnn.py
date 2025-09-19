from typing import TypedDict, override

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GATConv


class GraphObservation(TypedDict):
    node_features: torch.Tensor
    edge_list: torch.Tensor  # TODO or should it be numpy???


class GCNN(nn.Module):
    """Graph Convolutional Neural Network using PyG Sequential model."""

    output_dim: int

    def __init__(self, input_dim: int, channel_sizes: list[int]):
        """Initialize GCNN with specified input dimension and channel sizes.

        Args:
            input_dim: Input feature dimension.
            channel_sizes: List of feature dimensions for each layer.
        """
        super().__init__()

        actual_channel_sizes = [input_dim, *channel_sizes]
        self.output_dim = actual_channel_sizes[-1]

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
        self.graph_convolutions = gnn.Sequential("x, edge_index", graph_layers)

    @override
    def forward(self, data: PyGData) -> torch.Tensor:
        """Forward pass through the GCNN.

        Args:
            data: PyG Data object containing x (node features) and edge_index

        Returns:
            Extracted node feature
        """
        # Pass through PyG graph layers
        node_embeddings = self.graph_convolutions(data.x, data.edge_index)

        # TODO we'll replace this part with attention heads when trying graph attention layers
        # Takes the elementwise mean of node features across all nodes
        graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)

        return graph_embedding


class GCNNFeatureExtractor(BaseFeaturesExtractor):
    """Uses a GCNN to extract graph features in stable-baselines."""

    def __init__(self, observation_space: gym.spaces.Dict, gcnn: GCNN, features_dim: int):
        """Create Graph Convolutional feature extractor.

        Args:
            observation_space: Gymnasium observation space the model will be trained in
            gcnn: Graph convolutional model the feature extractor is based on
            features_dim: length of output feature vector. Must equal sb3 expected feature length
        """
        super().__init__(observation_space, features_dim)
        # TODO size checks that observation_space will line up with model and features_dim
        self.graph_model = gcnn

        self.feature_projection = nn.Linear(gcnn.output_dim, features_dim)

    @override
    # TODO dict might need to have an nparray?
    def forward(self, observations: GraphObservation) -> torch.Tensor:
        node_features = observations["node_features"]
        edge_list = observations["edge_list"]

        graph = PyGData(x=node_features, edge_index=edge_list)

        raw_logits = self.graph_model(graph)
        return self.feature_projection(raw_logits)
