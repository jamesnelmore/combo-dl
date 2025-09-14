"""Tests for the GCNN model."""

import pytest
import torch
from torch_geometric.data import Data

from combo_dl.models.gcnn import GCNN


class TestGCNN:
    """Test class for GCNN model functionality."""

    def test_gcnn_initialization(self):
        """Test that GCNN initializes correctly with given parameters."""
        channel_sizes = [16, 32, 64, 32]
        num_classes = 2

        model = GCNN(channel_sizes=channel_sizes, num_classes=num_classes)

        assert isinstance(model, GCNN)
        assert hasattr(model, "graph_model")
        assert hasattr(model, "classifier")
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_gcnn_forward_pass(self):
        """Test forward pass through GCNN with random data."""
        # Create a simple test graph
        num_nodes = 10
        num_features = 16

        # Create random node features
        x = torch.randn(num_nodes, num_features)

        # Create a simple graph (fully connected)
        edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
        # Make it undirected by adding reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        # Create and test the model
        channel_sizes = [16, 32, 64, 32]
        num_classes = 2

        model = GCNN(channel_sizes=channel_sizes, num_classes=num_classes)
        model.eval()

        with torch.no_grad():
            output = model(data)

        # Assertions
        assert output.shape == (num_nodes, num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize(
        "channel_sizes",
        [
            [16, 32],
            [8, 16, 32],
            [4, 8, 16, 32, 16],
        ],
    )
    def test_gcnn_with_different_channel_sizes(self, channel_sizes):
        """Test GCNN with various channel size configurations."""
        model = GCNN(channel_sizes=channel_sizes, num_classes=3)

        # Create minimal test data
        num_nodes = 3
        x = torch.randn(num_nodes, channel_sizes[0])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            output = model(data)

        assert output.shape == (num_nodes, 3)

    def test_gcnn_model_parameters_count(self):
        """Test that the model has a reasonable number of parameters."""
        channel_sizes = [16, 32, 64, 32]
        num_classes = 2

        model = GCNN(channel_sizes=channel_sizes, num_classes=num_classes)
        param_count = sum(p.numel() for p in model.parameters())

        # Should have parameters from both graph layers and classifier
        assert param_count > 0
        # Reasonable upper bound for this architecture
        assert param_count < 100000
