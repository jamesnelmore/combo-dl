"""Tests for FFModel."""

import pytest
import torch

from thesis.models.ff_model import FFModel


class TestFFModel:
    """Test cases for FFModel."""

    def test_model_initialization(self) -> None:
        """Test that FFModel initializes correctly."""
        n = 4
        model = FFModel(n=n)

        assert model.n == n
        assert model.edges == (n**2 - n) // 2  # Should be 6 for n=4
        assert isinstance(model.layers, torch.nn.Sequential)

    def test_model_initialization_with_custom_layers(self) -> None:
        """Test FFModel initialization with custom hidden layer sizes."""
        n = 5
        hidden_layers = [64, 32]
        model = FFModel(n=n, hidden_layer_sizes=hidden_layers)

        assert model.n == n
        assert model.edges == (n**2 - n) // 2  # Should be 10 for n=5

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        n = 4
        batch_size = 3
        model = FFModel(n=n)

        device = next(model.parameters()).device
        x = torch.zeros((batch_size, model.edges), device=device)
        i = torch.tensor([0, 1, 2], device=device)

        output = model.forward(x, i)

        assert output.shape == (batch_size, 2)
        assert output.dtype == x.dtype

    def test_forward_pass_single_sample(self) -> None:
        """Test forward pass with single sample (1D tensor)."""
        n = 4
        model = FFModel(n=n)

        device = next(model.parameters()).device
        x = torch.zeros(model.edges, device=device).unsqueeze(0)  # Add batch dimension
        i = torch.tensor(0, device=device)

        output = model.forward(x, i)

        assert output.shape == (1, 2)

    def test_sampling_shape_and_values(self) -> None:
        """Test that sampling produces correct shape and binary values."""
        n = 4
        batch_size = 5
        model = FFModel(n=n)

        samples = model.sample(batch_size=batch_size)

        assert samples.shape == (batch_size, model.edges)
        assert torch.all((samples == 0) | (samples == 1)), "Samples should be binary (0 or 1)"
        assert samples.dtype in [torch.float32, torch.long, torch.int64]

    def test_sampling_preserves_training_mode(self) -> None:
        """Test that sampling preserves the model's training mode."""
        n = 4
        model = FFModel(n=n)

        # Test in training mode
        model.train()
        assert model.training
        _ = model.sample(batch_size=2)
        assert model.training

        # Test in eval mode
        model.eval()
        assert not model.training
        _ = model.sample(batch_size=2)
        assert not model.training

    def test_different_graph_sizes(self) -> None:
        """Test model works with different graph sizes."""
        for n in [3, 5, 6, 10]:
            model = FFModel(n=n)
            expected_edges = (n**2 - n) // 2

            assert model.edges == expected_edges

            # Test sampling works
            samples = model.sample(batch_size=2)
            assert samples.shape == (2, expected_edges)

    @pytest.mark.parametrize(
        "n,expected_edges",
        [
            (3, 3),  # (9-3)/2 = 3
            (4, 6),  # (16-4)/2 = 6
            (5, 10),  # (25-5)/2 = 10
            (6, 15),  # (36-6)/2 = 15
        ],
    )
    def test_edge_calculation(self, n: int, expected_edges: int) -> None:
        """Test edge calculation for different graph sizes."""
        model = FFModel(n=n)
        assert model.edges == expected_edges

    def test_model_device_consistency(self) -> None:
        """Test that model handles device consistency correctly."""
        n = 4
        model = FFModel(n=n)
        device = next(model.parameters()).device

        # Test forward pass device handling
        x = torch.zeros((2, model.edges), device=device)
        i = torch.tensor([0, 1], device=device)
        output = model.forward(x, i)

        assert output.device == device

        # Test sampling device handling
        samples = model.sample(batch_size=3)
        assert samples.device == device
