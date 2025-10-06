"""Model used by Wagner in DCE."""

from collections import OrderedDict
from typing import override

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .protocols import SamplingModel


class FFModel(SamplingModel):
    """Feed forward network.

    Regularized with dropout and layer normalization
    """

    def __init__(
        self,
        n: int,
        hidden_layer_sizes: list[int] | None = None,
        output_size: int = 2,
        dropout_probability: float = 0.1,
    ):
        """Create FFModel object.

        Args:
            n: number of vertices in the graph.
            hidden_layer_sizes: List of hidden layer sizes.
            output_size: Size of final layer
            dropout_probability: probability of a dropout at each layer during training
        """
        super().__init__()
        self.n = n
        self.edges = (n**2 - n) // 2
        activation = nn.ReLU

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [128, 64, 64]  # From Wagner 2021

        model_layers = []
        model_layers.append(("input_layer", nn.Linear(2 * self.edges, hidden_layer_sizes[0])))
        model_layers.append(("layernorm_input", nn.LayerNorm(hidden_layer_sizes[0])))
        model_layers.append(("activation_input", activation()))
        model_layers.append(("dropout_input", nn.Dropout(dropout_probability)))

        prev_layer_size = hidden_layer_sizes[0]
        for i, layer_size in enumerate(hidden_layer_sizes[1:]):
            model_layers.append((f"hidden_{i}", nn.Linear(prev_layer_size, layer_size)))
            model_layers.append((f"layernorm_{i}", nn.LayerNorm(layer_size)))
            model_layers.append((f"activation_{i}", activation()))
            model_layers.append((f"dropout_{i}", nn.Dropout(dropout_probability)))
            prev_layer_size = layer_size

        model_layers.append(("output", nn.Linear(prev_layer_size, output_size)))

        self.layers = nn.Sequential(OrderedDict(model_layers))

    @override
    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        one_hot_position = F.one_hot(i.to(device=device), num_classes=self.edges).to(dtype=x.dtype)

        if one_hot_position.dim() == 1:
            one_hot_position = one_hot_position.unsqueeze(0)

        input_tensor = torch.cat((x, one_hot_position), dim=-1)
        return self.layers(input_tensor)

    @override
    def sample(self, batch_size: int, sample_length: int | None = None) -> torch.Tensor:
        sample_length = sample_length or self.edges
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                w = torch.zeros((batch_size, sample_length), device=device)
                for i in range(sample_length):
                    i_tensor = torch.full((batch_size,), i, dtype=torch.long, device=device)
                    x = self.forward(w, i_tensor)
                    assert x.shape == (batch_size, 2)
                    probs = F.softmax(x, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    w[:, i] = sampled
        finally:
            if was_training:
                self.train()
        return w


class PaddedFFModel(FFModel):
    def __init__(
        self,
        n: int,
        hidden_layer_sizes: list[int] | None = None,
        output_size: int = 2,
        dropout_probability: float = 0.1,
    ):
        super().__init__(n, hidden_layer_sizes, output_size, dropout_probability)

    @override
    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        input_length = x.shape[1]

        if input_length > self.edges:
            raise ValueError("PaddedFFModel can only accept graphs with up to n vertices.")
        if input_length == self.edges:
            padded_x = x
        else:
            padded_x = F.pad(x, (0, self.edges - input_length), mode="constant", value=0)

        x = super().forward(padded_x, i)
        return x[:, :input_length]
