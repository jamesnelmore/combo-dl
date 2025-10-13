from collections import OrderedDict
from typing import Literal, override

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Feed forward network.

    Regularized with dropout and layer normalization
    """

    def __init__(
        self,
        n: int,
        hidden_layer_sizes: list[int] | None = None,
        output_size: int = 2,
        dropout_probability: float = 0.1,
        layernorm: bool = True,
        activation_function: Literal["relu", "gelu"] = "relu",
    ):
        """Create MLP object.

        Args:
            n: number of vertices in the graph.
            hidden_layer_sizes: List of hidden layer sizes.
            output_size: Size of final layer
            dropout_probability: probability of a dropout at each layer during training
            layernorm: whether or not to include layer normalization layers
            activation_function: which activation function to use between hidden layers

        Raises:
            ValueError: if dropout_probability is not in range [0, 1]
        """
        super().__init__()
        self.n = n
        self.edges = (n**2 - n) // 2
        activation: type[nn.Module] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[activation_function.lower()]
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError("dropout_probability must be a probability")

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [128, 64, 64]  # From Wagner 2021

        model_layers = []
        model_layers.append(("input_layer", nn.Linear(2 * self.edges, hidden_layer_sizes[0])))
        if layernorm:
            model_layers.append(("layernorm_input", nn.LayerNorm(hidden_layer_sizes[0])))
        model_layers.append(("activation_input", activation()))
        if dropout_probability > 0:
            model_layers.append(("dropout_input", nn.Dropout(dropout_probability)))

        prev_layer_size = hidden_layer_sizes[0]
        for i, layer_size in enumerate(hidden_layer_sizes[1:]):
            model_layers.append((f"hidden_{i}", nn.Linear(prev_layer_size, layer_size)))
            if layernorm:
                model_layers.append((f"layernorm_{i}", nn.LayerNorm(layer_size)))
            model_layers.append((f"activation_{i}", activation()))
            if dropout_probability > 0:
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

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample batch_size examples from model.

        Returns:
            example tensor
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                w = torch.zeros((batch_size, self.edges), device=device)
                for i in range(self.edges):
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

    @staticmethod
    def wagner_model(n: int) -> "MLP":
        """Feed forward network used in [Wagner 2021](http://arxiv.org/abs/2104.14516).

        Returns:
            Pytorch implementation of Wagner's model
        """
        return MLP(
            n=n,
            hidden_layer_sizes=[128, 64, 4],
            output_size=2,
            dropout_probability=0.0,
            layernorm=False,
            activation_function="relu",
        )
