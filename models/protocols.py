"""Protocols that allow algorithms to specify the type of models they support."""

from typing import Protocol

from torch import Tensor, nn


class SupportsSampling(Protocol):
    """Protocol for Models that can be sampled from."""

    def sample(self, batch_size: int) -> Tensor:
        """Sample one batch from the Model."""
        ...

    def forward(self, *args, **kwargs) -> Tensor:
        """Run a normal forward pass."""
        ...


class SamplingModel(nn.Module, SupportsSampling):
    """Model that can be sampled from."""

    ...
