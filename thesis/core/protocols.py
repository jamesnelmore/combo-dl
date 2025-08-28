from typing import Protocol
from torch import nn, Tensor

class SupportsSampling(Protocol):
    """Protocol for Models that can be sampled from."""

    def sample(self, batch_size: int) -> Tensor:
        ...
    
    def forward(self, *args, **kwargs) -> Tensor:
        ...

class SamplingModel(nn.Module, SupportsSampling):
    """Model that can be sampled from."""
    pass