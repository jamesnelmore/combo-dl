import torch
from torch import nn
from torch.nn import functional as F

from .protocols import SamplingModel

class WagnerModel(SamplingModel):
    def __init__(self, n: int):
        """
        n: number of vertices in the graph
        """
        super().__init__()
        self.n = n
        self.edges = (n**2 - n) // 2

        self.layers = nn.Sequential(
            nn.Linear(2 * self.edges, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        one_hot_position = F.one_hot(i.to(device=device), num_classes=self.edges).to(
            dtype=x.dtype
        )

        if one_hot_position.dim() == 1:
            one_hot_position = one_hot_position.unsqueeze(0)

        input_tensor = torch.cat((x, one_hot_position), dim=-1)
        return self.layers(input_tensor)

    def sample(self, batch_size: int) -> torch.Tensor:
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
