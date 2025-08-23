import torch
from torch import nn
from torch.nn import Linear, GELU, Dropout
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

device = "mps"


class WagnerModel(nn.Module):
    def __init__(self, n: int):
        """
        n: number of vertices in the graph
        """

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
        assert i.dim() == 0 or (i.dim() == 1 and i.numel() == 1), (
            "i must be a singleton tensor"
        )
        assert i.item() < self.edges, (
            f"i ({i.item()}) must be less than number of edges ({self.edges})"
        )
        one_hot_position = F.one_hot(i.to(device=device), num_classes=self.edges).to(
            dtype=x.dtype
        )

        if one_hot_position.dim() == 1:
            one_hot_position = one_hot_position.unsqueeze(0)

        input_tensor = torch.cat((x, one_hot_position), dim=-1)
        return self.layers(input_tensor)

    def score(self, x: torch.Tensor) -> torch.Tensor: ...


def generate_sampled_constructions(model: WagnerModel, batch_size=64):
    w = torch.zeros((batch_size, model.n))
    for i in range(model.edges):
        i_tensor = torch.full((batch_size,), i, dtype=torch.long, device=w.device)
        x = model(w, i_tensor)
        assert x.shape == (batch_size, 2)
        probs = F.softmax(x, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        w[:, i] = sampled
    return w


def select_elite(
    constructions: torch.Tensor,
    batch_scores: torch.Tensor,
    elite_proportion: float = 0.1,
):
    batch_size = len(batch_scores)
    return_count = int(batch_size * elite_proportion)
    elite_indices = torch.argsort(constructions)
    return torch.tensor([constructions[i] for i in elite_indices]).to(
        constructions.device
    )


def extract_examples(
    elite_constructions: torch.Tensor, output_batch_size=64
) -> DataLoader:
    # For each construction, mask it past i with 0s. Then the state is the masked construction, the position is i + 1, and the action is what was at i + 1 before the mask

    observations = []
    positions = []
    actions = []
    for construction in elite_constructions:  # TODO heavily optimize
        edges = len(construction)
        for position in range(edges):
            positions.append(position)
            actions.append(construction[position])
            step_mask = torch.arange(edges) < position
            observations.append(construction * step_mask)

    obs_tensor = torch.stack(observations)
    pos_tensor = torch.tensor(positions, dtype=torch.long)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

    return dataloader


def train(n=600, batch_size=1_000):
    model = WagnerModel(n)

    while True:  # best found construction is not a counterexample
        # Generate constructions via random sampling from action space
        w = generate_sampled_constructions(model)

        # Evaluate the score of each construction
        batch_scores = model.score(w)
        # Outputs the edge tuple. We can make model.edges examples from this
        remaining_constructions = select_elite(w, batch_scores, 0.1)

        break
