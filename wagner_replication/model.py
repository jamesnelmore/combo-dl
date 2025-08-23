import torch
from torch import nn
from torch.nn import Linear, GELU, Dropout
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

device = "mps"


class WagnerModel(nn.Module):
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
        # assert i.dim() == 0 or (i.dim() == 1 and i.numel() == 1), (
        #     "i must be a singleton tensor"
        # )
        # assert i.item() < self.edges, (
        #     f"i ({i.item()}) must be less than number of edges ({self.edges})"
        # )
        one_hot_position = F.one_hot(i.to(device=device), num_classes=self.edges).to(
            dtype=x.dtype
        )

        if one_hot_position.dim() == 1:
            one_hot_position = one_hot_position.unsqueeze(0)

        input_tensor = torch.cat((x, one_hot_position), dim=-1)
        return self.layers(input_tensor)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, dim=-1)


def generate_sampled_constructions(model: WagnerModel, batch_size=64):
    w = torch.zeros((batch_size, model.edges), device=device)
    for i in range(model.edges):
        i_tensor = torch.full((batch_size,), i, dtype=torch.long, device=device)
        x = model(w, i_tensor)
        assert x.shape == (batch_size, 2)
        probs = F.softmax(x, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        w[:, i] = sampled
    return w


def select_elites(
    constructions: torch.Tensor,
    batch_scores: torch.Tensor,
    elite_proportion: float = 0.1,
):
    batch_size = len(batch_scores)
    return_count = int(batch_size * elite_proportion)
    elite_indices = torch.argsort(batch_scores)
    return torch.stack([constructions[i] for i in elite_indices]).to(
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
            step_mask = torch.arange(edges, device=construction.device) < position
            observations.append(construction * step_mask)

    # Get the device from elite_constructions
    target_device = elite_constructions.device

    obs_tensor = torch.stack(observations).to(target_device)
    pos_tensor = torch.tensor(positions, dtype=torch.long, device=target_device)
    actions_tensor = torch.stack(actions).to(target_device)
    dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

    return dataloader


from tqdm import tqdm

def train(model: WagnerModel, train_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for batch_idx, (obs, pos, target_actions) in enumerate(progress_bar):
        optimizer.zero_grad()
        outputs = model(obs, pos)
        loss = criterion(outputs, target_actions)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += target_actions.size(0)
        train_correct += (predicted == target_actions).sum().item()

        avg_loss = train_loss / (batch_idx + 1)
        accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{accuracy:.2f}%"
        })

if __name__ == "__main__":
    model = WagnerModel(4).to(device)
    w = generate_sampled_constructions(model, batch_size=16)
    batch_scores = model.score(w)
    elites = select_elites(w, batch_scores, 0.1)
    dataloader = extract_examples(elites)
    
    train(model, dataloader)
