import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Callable

from wagner_model import WagnerModel, generate_sampled_constructions

device = "mps"


def select_elites(
    constructions: torch.Tensor,
    batch_scores: torch.Tensor,
    elite_proportion: float = 0.1,
):
    batch_size = len(batch_scores)
    return_count = int(batch_size * elite_proportion)
    elite_indices = torch.argsort(batch_scores, descending=True)[:return_count]
    return constructions[elite_indices]


def extract_examples(
    elite_constructions: torch.Tensor, output_batch_size: int = 512
) -> DataLoader:
    """For each construction, mask it past i with 0s. Then the state is the masked construction, the position is i + 1, and the action is what was at i + 1 before the mask. By default the output batch size will match the input batch size
    """

    num_constructions, num_edges = elite_constructions.shape
    target_device = elite_constructions.device

    # Create mask to hide future positions during training
    # For each position i, we want to mask out positions >= i
    pos_tensor = (
        torch.arange(num_edges).repeat(num_constructions).to(target_device)
    )  # The current position we're predicting for
    edge_indices = torch.arange(num_edges).to(
        target_device
    )  # All edge positions [0, 1, 2, 3, 4, 5]
    mask = (edge_indices.unsqueeze(0) < pos_tensor.unsqueeze(1)).to(target_device)

    obs_tensor = elite_constructions.repeat_interleave(num_edges, dim=0) * mask
    obs_tensor.to(target_device)
    construction_indices = (
        torch.arange(num_constructions).repeat_interleave(num_edges).to(target_device)
    )  # [0, 0, 0, 1, 1, 1, ...]
    # Defined elementwise: T[i] = elite_constructions[i][pos_tensor[i]]
    actions_tensor = elite_constructions[construction_indices, pos_tensor].to(
        target_device
    )

    dataset = TensorDataset(obs_tensor, pos_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=output_batch_size, shuffle=True)

    return dataloader


def train(
    model: WagnerModel,
    train_loader: DataLoader,
    progress_callback: Callable[[float, float], None] | None = None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (obs, pos, target_actions) in enumerate(train_loader):
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
        if progress_callback is not None:
            progress_callback(avg_loss, accuracy)


def deep_cross_entropy_wagner(model: WagnerModel, batch_size=512):
    iterations = 10_000
    progress_bar = tqdm(range(iterations), desc="DCE Iterations")
    best_score = float('-inf')
    
    for iteration in progress_bar:
        w = generate_sampled_constructions(model, batch_size=batch_size)
        batch_scores = model.score(w)
        current_best = torch.max(batch_scores).item()
        
        if current_best > best_score:
            best_score = current_best
            
        elites = select_elites(w, batch_scores, 0.1)
        
        # Check if we actually have elites
        if len(elites) == 0:
            print(f"Warning: No elites selected at iteration {iteration}")
            continue
            
        dataloader = extract_examples(elites, output_batch_size=min(batch_size, len(elites) * model.edges))

        def progress_callback(avg_loss, accuracy):
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}", 
                "acc": f"{accuracy:.2f}%",
                "best_score": f"{best_score:.4f}",
                "avg_score": f"{torch.mean(batch_scores).item():.4f}"
            })

        train(model, dataloader, progress_callback=progress_callback)


def main():
    model = WagnerModel(4).to(device)
    deep_cross_entropy_wagner(model, batch_size=4096)


if __name__ == "__main__":
    main()
