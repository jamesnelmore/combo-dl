# R(s,t) is the minimum number of vertices n such that any 2-coloring of the edges of K_n must
# contain either
# - A red clique of size s, OR
# - A blue clique of size t
# Prove a lower bound l by finding a coloring on l-1 vertices not containing a red or blue clique
# A (s,t)-Ramsey graph is a graph on R(s,t) - 1 vertices avoiding both a clique of size s and an
# independent set of size t

# warnings.filterwarnings("ignore", message=r".*UnsupportedFieldAttributeWarning.*frozen.*")
from itertools import combinations
from multiprocessing import Pool
from typing import override

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from combo_dl import WagnerDeepCrossEntropy
from combo_dl.graph_utils import edge_vec_to_adj
from combo_dl.models.mlp import MLP


# Bron-Kerbosch algorithm ###########
def bron_kerbosch_pivot(adj: np.ndarray, R: set[int], P: set[int], X: set[int]) -> int:  # noqa: N803
    """Recursive step of the Bron-Kerbosch pivot algorithm to find maximum clique size.
    Args:
        adj: adjacency matrix
        R: current clique under construction
        P: candidate vertices
        X: eliminated vertices
    Returns:
        Size of r once fully built
    """
    if not P and not X:
        return len(R)

    pivot = max(P | X, key=lambda v: len(P & set(np.where(adj[v])[0])))
    pivot_neighbors = set(np.where(adj[pivot])[0])

    max_size = 0
    for v in P - pivot_neighbors:
        neighbors = set(np.where(adj[v])[0])
        clique_size = bron_kerbosch_pivot(adj=adj, R=R | {v}, P=P & neighbors, X=X & neighbors)
        P.remove(v)
        X.add(v)
        max_size = max(max_size, clique_size)
    return max_size


def max_clique_size(adj: np.ndarray) -> int:
    n = adj.shape[0]
    return bron_kerbosch_pivot(adj, R=set(), P=set(range(n)), X=set())


def batched_max_clique(adjs: np.ndarray, n_workers: int | None = None):
    with Pool(n_workers) as pool:
        rewards = pool.map(max_clique_size, adjs)
    return torch.tensor(rewards)


# End Bron-Kerbosch algorithm ###########


def ramsey_reward(
    adjs: torch.Tensor,
    r: int,
    s: int,
    spectral_weight: float = 0.1,
    violation_penalty: float = 50.0,
    valid_bonus: float = 1000.0,
) -> torch.Tensor:
    """Compute reward for Ramsey graph generation.

    Args:
        adjs: Batched adjacency matrices of shape (batch_size, n, n)
        r: Maximum allowed clique size (graph avoids K_r)
        s: Maximum allowed independent set size (graph avoids I_s)
        spectral_weight: Weight for spectral gap bonus (0 to disable)
        violation_penalty: Penalty multiplier for violations
        valid_bonus: Bonus reward for valid Ramsey graphs

    Returns:
        Tensor of shape (batch_size,) with rewards
    """
    batch_size, n, _ = adjs.shape
    device = adjs.device

    adjs_np = adjs.detach().cpu().numpy()
    # Compute complement: ~adj & ~eye (exclude self-loops)
    eye = torch.eye(n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    adjs_complement = (~adjs.bool() & ~eye.bool()).detach().cpu().numpy()

    max_cliques = batched_max_clique(adjs_np).to(device)
    max_indep_sets = batched_max_clique(adjs_complement).to(device)

    excess_colored_sets = torch.maximum(max_cliques - r, max_indep_sets - s)

    # Compute spectral gap: Largest eigval - second largest eigval
    # MPS doesn't support eigenvalues
    original_device = adjs.device
    adjs_for_eig = adjs.float()
    if adjs_for_eig.device.type == "mps":
        adjs_for_eig = adjs_for_eig.cpu()

    eigvals = torch.linalg.eigvals(adjs_for_eig).real  # Adjacency matrices are real-symmetric
    eigvals = eigvals.to(original_device)  # Move back to original device
    eigvals_sorted, _ = torch.sort(eigvals, descending=True, dim=1)
    if eigvals_sorted.shape[1] > 1:
        spectral_gaps = eigvals_sorted[:, 0] - eigvals_sorted[:, 1]
    else:
        spectral_gaps = torch.zeros(batch_size, device=device)

    # Reward structure:
    # 1. Primary reward: Large bonus for valid graphs, penalty for violations
    is_valid = excess_colored_sets < 0
    base_rewards = torch.where(
        is_valid,
        valid_bonus
        + excess_colored_sets,  # Bonus plus small reward for being well below threshold
        -violation_penalty * excess_colored_sets,  # Penalty proportional to violation size
    )

    # 2. Spectral gap bonus: Normalize by expected gap for regular graphs
    # Expected gap ≈ 2√(d-1) for d-regular graphs (Alon-Boppana bound)
    if spectral_weight > 0:
        avg_degrees = adjs.sum(dim=(1, 2)) / n
        expected_gaps = avg_degrees - 2 * torch.sqrt(torch.clamp(avg_degrees - 1, min=0.0))
        normalized_gaps = spectral_gaps / (expected_gaps + 1e-8)
        spectral_bonus = spectral_weight * normalized_gaps
        base_rewards = base_rewards + spectral_bonus

    return base_rewards


class RamseyNumbersProblem:
    """Problem class for finding Ramsey graphs using deep cross-entropy method.

    A (r,s)-Ramsey graph on n vertices avoids both:
    - A clique of size r (K_r)
    - An independent set of size s

    This is used to prove lower bounds on Ramsey numbers R(r,s).
    """

    def __init__(
        self,
        n: int,
        r: int,
        s: int,
        spectral_weight: float = 0.1,
        violation_penalty: float = 50.0,
        valid_bonus: float = 1000.0,
    ):
        """Initialize Ramsey numbers problem.

        Args:
            n: Number of vertices in the graph
            r: Maximum allowed clique size (graph avoids K_r)
            s: Maximum allowed independent set size (graph avoids I_s)
            spectral_weight: Weight for spectral gap bonus (0 to disable)
            violation_penalty: Penalty multiplier for violations
            valid_bonus: Bonus reward for valid Ramsey graphs
        """
        self.n = n
        self.r = r
        self.s = s
        self.edges = (n * (n - 1)) // 2
        self.spectral_weight = spectral_weight
        self.violation_penalty = violation_penalty
        self.valid_bonus = valid_bonus

        # TODO: Remove these dummy attributes once WagnerDeepCrossEntropy is refactored
        # to support generic problem types instead of hardcoding StronglyRegularGraphs attributes
        # These are only used for wandb logging compatibility
        self.k = -1  # Dummy attribute for SRG compatibility
        self.lambda_param = -1  # Dummy attribute for SRG compatibility
        self.mu = -1  # Dummy attribute for SRG compatibility

        print(f"Ramsey problem: Find graph on {n} vertices avoiding K_{r} and I_{s}")
        print(f"Goal: Find valid (r={r}, s={s})-Ramsey graph on n={n} vertices")

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reward for each graph in the batch.

        Args:
            x: Tensor of shape (batch_size, edges) where each entry is 0 or 1
               representing whether an edge is present in the graph

        Returns:
            Tensor of shape (batch_size,) with rewards for each construction.
            Higher rewards are better.
        """
        # Convert edge vectors to adjacency matrices
        adj_matrices = edge_vec_to_adj(x, self.n)

        # Compute rewards using the ramsey_reward function
        rewards = ramsey_reward(
            adj_matrices,
            self.r,
            self.s,
            spectral_weight=self.spectral_weight,
            violation_penalty=self.violation_penalty,
            valid_bonus=self.valid_bonus,
        )

        return rewards

    def should_stop_early(self, best_score: float) -> tuple[bool, str]:
        """Check if optimization should stop early (found valid Ramsey graph).

        Args:
            best_score: Current best reward score

        Returns:
            Tuple of (should_stop, reason_message)
        """
        # A valid Ramsey graph should have reward >= valid_bonus
        # (since valid_bonus is the base reward for valid graphs)
        if best_score >= self.valid_bonus:
            return (
                True,
                f"Valid Ramsey graph found! Score: {best_score:.2f} >= {self.valid_bonus:.2f}",
            )
        return False, ""

    def is_valid_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Check if solution is valid (binary values and correct dimension).

        Args:
            solution: Tensor of shape (batch_size, edges) where each entry should be 0 or 1

        Returns:
            Tensor of shape (batch_size,) with boolean values indicating validity
        """
        # Check that the tensor has the correct shape
        if solution.dim() != 2 or solution.shape[1] != self.edges:
            # Return False for all batch items if shape is wrong
            batch_size = solution.shape[0] if solution.dim() >= 1 else 1
            return torch.zeros(batch_size, dtype=torch.bool, device=solution.device)

        # Check that each solution vector contains only binary values (0 or 1)
        return torch.all((solution == 0) | (solution == 1), dim=1)


class PolicyNet(nn.Module):
    def __init__(self, n_edges):
        super().__init__()
        # Larger network to use more GPU compute
        self.net = nn.Sequential(
            nn.Linear(2 * n_edges, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    @override
    def forward(self, partial, position):
        x = torch.cat([partial, position], dim=-1)
        return self.net(x)


def train(
    n=5,
    s=3,
    t=3,
    iterations=5,
    batch_size=16000,
    reward_type="distance_based",  # noqa: ARG001
):
    # Determine device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    n_edges = n * (n - 1) // 2
    edge_list = list(combinations(range(n), 2))

    policy = PolicyNet(n_edges).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    print(f"Training parameters: n={n}, s={s}, t={t}, n_edges={n_edges}, batch_size={batch_size}")
    print(f"Policy network: {policy}")
    print("Starting training...\n")

    best_reward = -100

    pbar = tqdm(range(iterations), desc="Training")
    for it in pbar:
        # Generate batch using batched operations
        # Initialize batch: all graphs start with no edges
        partial_batch = torch.zeros(batch_size, n_edges, device=device)
        adjs = np.zeros((batch_size, n, n), dtype=int)
        # Store actions as a tensor for efficient computation
        all_actions = torch.zeros(batch_size, n_edges, dtype=torch.long, device=device)

        # Process all edges in sequence, but batch across all graphs
        for edge_idx, _ in enumerate(edge_list):
            # Create position vector for this edge (same for all graphs)
            position = torch.zeros(n_edges, device=device)
            position[edge_idx] = 1
            position_batch = position.unsqueeze(0).expand(batch_size, -1)

            # Batch forward pass: process all graphs at once
            with torch.no_grad():
                logits = policy(partial_batch, position_batch)  # [batch_size, 2]
                probs = torch.softmax(logits, dim=-1)  # [batch_size, 2]
                actions = torch.multinomial(probs, 1).squeeze(-1)  # [batch_size]

            # Store actions
            all_actions[:, edge_idx] = actions

            # Vectorized update: update partial states where action == 1
            action_mask = actions == 1
            partial_batch[action_mask, edge_idx] = 1

        # Vectorized adjacency matrix construction (keep on GPU for speed)
        # Convert actions to numpy only once at the end
        all_actions_np = all_actions.cpu().numpy()
        for edge_idx, (i, j) in enumerate(edge_list):
            edge_mask = all_actions_np[:, edge_idx] == 1
            adjs[edge_mask, i, j] = 1
            adjs[edge_mask, j, i] = 1

        # Check all graphs (efficient numpy operations make this fast)
        # Get both validity and rewards - validity check is separate from reward
        # NOTE: check_ramsey function was removed. This train function is deprecated.
        # Use main() with mlp_dce instead.
        # For now, using a simplified reward computation:
        adjs_tensor = torch.from_numpy(adjs).float()
        rewards = ramsey_reward(adjs_tensor, r=s, s=t).numpy()
        # Check validity: valid if reward >= valid_bonus (1000.0 by default)
        valids = rewards >= 1000.0
        batch_graphs = [(adjs[b_idx], rewards[b_idx]) for b_idx in range(batch_size)]

        # CRITICAL: Check for valid graphs FIRST (early stopping priority)
        # Positive reward does NOT mean valid - invalid graphs can have positive rewards!
        valid_indices = np.where(valids)[0]
        if len(valid_indices) > 0:
            # Found valid graph(s)! Use the one with highest reward
            valid_rewards = rewards[valid_indices]
            best_valid_idx = valid_indices[np.argmax(valid_rewards)]
            best_valid_reward = rewards[best_valid_idx]

            best_reward = max(best_reward, best_valid_reward)
            msg_valid = (
                f"\nIteration {it}: Found VALID Ramsey graph with reward {best_valid_reward}!"
            )
            pbar.write(msg_valid)
            pbar.write(str(adjs[best_valid_idx]))
            msg = f"\nEarly stopping: Valid Ramsey graph found with {n} vertices!"
            pbar.write(msg)
            break

        # No valid graphs found - continue training with best invalid graph
        max_reward_idx = np.argmax(rewards)
        max_reward = rewards[max_reward_idx]
        avg_reward = np.mean(rewards)

        # Update progress bar with metrics
        pbar.set_postfix({
            "avg_reward": f"{avg_reward:.1f}",
            "max_reward": f"{max_reward:.1f}",
            "best_reward": f"{best_reward:.1f}",
        })

        if max_reward > best_reward:
            best_reward = max_reward
            # Note: This graph is NOT valid (we already checked above)
            # But it has the best reward so far
            pbar.write(f"\nIteration {it}: Found invalid graph with reward {max_reward}")
            pbar.write(str(adjs[max_reward_idx]))

        # Select top 10% and reconstruct trajectories only for them
        top_k = max(1, batch_size // 10)
        top_indices = np.argsort(rewards)[-top_k:]

        # Reconstruct trajectories only for top performers (lazy evaluation)
        batch_trajectories = []
        for idx in top_indices:
            trajectory = []
            partial = torch.zeros(n_edges, device=device)
            for edge_idx in range(n_edges):
                position = torch.zeros(n_edges, device=device)
                position[edge_idx] = 1
                action = all_actions[idx, edge_idx].item()
                trajectory.append((partial.clone(), position.clone(), action))
                if action == 1:
                    partial[edge_idx] = 1
            batch_trajectories.append((idx, trajectory))

        # Train on top performers - batch all forward passes
        if batch_trajectories:
            # Collect all (partial, position, action) tuples
            all_partials = []
            all_positions = []
            all_actions = []
            for _, trajectory in batch_trajectories:
                for partial_traj, position_traj, action in trajectory:
                    all_partials.append(partial_traj)
                    all_positions.append(position_traj)
                    all_actions.append(action)

            # Batch forward pass
            partials_batch = torch.stack(all_partials)  # [N, n_edges]
            positions_batch = torch.stack(all_positions)  # [N, n_edges]
            actions_batch = torch.tensor(all_actions, device=device)

            logits_batch = policy(partials_batch, positions_batch)  # [N, 2]
            log_probs = torch.log_softmax(logits_batch, dim=-1)  # [N, 2]
            # Gather log probs for the taken actions
            loss = -log_probs[range(len(all_actions)), actions_batch].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return best_reward


def main():
    """Main function to run Ramsey numbers problem with mlp_dce."""
    # Print immediately to show script is starting (before expensive operations)
    print("=" * 60)
    print("Ramsey Numbers Problem Configuration")
    print("=" * 60)

    # Configuration
    n = 17  # Number of vertices (R(4,4) = 18, so test with n=17)
    r = 4  # Maximum allowed clique size
    s = 4  # Maximum allowed independent set size

    # Training hyperparameters
    iterations = 10000
    batch_size = 64
    learning_rate = 0.001
    elite_proportion = 0.1
    early_stopping_patience = iterations

    # Reward function hyperparameters
    spectral_weight = 0.1
    violation_penalty = 50.0
    valid_bonus = 1000.0

    # Model hyperparameters
    hidden_layer_sizes = [64, 32, 16, 8, 4]
    output_size = 2
    dropout_probability = 0.1
    layernorm = True
    activation_function = "relu"

    # Device setup
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Graph: n={n}, r={r}, s={s}")
    print(f"Training: iterations={iterations}, batch_size={batch_size}")
    print(f"Device: {device}")
    print("=" * 60)

    # Initialize problem
    problem = RamseyNumbersProblem(
        n=n,
        r=r,
        s=s,
        spectral_weight=spectral_weight,
        violation_penalty=violation_penalty,
        valid_bonus=valid_bonus,
    )

    # Initialize model
    model = MLP(
        n=problem.n,
        hidden_layer_sizes=hidden_layer_sizes,
        output_size=output_size,
        dropout_probability=dropout_probability,
        layernorm=layernorm,
        activation_function=activation_function,
    )

    # Initialize Deep Cross Entropy optimizer
    # Create experiment name with Ramsey parameters
    experiment_name = f"ramsey_n{n}_r{r}_s{s}"
    dce = WagnerDeepCrossEntropy(
        model,
        problem,  # type: ignore[arg-type]
        iterations=iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        elite_proportion=elite_proportion,
        early_stopping_patience=early_stopping_patience,
        device=device,
        hydra_cfg=None,
        checkpoint_frequency=50,
        save_best_constructions=True,
        experiment_name=experiment_name,
    )

    # Run optimization
    dce.optimize()


if __name__ == "__main__":
    main()
