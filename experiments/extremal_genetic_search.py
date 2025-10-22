"""Extremal Graph Theory Genetic Search Algorithm.

Adapted from guided_genetic_search.py for extremal graph theory problems.
Uses individual edge toggles instead of edge swaps and integrates with Wagner corollary problem.
"""

from collections import OrderedDict
from collections.abc import Callable
from typing import Literal, override
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
from torch import nn
from tqdm import tqdm

from combo_dl.wagner_corollary_2_1_problem import WagnerCorollary21


class ExtremalGraphMLP(nn.Module):
    """Neural network for extremal graph theory problems.

    Takes a graph as input and outputs logits over individual edges to toggle.
    Unlike the construction problem, we don't maintain k-regularity constraints.
    """

    def __init__(
        self,
        n: int,
        hidden_layer_sizes: list[int] | None = None,
        dropout_probability: float = 0.1,
        layernorm: bool = True,
        activation_function: Literal["relu", "gelu"] = "relu",
    ):
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
            hidden_layer_sizes = [128, 64, 64]

        model_layers = []
        model_layers.append(("input_layer", nn.Linear(self.edges, hidden_layer_sizes[0])))
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

        model_layers.append(("output", nn.Linear(prev_layer_size, self.edges)))

        self.layers = nn.Sequential(OrderedDict(model_layers))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes in a graph edge list, outputs logits over edges to toggle.

        Args:
            x: Tensor of shape (batch_size, edges) representing graph edge vectors

        Returns:
            Tensor of shape (batch_size, edges) with logits for edge toggles
        """
        return self.layers(x)


def random_toggle_mutation(g: torch.Tensor, device: str) -> torch.Tensor:
    """Random mutation function for extremal graph problems.

    Returns uniform random logits over all edges for toggling.
    Unlike the construction problem, we can toggle any edge.

    Args:
        g: Tensor of shape (batch_size, edges) representing graphs
        device: Device to use for tensor operations

    Returns:
        Tensor of shape (batch_size, edges) with uniform random logits
    """
    return torch.rand(g.shape, device=device)


def gen_random_graphs(
    pop_size: int, edges: int, edge_probability: float, device: str
) -> torch.Tensor:
    """Generate random graphs for extremal problems.

    Unlike construction problems, we don't need to maintain k-regularity.
    We can generate graphs with any edge density.

    Args:
        pop_size: Number of graphs to generate
        edges: Number of possible edges
        edge_probability: Probability of each edge being present
        device: Device to use for tensor operations

    Returns:
        Tensor of shape (pop_size, edges) with binary edge vectors
    """
    return torch.bernoulli(torch.full((pop_size, edges), edge_probability, device=device))


def extremal_search_loop(
    num_random: int,
    top_old_elite: int,
    top_new_elite: int,
    iterations: int,
    mutator: Callable,  # returns logits over the edges to toggle
    problem: WagnerCorollary21,
    device: str,
    edge_probability: float = 0.3,  # Initial edge density
):
    """Main extremal graph theory genetic search loop.

    Args:
        num_random: Number of random graphs in each generation
        top_old_elite: Number of top elites from previous generation
        top_new_elite: Number of top elites from current generation
        iterations: Number of iterations to run
        mutator: Function that returns logits for edge toggles
        problem: WagnerCorollary21 problem instance
        device: Device to use for tensor operations
        edge_probability: Initial probability of edges being present
    """
    pop_size = num_random + top_old_elite + top_new_elite

    fitness = problem.reward
    edges = problem.edges

    # Generate initial population with random edge density
    pop = gen_random_graphs(pop_size, edges, edge_probability, device)
    print(f"Population size: {pop_size}, pop array shape: {pop.shape}")
    print(f"Initial edge density: {pop.mean().item():.3f}")

    random_pop = pop[0:num_random]
    old_elite_pop = pop[num_random : num_random + top_old_elite]
    new_elite_pop = pop[num_random + top_old_elite :]

    pop_fitness = torch.zeros(pop_size, device=device)

    # Initialize previous generation elites (empty for first iteration)
    prev_elites = torch.empty(0, edges, device=device)

    # Track mutation effectiveness
    total_mutations = 0
    successful_mutations = 0

    pbar = tqdm(range(iterations))
    for _ in pbar:
        # Get fitness before mutations
        pop_fitness_before = fitness(pop)
        _, old_elite_indices = torch.topk(pop_fitness_before, top_old_elite)
        old_elite = pop[old_elite_indices].clone()

        # Get mutation probabilities from the mutator
        mutation_logits = mutator(pop, device)
        mutation_probs = torch.softmax(mutation_logits, dim=-1)

        # Sample edges to toggle based on mutation probabilities
        edges_to_toggle = torch.multinomial(mutation_probs, num_samples=1, replacement=False)

        # Apply mutations by toggling selected edges
        # Use advanced indexing to toggle the selected edges for each graph
        batch_indices = torch.arange(pop_size, device=device).unsqueeze(1)
        pop[batch_indices, edges_to_toggle] = 1 - pop[batch_indices, edges_to_toggle]

        # Compute fitness after mutations
        pop_fitness = fitness(pop).to(device)

        # Track mutation effectiveness
        improvements = (pop_fitness > pop_fitness_before).sum().item()
        total_mutations += pop_size
        successful_mutations += improvements

        # Calculate per-iteration success rate
        iter_success_rate = (improvements / pop_size * 100) if pop_size > 0 else 0

        # Get new elites
        _, new_elite_indices = torch.topk(pop_fitness, top_new_elite)
        new_elite = pop[new_elite_indices].clone()

        # Check for early stopping
        best_score = pop_fitness.max().item()
        should_stop, stop_reason = problem.should_stop_early(best_score)
        if should_stop:
            print(f"\nEarly stopping: {stop_reason}")
            break

        # Update population for next iteration
        # Keep the best individuals from both current and previous generations
        components = [new_elite]  # Keep the best elites from current generation

        # Add previous generation elites if we have them
        if prev_elites.shape[0] > 0:
            components.append(prev_elites)

        # Fill the rest with random graphs
        remaining_size = pop_size - new_elite.shape[0] - prev_elites.shape[0]
        if remaining_size > 0:
            # Use current best edge density as a guide for new random graphs
            current_density = pop.mean().item()
            components.append(gen_random_graphs(remaining_size, edges, current_density, device))

        pop = torch.cat(components, dim=0)

        # Update previous elites for next iteration
        prev_elites = new_elite.clone()

        max_fitness = pop_fitness.max().item()
        avg_fitness = pop_fitness.mean().item()
        current_density = pop.mean().item()

        # Calculate mutation success rate
        success_rate = (successful_mutations / total_mutations * 100) if total_mutations > 0 else 0

        pbar.set_postfix({
            "max_fitness": f"{max_fitness:.4f}",
            "avg_fitness": f"{avg_fitness:.4f}",
            "edge_density": f"{current_density:.3f}",
            "iter_success": f"{iter_success_rate:.1f}%",
            "total_success": f"{success_rate:.1f}%",
        })


def main():
    """Main function to run extremal graph theory genetic search."""
    # Set device in main function
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    iterations = 10_000
    n = 20  # Number of vertices
    problem = WagnerCorollary21(n)

    # Parameters for extremal graph search
    # Smaller population sizes since we don't need to maintain k-regularity
    num_random = 100
    top_old_elite = 50
    top_new_elite = 50

    print(f"Running extremal genetic search on Wagner Corollary 2.1 with n={n}")
    print(f"Goal: Find graphs with eigenvalue + matching < {-(problem.goal_score):.6f}")

    extremal_search_loop(
        num_random=num_random,
        top_old_elite=top_old_elite,
        top_new_elite=top_new_elite,
        iterations=iterations,
        mutator=random_toggle_mutation,
        problem=problem,
        device=device,
        edge_probability=0.3,
    )


if __name__ == "__main__":
    main()
