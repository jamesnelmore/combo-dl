"""Deep Cross-Entropy Algorithm."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Literal, override
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
from torch import nn
from tqdm import tqdm

from combo_dl.strongly_regular_graphs_problem import StronglyRegularGraphs

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


class NonSequentialGraphMLP(nn.Module):
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
        """Takes in a graph edge list, logits of a distribution over edges representing the probability that toggling a given edge will bring the graph closer to the objecive."""
        return self.layers(x)


"""Guided local search. Like genetic search with a neural approximation for the mutation function.
Some mix of random, old elite, and new elite graphs in each batch.
- Fitness function F
- Policy P generates distributions over the edges. We sample a pairs of edges to swap.
    - Train the policy with supervised learning on random edge swaps
    - Try live training via PPO or REINFORCE if necessary (likely will be)

1. Run loop with random fitness function
"""


def random_swap_mutation(g: torch.Tensor) -> torch.Tensor:
    # g must be batched
    uniform_random = torch.rand(g.shape, device=device)
    return uniform_random


def gen_random_reg(pop_size: int, edges: int, k: int) -> torch.Tensor:
    pop = torch.zeros((pop_size, edges), device=device)
    random_values = torch.rand(pop_size, edges, device=device)
    _, topk_indices = torch.topk(random_values, k)
    return pop.scatter_(dim=-1, index=topk_indices, value=1.0)


def guided_search_loop(
    num_random: int,
    top_old_elite: int,
    top_new_elite: int,
    iterations: int,
    mutator: Callable,  # returns logits over the edges to mutate
    problem: StronglyRegularGraphs,
):
    pop_size = num_random + top_old_elite + top_new_elite

    fitness = problem.reward
    edges = problem.edges()

    pop = gen_random_reg(pop_size, edges, problem.k)
    print(f"Population size: {pop_size}, pop array shape: {pop.shape}")
    assert (pop.sum(dim=-1) == problem.k).all(), "k-regularity broken"

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
    for idx in pbar:
        # Algorithm at step i
        # Get fitness of population
        # Determine current elites
        # Mutate the population
        # Get fitness of new population
        # Extract the old and new elite, generate random graphs, repeat

        # Get fitness before mutations
        pop_fitness_before = fitness(pop)
        _, old_elite_indices = torch.topk(pop_fitness_before, top_old_elite)
        old_elite = pop[old_elite_indices].clone()
        mutation_probs = mutator(pop)

        # Sample present edge to remove
        present_mask = (pop == 1).float()
        absent_mask = (pop == 0).float()

        # Check for invalid graphs (completely full or empty)
        num_present_edges = present_mask.sum(dim=-1)
        num_absent_edges = absent_mask.sum(dim=-1)

        # Only mutate graphs that have both present and absent edges
        valid_for_mutation = (num_present_edges > 0) & (num_absent_edges > 0)

        # Filter to only valid graphs
        valid_pop = pop[valid_for_mutation]
        valid_mutation_probs = mutation_probs[valid_for_mutation]

        # Sample present edge to remove (only from valid graphs)
        valid_present_mask = (valid_pop == 1).float()
        remove_probs = valid_mutation_probs * valid_present_mask
        remove_probs_sum = remove_probs.sum(dim=-1, keepdim=True)
        remove_probs = remove_probs / remove_probs_sum  # More efficient than /=
        edges_to_remove = torch.multinomial(remove_probs, num_samples=1, replacement=False)

        # Sample absent edge to add (only from valid graphs)
        valid_absent_mask = (valid_pop == 0).float()
        add_probs = valid_mutation_probs * valid_absent_mask
        add_probs_sum = add_probs.sum(dim=-1, keepdim=True)
        add_probs = add_probs / add_probs_sum  # More efficient than /=
        edges_to_add = torch.multinomial(add_probs, num_samples=1, replacement=False)

        # Apply mutations back to the original population
        valid_indices = torch.where(valid_for_mutation)[0]
        pop[valid_indices, edges_to_add.squeeze()] = 1
        pop[valid_indices, edges_to_remove.squeeze()] = 0

        pop_fitness = fitness(pop).to(device)

        # Track mutation effectiveness BEFORE setting invalid ones to -inf
        if valid_for_mutation.any():
            valid_before = pop_fitness_before[valid_for_mutation]
            valid_after = pop_fitness[valid_for_mutation]
            improvements = (valid_after > valid_before).sum().item()
            total_valid = valid_for_mutation.sum().item()

            # Debug: print first few iterations
            if idx < 3:
                print(f"Iteration {idx}: {improvements}/{total_valid} improvements")

            total_mutations += total_valid
            successful_mutations += improvements

            # Calculate per-iteration success rate
            iter_success_rate = (improvements / total_valid * 100) if total_valid > 0 else 0
        else:
            iter_success_rate = 0

        # Set fitness to -inf for graphs that couldn't be mutated
        pop_fitness[~valid_for_mutation] = float("-inf")

        _, new_elite_indices = torch.topk(pop_fitness, top_old_elite)
        new_elite = pop[new_elite_indices].clone()

        # Update population for next iteration
        # Keep the best individuals from both current and previous generations
        components = [new_elite]  # Keep the best elites from current generation

        # Add previous generation elites if we have them
        if prev_elites.shape[0] > 0:
            components.append(prev_elites)

        # Fill the rest with random graphs
        remaining_size = pop_size - new_elite.shape[0] - prev_elites.shape[0]
        if remaining_size > 0:
            components.append(gen_random_reg(remaining_size, edges, problem.k))

        pop = torch.cat(components, dim=0)

        # Update previous elites for next iteration
        prev_elites = new_elite.clone()

        max_fitness = pop_fitness.max().item()
        avg_fitness = pop_fitness.mean().item()

        # Calculate mutation success rate
        success_rate = (successful_mutations / total_mutations * 100) if total_mutations > 0 else 0

        pbar.set_postfix({
            "max_fitness": f"{max_fitness:.4f}",
            "avg_fitness": f"{avg_fitness:.4f}",
            "valid_mutations": f"{valid_for_mutation.sum().item()}/{pop_size}",
            "iter_success": f"{iter_success_rate:.1f}%",
            "total_success": f"{success_rate:.1f}%",
        })


def main():
    iterations = 10_000
    problem = StronglyRegularGraphs.peterson_graph_problem()
    # Increased batch sizes for better GPU utilization
    guided_search_loop(200, 3000, 2000, iterations, random_swap_mutation, problem)


if __name__ == "__main__":
    main()
