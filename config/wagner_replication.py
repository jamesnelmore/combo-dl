import math

import torch

from deep_cross_entropy import WagnerDeepCrossEntropy
from wagner_model import WagnerModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
n = 19
goal_score = math.sqrt(n - 1) + 1
print(f"Goal score (sqrt({n - 1}) + 1): {goal_score:.6f}")
print(f"Searching for graphs with eigenvalue + matching < {goal_score:.6f}")
model = WagnerModel(n)

# Track if we found a solution
found_solution = False


def progress_callback(iteration: int, metrics: dict):
    nonlocal found_solution
    if metrics["best_score"] < goal_score and not found_solution:
        print(f"\n🎉 SOLUTION FOUND at iteration {iteration}!")
        print(f"Best score: {metrics['best_score']:.6f} < {goal_score:.6f}")
        found_solution = True


# Create optimizer instance
optimizer = WagnerDeepCrossEntropy(
    model=model,
    batch_size=4096,
    iterations=15_000,  # Increased iterations
    learning_rate=0.0001,
    elite_proportion=0.1,
    device=device,
)

# Run optimization with early stopping
results = optimizer.optimize(progress_callback=progress_callback, goal_score=goal_score)

print("\n=== FINAL RESULTS ===")
print(f"Best score achieved: {results['best_score']:.6f}")
print(f"Goal score: {goal_score:.6f}")
print(f"Success: {'YES' if results['best_score'] < goal_score else 'NO'}")
print(f"Early stopped: {'YES' if results.get('early_stopped', False) else 'NO'}")
print(f"Iterations completed: {results.get('iterations_completed', 0)}")

if results["best_construction"] is not None:
    print(f"Best construction shape: {results['best_construction'].shape}")
    print(f"Number of edges in best graph: {results['best_construction'].sum().item()}")

    # Analyze the best construction
    model.eval()
    with torch.no_grad():
        adj_matrix = model._edge_vector_to_adjacency(results["best_construction"])
        largest_eigenvalue = model._compute_largest_eigenvalue(adj_matrix)
        matching_number = model._compute_maximum_matching(adj_matrix)

    print(f"Largest eigenvalue: {largest_eigenvalue:.6f}")
    print(f"Matching number: {matching_number:.6f}")
    print(f"Sum: {largest_eigenvalue + matching_number:.6f}")
else:
    print("No construction found")
