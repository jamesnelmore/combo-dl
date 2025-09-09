![Project Status](https://img.shields.io/badge/status-early%20development-yellow) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.13+-blue)

# Graph Optimization Via Deep Learning
> **Warning:** This project is in early development. Features, APIs, and results may change without notice.

This project uses deep reinforcement learning to solve combinatorial optimization problems in graph theory.
It currently focuses on finding novel [Strongly Regular Graphs](https://en.wikipedia.org/wiki/Strongly_regular_graph),
but the methods and code used are intentionally general and adaptable to a wide variety of other combinatorial optimization problems.

## Contributions
This project is part of an undergraduate honors thesis and is currently not accepting third-party contributions.
This will likely change after thesis submission.

## Project Structure
```
├── config/                # Hydra experiment configuration files
├── src/thesis/          
│   ├── algorithms/        # Optimization algorithms
│   ├── experiment_logger/ # Logging utilities
│   ├── models/            # Neural network models
│   ├── problems/          # Problem definitions
│   └── main.py            # Main entry point
├── tests/                 # Test files
```
Algorithms, models, and problem statements are implemented in their respective directories.
Experiments are configured and run with [Hydra](https://hydra.cc) for reproducibility and logged with [Weights and Biases](https://wandb.ai/site/).
