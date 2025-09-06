# Thesis: Graph Optimization using Deep Learning

This project implements graph optimization algorithms using deep learning techniques, with a focus on eigenvalue problems and matching theory.

## Quick Start

```bash
# Install dependencies
uv sync

# Install development tools
uv sync --extra dev

# Run experiments
python main.py

# Run development checks
make lint-all
```

## Development

This project uses modern Python development tools:

- **Ruff**: Fast linting and formatting (replaces black, isort, flake8)
- **PyRight**: Static type checking
- **uv**: Fast Python package manager
- **Hydra**: Configuration management

Development tools are configured in `pyproject.toml` and can be run via the `Makefile`.

## Project Structure

```
├── src/thesis/          
│   ├── algorithms/        # Optimization algorithms
│   ├── experiment_logger/ # Logging utilities
│   ├── models/            # Neural network models
│   ├── problems/          # Problem definitions
│   └── main.py            # Main entry point
├── config/                # Hydra experiment configuration files
├── tests/                 # Test files
```

## Configuration

The project uses Hydra for configuration management. Default configs are in `config/`, and you can override parameters:

```bash
# Use different model size
python main.py model.n=25

# Use different algorithm
python main.py algorithm=ppo

# Run multiple experiments
python main.py --multirun seed=1,2,3
```
