# Development Setup Guide

This document explains how to use the Ruff and PyRight tools configured for this project.

## Quick Start

```bash
# Install development dependencies
uv sync --extra dev

# Run all checks at once
make lint-all

# Or run individual tools
make lint        # Ruff linting (check only)
make format      # Ruff formatting
make check-types # PyRight type checking
make lint-fix    # Ruff linting with auto-fix
```

## Tool Overview

### Ruff
**Purpose**: Fast Python linter and formatter (replaces black, isort, flake8, and more)

**Key Features**:
- Import sorting (like isort)
- Code formatting (like black)
- Comprehensive linting rules
- Auto-fixing many issues
- Extremely fast

### PyRight
**Purpose**: Static type checker for Python

**Key Features**:
- Strict type checking
- Excellent IDE integration
- Fast incremental checking
- Comprehensive type analysis

## Detailed Usage

### Ruff Commands

```bash
# Check for linting issues (no changes)
uv run ruff check .

# Check specific files
uv run ruff check main.py algorithms/

# Auto-fix issues where possible
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check only import sorting
uv run ruff check . --select I

# Check only specific rule categories
uv run ruff check . --select E,W  # pycodestyle errors and warnings
```

### PyRight Commands

```bash
# Type check entire project
uv run pyright

# Type check specific files
uv run pyright main.py

# Get statistics
uv run pyright --stats

# Watch mode (re-check on file changes)
uv run pyright --watch

# Generate type stubs
uv run pyright --createstub <package_name>
```

### Make Commands (Convenience)

```bash
make help         # Show all available commands
make install-dev  # Install development dependencies
make lint         # Run Ruff linting (check only)
make format       # Run Ruff formatting
make check-types  # Run PyRight type checking
make lint-fix     # Run Ruff linting with auto-fix
make lint-all     # Run all linting, formatting, and type checking
make clean        # Clean up cache and temporary files
```

## Cursor/VSCode Integration

The project includes `.vscode/settings.json` with the following features:

### Automatic Actions
- **Format on save**: Code is automatically formatted when you save a file
- **Fix imports on save**: Import sorting and unused import removal
- **Real-time linting**: Issues shown as you type

### Extensions
The configuration recommends these extensions:
- `charliermarsh.ruff` - Ruff linter and formatter
- `ms-python.python` - Python support
- `ms-python.vscode-pylance` - Python type checking (uses PyRight)

### Key Features
- Line length ruler at 88 characters
- Automatic final newline insertion
- Whitespace trimming
- Real-time type checking
- Hover documentation
- Auto-completion with type information

## Configuration Details

### Ruff Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
src = [".", "algorithms", "models", "problems", "experiment_logger"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort (import sorting)
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RET", # flake8-return
    "ARG", # flake8-unused-arguments
    "PTH", # flake8-use-pathlib
    "ERA", # eradicate (remove commented code)
    "PL",  # pylint
    "RUF", # ruff-specific rules
]
```

### PyRight Configuration (pyproject.toml)

```toml
[tool.pyright]
include = [".", "algorithms", "models", "problems", "experiment_logger"]
exclude = ["**/__pycache__", "**/old", "**/.venv", "**/venv"]
pythonVersion = "3.11"
typeCheckingMode = "strict"
```

## Common Workflows

### Before Committing
```bash
# Run all checks and auto-fix issues
make lint-fix
make format
make check-types

# Or run everything at once
make lint-all
```

### Setting Up New Files
1. Create your Python file
2. Save it (Cursor will auto-format)
3. Add type hints as you write functions
4. Run `make check-types` to verify types

### Fixing Type Issues
1. Run `uv run pyright <file>` to see specific issues
2. Add type hints to function signatures
3. Use type annotations for complex variables
4. Import types from `typing` module as needed

### Ignoring Specific Issues

#### Ruff
```python
# noqa: E501  # Ignore line too long
# ruff: noqa  # Ignore entire file
```

#### PyRight
```python
# type: ignore  # Ignore type checking on this line
# pyright: ignore[reportUnknownParameterType]  # Ignore specific rule
```

## Troubleshooting

### Ruff Issues
- **"Command not found"**: Run `uv sync --extra dev` to install
- **"No fixes available"**: Some issues require manual fixing
- **"Line too long"**: Break long lines or use parentheses for line continuation

### PyRight Issues
- **"Module not found"**: Check your import paths and PYTHONPATH
- **"Unknown type"**: Add type hints or import appropriate types
- **"Partially unknown"**: Make return types more specific

### IDE Integration Issues
- **Not formatting on save**: Check that the Ruff extension is installed and enabled
- **No type checking**: Ensure Pylance extension is installed
- **Wrong Python interpreter**: Set the correct interpreter in VSCode/Cursor

## Best Practices

### Type Hints
```python
# Function signatures (always)
def process_data(items: list[str], count: int) -> dict[str, int]:
    return {}

# Variables (when type is unclear)
results: list[tuple[str, float]] = []

# Use union syntax (preferred in this project)
def find_item(name: str) -> Item | None:
    return None
```

### Import Organization
Ruff automatically sorts imports in this order:
1. Standard library imports
2. Third-party imports  
3. Local imports

```python
# Standard library
import math
import sys
from pathlib import Path

# Third-party
import torch
import numpy as np
from tqdm import tqdm

# Local
from .algorithms import BaseAlgorithm
from problems.base_problem import BaseProblem
```

### Code Style
- Use 88-character line length
- Prefer double quotes for strings
- Use trailing commas in multi-line structures
- Remove unused imports and variables
- Follow PEP 8 naming conventions
