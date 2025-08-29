.PHONY: help lint format check-types lint-fix lint-all install-dev clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install-dev:  ## Install development dependencies
	uv sync --extra dev

lint:  ## Run Ruff linting (check only)
	uv run ruff check .

format:  ## Run Ruff formatting
	uv run ruff format .

check-types:  ## Run PyRight type checking
	uv run pyright

lint-fix:  ## Run Ruff linting with auto-fix
	uv run ruff check --fix .

lint-all:  ## Run all linting, formatting, and type checking
	uv run python scripts.py

clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pyright" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
