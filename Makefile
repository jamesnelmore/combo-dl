.PHONY: help lint format check-types lint-fix lint-all install-dev clean run

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

run:  ## Run experiment with config name (make run CONFIG=wagner_corollary_2_1)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run CONFIG=experiment_name"; \
		echo "Available configs:"; \
		ls config/*.yaml | grep -v base_config.yaml | sed 's/config\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m thesis.main --config-name $(CONFIG)

run-with-args:  ## Run experiment with additional args (make run-with-args CONFIG=wagner_corollary_2_1 ARGS="seed=999 algorithm.iterations=100")
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run-with-args CONFIG=experiment_name ARGS=\"key=value\""; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m thesis.main --config-name $(CONFIG) $(ARGS)

list-configs:  ## List available experiment configurations
	@echo "Available experiment configurations:"
	@ls config/*.yaml | grep -v base_config.yaml | sed 's/config\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'

clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pyright" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	uv clean
	ruff clean