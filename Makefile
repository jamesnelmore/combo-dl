.PHONY: help lint format check-types lint-fix lint-all install-dev clean run run-slurm run-slurm-with-args

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

run:  ## Run experiment with config name (make run CONFIG=wagner_corollary_2_1)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run CONFIG=experiment_name"; \
		echo "Available configs:"; \
		ls config/*.yaml | grep -v base_config.yaml | sed 's/config\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m combo_dl.main --config-name $(CONFIG)

run-with-args:  ## Run experiment with additional args (make run-with-args CONFIG=wagner_corollary_2_1 ARGS="seed=999 algorithm.iterations=100")
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run-with-args CONFIG=experiment_name ARGS=\"key=value\""; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m combo_dl.main --config-name $(CONFIG) $(ARGS)

run-slurm:  ## Submit experiment to Slurm (make run-slurm CONFIG=palay_large_model)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run-slurm CONFIG=experiment_name"; \
		echo "Available configs:"; \
		ls config/*.yaml | grep -v base_config.yaml | sed 's/config\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m combo_dl.main --config-name $(CONFIG)

run-slurm-with-args:  ## Submit experiment to Slurm with additional args (make run-slurm-with-args CONFIG=palay_large_model ARGS="seed=999")
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Usage: make run-slurm-with-args CONFIG=experiment_name ARGS=\"key=value\""; \
		exit 1; \
	fi
	PYTHONPATH=./src uv run python -m combo_dl.main --config-name $(CONFIG) $(ARGS)

list-configs:  ## List available experiment configurations
	@echo "Available experiment configurations:"
	@ls config/*.yaml | grep -v base_config.yaml | sed 's/config\///g' | sed 's/\.yaml//g' | sed 's/^/  - /'

ci:  ## Run all CI checks locally (lint, format, type-check, test)
	@echo "Running CI checks..."
	$(MAKE) lint-fix
	$(MAKE) check-types
	$(MAKE) test
	@echo "All CI checks passed!"

test:  ## Run tests
	uv run pytest tests/ -v


clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pyright" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	uv clean
	ruff clean