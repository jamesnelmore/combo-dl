.PHONY: help check-types clean format install-dev lint lint-fix strip-notebooks strip-notebooks-dry

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

check-types:  ## Run PyRight type checking
	uv run pyright

check-notebooks:  ## Run PyRight type checking on notebooks
	uv run nbqa pyright notebooks/

check-all:  ## Run type checking on all code and notebooks
	uv run pyright
	uv run nbqa pyright notebooks/

clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pyright" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	uv clean
	ruff clean

format:  ## Run Ruff formatting
	uv run ruff format .

install-dev:  ## Install development dependencies and editable package
	uv sync --extra dev
	uv pip install -e .

lint:  ## Run Ruff linting (check only)
	uv run ruff check .

lint-fix:  ## Run Ruff linting with auto-fix
	uv run ruff check --fix .

strip-notebooks:  ## Strip outputs from all notebooks
	@echo "Stripping notebook outputs..."
	@for notebook in notebooks/*.ipynb; do \
		if [ -f "$$notebook" ]; then \
			echo "Processing $$notebook..."; \
			uv run nbstripout "$$notebook" || echo "Skipping invalid notebook: $$notebook"; \
		fi; \
	done

strip-notebooks-dry:  ## Show which notebooks would be stripped (dry run)
	@echo "Checking which notebooks would be stripped..."
	@for notebook in notebooks/*.ipynb; do \
		if [ -f "$$notebook" ]; then \
			uv run nbstripout --dry-run "$$notebook" || echo "Invalid notebook: $$notebook"; \
		fi; \
	done