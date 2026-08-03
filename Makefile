.PHONY: test lint typecheck check clean

test:
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check src tests environments

typecheck:
	uv run --extra dev mypy src

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
