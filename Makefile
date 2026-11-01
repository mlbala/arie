.PHONY: check lint types boundaries test-unit test-contracts

check: lint boundaries types test-unit

lint:
	uv run ruff check .

boundaries:
	uv run lint-imports

types:
	uv run pyright

test-unit:
	uv run pytest core/ -q --tb=short

test-contracts:
	uv run pytest contract_tests/ -q --tb=short
