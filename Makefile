.PHONY: check lint types boundaries test-unit test-contracts eval-llm

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

# Uses the configured provider and fixed synthetic source-of-truth cases.
eval-llm:
	uv run python -m arie_api.eval_answers
