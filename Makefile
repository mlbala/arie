.PHONY: check lint boundaries types test test-unit test-api test-contracts web-check eval-llm

# Everything CI runs for Python: lint, import boundaries, strict types, all test suites.
check: lint boundaries types test

lint:
	uv run ruff check .

boundaries:
	uv run lint-imports

types:
	uv run pyright

test: test-unit test-api test-contracts

test-unit:
	uv run pytest core/ -q --tb=short

test-api:
	uv run pytest api/tests -q --tb=short

# Slot compliance and per-plugin behaviour; no network. The real Tier 3 model test runs
# only when the model is cached (or ARIE_NLI_CACHE points at a Hugging Face cache).
test-contracts:
	uv run pytest contract_tests/ -q --tb=short

web-check:
	cd web && npm run lint && npm run typecheck && npm test

# Uses the configured provider and fixed synthetic source-of-truth cases.
eval-llm:
	uv run python -m arie_api.eval_answers
