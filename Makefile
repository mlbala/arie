.PHONY: check lint boundaries types test test-unit test-api test-contracts web-check eval-llm eval-paper eval-qasper

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

# Evaluates answers on one ingested paper, its own text as the ground truth:
# make eval-paper PAPER=2610.40325 [ARGS="--compare-reranker --regenerate"]
eval-paper:
	uv run python -m arie_api.eval_paper $(PAPER) $(ARGS)

# Evaluates answers on the first N papers of Qasper's dev split (imports them from arXiv):
# make eval-qasper N=30 (ARGS="--mode explain" scores explain mode)
N ?= 5
eval-qasper:
	uv run python -m arie_api.eval_qasper $(N) $(ARGS)
