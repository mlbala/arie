# ARIE: agent guide

ARIE syncs research papers, extracts their text and answers questions about one paper
at a time, with every claim tied to an exact quote and page offsets. This file is for
any coding agent. `docs/SPEC.md` describes what the code does; `docs/adr/` says why.

## Layout

| Path | Holds |
| --- | --- |
| `core/arie_core/` | Vendor-free kernel: `harness/` (answer path), `pipeline/` (chunking), `plugin_api/` (slot protocols), `contracts/dtos.py`, `registry/` (config + plugin discovery), `budget/`, `sched/` |
| `api/arie_api/` | FastAPI app (`routes.py`), runtime layers in `runtime/`, answer models (`models.py`), evals (`paper_eval.py`, `eval_paper.py`, `eval_qasper.py`, `eval_answers.py`) |
| `worker/arie_worker/` | `SupervisedPool` and spawned worker processes |
| `plugins/<slot>-<name>/arie_*/` | One package per plugin (LLM, embedder, parser, source, entailment, reranker, db, files, vectors) |
| `contract_tests/` | Slot compliance tests for every plugin, no network |
| `prompts/*.vN.yaml` | Versioned prompts |
| `web/src/` | React 19 + TanStack Query workspace; `api/generated.ts` is generated |
| `docs/explain-mode/` | Current work plan, session files and progress log |

## Commands

- `make check`: ruff, import boundaries, strict pyright, all Python tests (what CI runs)
- `make lint` · `make boundaries` · `make types` · `make test-unit` · `make test-api` · `make test-contracts`
- One test: `uv run pytest core/tests/test_x.py::test_name -q --tb=short`
- `make web-check`: oxlint, typecheck, vitest
- After any route or response-model change: `cd web && npm run generate:api`
- `make eval-paper PAPER=<id> ARGS="--compare-reranker"` runs the per-paper eval
- `make eval-qasper N=30` imports Qasper dev papers and runs the outside benchmark (network + LLM key)

## Invariants (never break; see SPEC section 1)

- **I1 Paper isolation.** `paper_id` is bound in Python, never in model input or tool schemas; every row is re-filtered by `paper_id`.
- **I2 No unverified claims.** Strict mode: every shown claim passes the verifier, or the answer abstains or falls back to excerpts. Explain mode (ADR 0010, once written) amends this: every sentence is cited or marked background, and every citation is checked.
- **I3 Bounded execution.** Every model call goes through `ModelSession` with a token and time budget.
- **I4 Clean ingestion.** Validated records, declared hosts only, size/type-checked PDFs, atomic writes.
- **I5 No vendor code in core.** `arie_core` imports no SDK or plugin; plugins import only `arie_core.plugin_api` and `arie_core.contracts`. `make boundaries` enforces it.

## Conventions

- Python 3.14, `uv`. Ruff line length 100. Pyright strict: no `Any` leaks, no ignores without a reason.
- Data types are frozen `msgspec.Struct`s. Async plugin calls take `ctx: Ctx` first.
- New plugins use `@plugin(slot=..., name=..., api=">=1,<2")` and get a contract test.
- Config keys are validated by `ArieConfig` (`core/arie_core/registry/manager.py`); unknown keys are rejected, so add new keys there.
- Never edit a released prompt file. Add `prompts/<name>.v<N+1>.yaml` and record the version on each answer.
- A decision that changes an invariant, a budget or a slot gets a new ADR (`docs/adr/NNNN-*.md`) and a row in `docs/adr/README.md`. Mark old ADRs amended; don't rewrite them.
- When behaviour changes, update the matching section of `docs/SPEC.md` in the same change.
- Tests: no network. Reuse the existing doubles (`ScriptedModel` in `core/tests/test_search_and_source_truth.py`, `FakePool` in `api/tests/test_answer_models.py`); move a double into a shared `conftest.py` when a second file needs it. Every change ships with a test.
- Secrets only in `.env`. Never commit `data/`, `*.db` or PDFs.

## Reading the code cheaply

- Search first (`grep -n`), then read only the lines you need. These files are long: `api/arie_api/routes.py`, `core/arie_core/harness/reader.py`, `core/arie_core/harness/answer.py`, `api/arie_api/runtime/sync.py`, `web/src/components/ChatWorkspace.tsx`.
- Read one SPEC section by its number, not the whole file.
- Don't open `data/`, `*.db`, `uv.lock`, `web/node_modules/` or `web/src/api/generated.ts` unless the task is about them.
- Run the narrowest test first; run `make check` once at the end.
