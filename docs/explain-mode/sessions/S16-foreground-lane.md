# S16 (optional): Foreground lane and answer workers

**Needs:** S03 · **Size:** M

## Goal

A question never competes with background work for CPU, and strict mode's reranker stops starving the claim check (SPEC §13).

## Read first

- SPEC §9 and §13.2–13.4
- `worker/arie_worker/pool.py` (`SupervisedPool.submit`), `api/arie_api/models.py`
- Where sync, embedding and guide tasks submit work (grep `submit(`)

## Change

- A `ForegroundGate` (asyncio): while any question is answering or streaming, background submitters (PDF parse, embedding, guide builds) wait before starting new work. Running work is not killed.
- Default `shared_model_worker: false` when the host has 2 CPUs or fewer.
- Skip reranking when the answer worker is still busy with an abandoned call (SPEC §13.4 option 2), and record `rerank_skipped_busy` in the trace.
- ADR via `arie-adr` amending ADR 0009.

## Tests

- The gate delays a background submit while a question runs and releases it after; reranking is skipped when the worker is busy.

## Done when

- `make check` passes. A strict-mode eval with `--compare-reranker` shows fewer `entailment_timeout` fallbacks than the S00 baseline (record both in PROGRESS).
