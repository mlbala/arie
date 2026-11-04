# S02: Remove dead code listed in SPEC §12

**Needs:** S00 · **Size:** S

## Goal

A smaller codebase that matches its docs, so later sessions read less and nothing misleads them.

## Read first

- SPEC §12 only.
- Before deleting anything, `grep -rn <symbol>` across `core api worker plugins contract_tests`. Delete only code with no caller outside its own tests.

## Change

- Delete `QoSPriorityGate` and `HostRateLimiter` from `core/arie_core/sched/__init__.py` and their tests, unless grep finds a real caller.
- Delete `StartupReconciler`, `IngestionPipeline.index_document`, `run` and `process_next` from `core/arie_core/pipeline/pipeline.py` if unused, and their export in `pipeline/__init__.py`. `chunk_text` also appears in `reader.py` and `verifier.py`: check whether those are the same function before touching it.
- `runtime_qos` keys `user_initiated_override_concurrency`, `background_pdf_stream_concurrency`, `arxiv_rate_limit_delay_sec`, `use_uvloop`: keep accepting them in `ArieConfig` so existing `arie.yaml` copies (including the Docker volume's) still load, log one deprecation warning when present, and remove them from the repo's `arie.yaml` and the README.
- README: fix the "four priority lanes" process-model text and the "Keyword search (SQLite FTS5)" feature line to describe what the code does.

## Don't

- Touch SQL migrations, `search_bm25` or the `chunks_fts` table. Mark them in SPEC §12 as "kept for paged retrieval".
- Rewrite ADR 0002; SPEC already notes the `get_table` gap.

## Tests

- An `arie.yaml` with the old keys still loads and warns once.

## Done when

- Grep finds no references to deleted symbols. `make check` passes.
- SPEC §12 rows say what was removed or kept. PROGRESS records the number of lines removed.
