# 0004: CPU-heavy work runs in recycled worker processes

**Status:** Accepted

## Context

PDF parsing (PyMuPDF and Docling), ONNX embeddings and the entailment model are
CPU-bound, can leak memory, and MuPDF is not thread-safe. Running them in the API
process or in threads would stall the event loop behind chat and could crash the API.

## Decision

- `arie_worker.pool.SupervisedPool` runs this work in spawn-started processes that
  recycle after 20 tasks by default (`max_tasks_per_child`; parsers can lower it with
  `plugins.parser.tasks_per_worker`). A call that times out, is
  cancelled, or lets a worker exceed 1.5 GB RSS kills the pool, which is rebuilt on the
  next call.
- Extraction, embedding and entailment each have their own pool, so a slow PDF never
  delays an answer's Tier 3 check.
- `asyncio.to_thread` is never used for MuPDF. Background loops run in an
  `asyncio.TaskGroup` through `supervised()`, never with a bare `create_task`, so one
  failing paper cannot cancel a batch.
- The event-loop lag monitor runs in the API's TaskGroup and warns when callbacks run
  later than `runtime_qos.slow_callback_threshold_ms`.

## Consequences

- Process start and model load are paid once per worker, not per call. A killed pool
  pays them again.
- Docling's layout and OCR models are slower and heavier than PyMuPDF4LLM. One
  conversion of a 15-page paper took 73-165 s and peaked at 1.38-1.53 GB, at the cap.
  Select it with a raised `timeout_sec` and `tasks_per_worker: 1`. Some papers still
  fail with a memory error unless the cap changes.
