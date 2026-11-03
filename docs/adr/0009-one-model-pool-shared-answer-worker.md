# 0009: One generic model pool, with the answer models in one worker

**Status:** Accepted; amends [ADR 0007](0007-cross-encoder-reranker.md) (the reranker no longer has its own pool)

## Context

Each local model slot had its own pool class (`EmbeddingPool`, `EntailmentPool`,
`RerankPool`), each about 60 lines and nearly identical, plus hand-written wiring in
`runtime.py`, `routes.py`, `monitor.py` and `eval_answers.py` for warm-up, retries,
status and shutdown. Adding the reranker touched all of them. The claim checker and
the reranker also ran in separate processes, so each paid the Python and ONNX Runtime
overhead.

Memory of the two answer-model workers, measured after scoring 12 claim pairs and 24
passages of ~500 tokens (the worst case per answer), RSS:

| Setup | macOS host (5 runs) | Linux, Docker, 2 vCPU (3 runs) |
| --- | --- | --- |
| Two workers, ONNX memory arena on | 444-639 MB | 1,233-1,382 MB |
| One shared worker, arena on | 497-801 MB | 1,212-1,225 MB |
| One shared worker, arena off | not measured | 685-738 MB |

macOS RSS varied too much to compare. On Linux, sharing saves about 10%. Most of the
memory is ONNX Runtime's CPU memory arena, which keeps its peak allocation: turning it
off saved about 40% more. Warm latency on the same stress inputs was unchanged
(entailment 20.1 s with the arena vs 19.7 s without, for 12 long pairs on 2 vCPUs;
reranking 9.7 s vs 9.5 s).

The first real per-paper evaluation also showed a cost of sharing. With the CPU busy,
5 of 8 rerank calls passed their 3 s budget; each timeout killed the worker, and in a
shared worker that unloaded the claim checker as well.

## Decision

- `arie_worker.models.ModelPool` serves any model plugin: one worker function loads
  `(slot, name, settings)` once per process and calls the requested method.
  `LocalModel` binds one configured plugin to a pool and checks result lengths. The
  three per-slot pool classes are removed.
- `api/arie_api/models.py` holds one `AnswerSlot` table (entailment, reranker). Config
  parsing, warm-up, 5-minute retries, status for the health pane and shutdown come
  from it, so routes and the monitor no longer name each model.
- `runtime_qos.shared_model_worker` (default `true`) runs every answer model in one
  worker; `false` gives each its own. The embedder keeps its own background-priority
  worker, because it runs during syncs at `nice 10`.
- Model pools set `keep_on_cancel`: when a caller stops waiting (the reranker's 3 s
  budget), the call finishes in the worker and the next call queues behind it. Only
  running past `kill_after`, a crash or the memory cap kills the worker.
- Both ONNX plugins create their sessions with the CPU memory arena off;
  `memory_arena: true` turns it back on.

## Consequences

- A new local model slot is one `AnswerSlot` entry plus the harness hook that uses its
  scores; it needs no pool class and no route or monitor edits. It still needs a
  `Protocol` in `plugin_api`, since the core decides how scores are used.
- A crash or memory breach in one answer model restarts both, and the next answer
  pays the model load (about 1-2 s). The answer path already falls back when either
  model is unavailable.
- Calls to models sharing a worker are serialized. Reranking and the claim check run
  at different stages of an answer, so one question does not wait on itself;
  concurrent questions can queue behind each other's model calls.
