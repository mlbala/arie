# 0012: Questions go first: a foreground gate, split answer workers on small hosts, and no reranking behind an abandoned call

**Status:** Accepted; amends [ADR 0009](0009-one-model-pool-shared-answer-worker.md) (the shared worker is no longer the default on hosts with 2 CPUs or fewer, and a busy worker skips reranking)

## Context

SPEC section 13 records the symptom: in the Docker evaluation (2 vCPUs) most answers fell
back to source excerpts and the reviewer never ran. Reading the code shows three ways
background work and the reranker take time from the claim check:

- PDF parsing, vector indexing and guide builds ran while questions were answered. PDF
  and embedding workers run at `nice 10`, which only helps when the CPU is contended and
  does not stop them from starting. Only guide builds waited for answers (S14).
- ADR 0009 measured 9.7 s for reranking and 20.1 s for 12 long claim pairs on 2 vCPUs,
  against a 3 s rerank budget and a 4 s Tier 3 cap. When the reranker passes its budget
  the caller leaves, the call keeps running (`keep_on_cancel`), and the claim check queues
  behind it in the shared worker with its 4 s cap already ticking.
- An abandoned claim check queues the next question's reranking the same way.

## Decision

- `arie_core.sched.ForegroundGate`: every answer or stream holds it (`hold()`), and
  background submitters call `wait_quiet()` before they start new work. These are PDF
  parsing (`SyncJobs.parse_pdf`), each embedding batch and the embedder warm-up in the
  vector indexer, and each guide build. Work that is running is never interrupted. A
  question's own query embedding does not wait: it runs inside the question.
- `runtime_qos.shared_model_worker` is `true`, `false` or unset. Unset means one shared
  worker above 2 CPUs (`os.process_cpu_count()`) and one worker per answer model at 2 or
  fewer. An explicit value always wins.
- `SupervisedPool.abandoned_calls` counts calls whose caller gave up and that the worker
  has not finished. `answer_question(rerank_busy=...)` skips reranking while the
  reranker's pool has one, and `AnswerTrace.rerank_skipped_busy` records it. The answer
  keeps the keyword and vector choice, as it does when the reranker times out.

## Consequences

- A question no longer shares the CPU with a parse, an embedding batch or a guide build
  that starts after it. One already running still does; `nice 10` covers that.
- Background work waits as long as questions keep arriving. An evaluation of many
  questions in a row holds the gate for its whole run, and a sync or guide build resumes
  after it. A paper imported during a long chat is parsed when the chat is quiet.
- A small host pays about 10% more memory for a second worker (ADR 0009) and loads each
  model in its own process. Split workers also mean a crash or memory breach in one model
  no longer restarts the other.
- A skipped rerank lowers retrieval quality for that answer. It happens only after an
  abandoned call, which on a healthy host is rare; the trace flag counts it. Skipping
  does not cancel the abandoned call.
- `os.process_cpu_count()` sees CPU affinity, not a Docker `--cpus` quota. On such a host
  set `shared_model_worker` explicitly.
- Undo: set `runtime_qos.shared_model_worker: true`; pass no `rerank_busy`; remove the
  `wait_quiet()` calls.
