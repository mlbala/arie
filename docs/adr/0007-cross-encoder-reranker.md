# 0007: A local cross-encoder reranker picks each answer's passages

**Status:** Accepted

## Context

Answers were built from passages chosen by keyword overlap, reciprocal-rank fusion
with vector hits, and hand-tuned section bonuses. Some of those bonuses named terms
from one paper ("attention", "parallelizable"). Wording that missed the keyword rules,
such as "describe this paper", reached the writer with no passages at all. The
`reranker` slot was declared in `plugin_api` but unused.

## Decision

- The `Reranker` slot scores passages: `score(ctx, query, passages) -> list[float]`,
  one score per passage, in input order.
- The default plugin, `onnx-cross-encoder`, runs `cross-encoder/ms-marco-MiniLM-L6-v2`
  (~90 MB) with ONNX Runtime. It refuses to run outside `arie-worker`.
- `RerankPool` is a one-process `SupervisedPool` with the default limits (20 tasks,
  60 s, 1.5 GB RSS) and normal CPU priority, because every answer waits on it.
- Reading collects up to 24 paper-scoped candidates: keyword hits, up to 12 vector
  hits, and for whole-paper questions the opening passage of each top-level section.
  The reranker orders them by relevance to the resolved question plus the planner's
  queries. The same per-section quota, 10-row and 8,500-byte caps then pick the rows.
- Reranking gets at most 3 s, and runs only while 14 s of the 25 s deadline remain.
  An error, a malformed result or a timeout keeps the keyword/vector reading.
- Reranking is not a model-visible tool call, so K <= 3 is unchanged. Every candidate
  is re-checked against the bound `paper_id` (I1).
- The paper-specific section bonuses are replaced with generic section names.

## Consequences

- One more worker process: measured at 440 MB RSS after scoring 24 passages of ~500
  tokens (the entailment worker measured 307 MB the same way). Both stay under the
  1.5 GB default limit.
- Default `threads: 2`. Measured on an 8-core Mac, 24 passages of ~500 tokens take
  3.7 s on one thread and 1.9 s on two. Two threads keep the worst case inside the
  3 s budget at the cost of about two cores for under two seconds per answer.
- A timeout kills and rebuilds the pool, so the next answer pays the 1.3 s model load.
- Without the reranker (`use: none`, still loading, or failed), answers use the
  keyword/vector order as before.
