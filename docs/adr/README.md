# Architecture decision records

Short records of decisions that shape ARIE. Each states the context, the decision and
its consequences. Add a new numbered file when a decision changes; mark the old one
superseded instead of rewriting it.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-microkernel-plugin-slots.md) | Microkernel core with pluggable adapter slots | Accepted |
| [0002](0002-single-paper-isolation.md) | Single-paper isolation by lexical closure and post-filtering | Accepted |
| [0003](0003-quote-verifier-ladder.md) | Three-tier quote verifier with a local entailment model | Accepted, amended by 0010 |
| [0004](0004-process-isolated-workers.md) | CPU-heavy work runs in recycled worker processes | Accepted, amended by 0005 |
| [0005](0005-per-parser-memory-limit.md) | The PDF parser may raise its worker memory limit | Accepted |
| [0006](0006-background-cpu-priority.md) | Background workers run at lower CPU priority | Accepted |
| [0007](0007-cross-encoder-reranker.md) | A local cross-encoder reranker picks each answer's passages | Accepted, amended by 0009 |
| [0008](0008-model-led-planning-and-conditional-review.md) | The planner decides the question type, and review runs when it adds a check | Accepted, amended by 0010 |
| [0009](0009-one-model-pool-shared-answer-worker.md) | One generic model pool, with the answer models in one worker | Accepted |
| [0010](0010-explain-mode-paper-in-the-prompt.md) | Explain mode puts the whole paper in the prompt and checks citations after streaming | Accepted, amended by 0011 |
| [0011](0011-no-local-grounding-model.md) | Explain mode adds no local grounding model; the LLM judge checks every cited sentence | Accepted |
