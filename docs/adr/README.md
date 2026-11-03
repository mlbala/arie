# Architecture decision records

Short records of decisions that shape ARIE. Each states the context, the decision and
its consequences. Add a new numbered file when a decision changes; mark the old one
superseded instead of rewriting it.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-microkernel-plugin-slots.md) | Microkernel core with pluggable adapter slots | Accepted |
| [0002](0002-single-paper-isolation.md) | Single-paper isolation by lexical closure and post-filtering | Accepted |
| [0003](0003-quote-verifier-ladder.md) | Three-tier quote verifier with a local entailment model | Accepted |
| [0004](0004-process-isolated-workers.md) | CPU-heavy work runs in recycled worker processes | Accepted, amended by 0005 |
| [0005](0005-per-parser-memory-limit.md) | The PDF parser may raise its worker memory limit | Accepted |
| [0006](0006-background-cpu-priority.md) | Background workers run at lower CPU priority | Accepted |
| [0007](0007-cross-encoder-reranker.md) | A local cross-encoder reranker picks each answer's passages | Accepted |
| [0008](0008-model-led-planning-and-conditional-review.md) | The planner decides the question type, and review runs when it adds a check | Accepted |
