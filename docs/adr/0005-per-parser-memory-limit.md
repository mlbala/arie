# 0005: The PDF parser may raise its worker memory limit

**Status:** Accepted. Amends the 1.5 GB cap in [ADR 0004](0004-process-isolated-workers.md).

## Context

ADR 0004 kills any worker above 1.5 GB RSS. One Docling conversion of a 15-page paper,
measured four ways with CPU-only PyTorch, took 73-165 s and peaked at 1.38-1.53 GB.
Batch sizes, thread counts and the table model did not lower the peak, because most of
it is PyTorch and the layout, table and OCR model weights. In container runs the same
paper sometimes finished and sometimes was killed, so Docling was unreliable at the cap.

## Decision

- `SupervisedPool` takes `max_rss_mb`, default 1536 (1.5 GB).
- Only the PDF parser pool reads an override, from `plugins.parser.max_rss_mb`, next to
  `timeout_sec` and `tasks_per_worker`. `ExtractionLimits` validates all three at
  startup: `max_rss_mb` must be 1536-8192, so the cap can be raised but not lowered.
- Docling is selected with `max_rss_mb: 2560`, `timeout_sec: 300`,
  `tasks_per_worker: 1` and one utility worker process.
- The embedding and entailment pools keep the 1.5 GB default with no override.

## Consequences

- Docling has about 1 GB of headroom over its measured peak and no longer fails on
  memory alone. A paper that still exceeds 2.5 GB fails by itself; the sync continues.
- A Docling worker can hold 2.5 GB, so each utility worker process needs that much free
  memory on the host or in the Docker VM.
- Other parsers, the embedder and the claim checker keep the original limit, so a leak
  there is still caught at 1.5 GB.
