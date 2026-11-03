# 0006: Background workers run at lower CPU priority

**Status:** Accepted

## Context

The QoS lanes in ARIE are enforced with asyncio semaphores inside `arie-api`, but the
CPU work itself runs in worker processes the OS schedules as equals. In a 2-CPU Docker
VM, a sync's PDF extraction worker (PyMuPDF and its ONNX layout model) held about
170% CPU. A question asked at the same time ran its Tier 3 claim check on roughly one
core, so answers slowed down while a sync was running. `utility_worker_processes: 4`
also allowed more PDF workers than CPUs, each holding about 500 MB.

## Decision

- `SupervisedPool` takes `nice` (0-19) and applies it in each worker's initializer.
- PDF extraction and embedding pools use `BACKGROUND_NICE = 10`. The entailment pool
  and `arie-api` keep normal priority.
- `ExtractionPool` caps its processes to `os.process_cpu_count()`.

## Consequences

- Background work still uses every idle core, so CPU meters still read near 100% per
  core during a sync; under contention Linux gives a nice-0 process about nine times
  the CPU share of a nice-10 one, so answers keep their speed.
- A query embedding from a local embedder shares the embedding pool's background
  priority. It is one short batch and competes only with other background work.
- Fewer PDF workers on small machines lowers peak memory; more parallel imports than
  CPUs queue instead of running at once.
