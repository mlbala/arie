# 0001: Microkernel core with pluggable adapter slots

**Status:** Accepted

## Context

ARIE must run fully local (SQLite, local files, local embeddings) and also against
cloud services (Postgres, S3, hosted LLMs and vector stores). Vendor SDKs change
quickly, and research workflows need custom scrapers. If any of that lived in the core,
every provider change would touch the code that enforces safety and grounding.

## Decision

- `arie_core` holds only the ingestion state machine, safety gates, the Q&A harness
  and the verifier. It imports no vendor SDK and no plugin package (invariant I5,
  enforced by `lint-imports` in `make check`).
- Every external capability is a **slot** with a `Protocol` in
  `arie_core.plugin_api`: `llm`, `embedder`, `vector_store`, `database`,
  `file_storage`, `parser`, `source`, `entailment` and others. Each external call takes
  `ctx: Ctx` (deadline, trace id, budget) first.
- Plugins register with `@plugin(slot=..., name=...)`. They are found in `plugins/`
  or through the `arie.plugins` entry point, chosen in `arie.yaml`, and import only
  `arie_core.plugin_api`. Plugins never import each other.
- `contract_tests/` holds the per-slot compliance suites that every implementation of
  a slot must pass.

## Consequences

- A provider can be swapped by editing `arie.yaml`, with no core change.
- The core cannot use SDK conveniences such as streaming helpers or retries. Adapters
  implement them behind the Protocol.
- A plugin that fails to import is skipped with a logged reason, so a broken optional
  plugin cannot stop the API from starting.
