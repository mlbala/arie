# 0002: Single-paper isolation by lexical closure and post-filtering

**Status:** Accepted

## Context

Every answer must come from the selected paper only (invariant I1,
P(contamination) = 0). Models can be prompt-injected by paper text, and vector stores
or remote SQL servers can return rows they should not. A `paper_id` argument in a
model-visible tool schema could be overwritten by the model.

## Decision

- Paper-scoped tools (`search_paper`, `read_section`, `get_table`, `get_page`) never
  expose `paper_id` in their schema. It is bound by a Python closure when the session
  is created.
- Every row from a vector store, the database or an MCP server is filtered again in
  core: `[r for r in rows if r.paper_id == paper_id]`. This happens even when the
  query already filtered by paper.
- Text that looks like an instruction to the model is dropped from evidence before
  any prompt is built.

## Consequences

- A faulty or malicious store cannot leak another paper's text into an answer. The
  contract tests feed foreign rows on purpose and assert that they are removed.
- Comparing papers needs separate single-paper sessions whose results are merged in
  the UI, never one session that sees several papers.
