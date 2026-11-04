# S10: Deterministic citation checks

**Needs:** S09 · **Size:** M

## Goal

Instant, free checks on every explain answer: each sentence is cited correctly, or flagged, or marked as background.

## Read first

- `core/arie_core/harness/verifier.py` (`preserves_anchors` and its number, negation and hedge helpers)
- `core/arie_core/harness/explain.py` (marker parsing)

## Change

- `core/arie_core/harness/citations.py`:
  - Split the answer into sentences (Markdown-aware: skip code blocks and headings; keep list items as sentences).
  - Per sentence, status `cited_ok`, `flagged` or `background`, with reasons: `invalid_marker` (marker not in the pack), `number_not_in_source` (a number not found in the cited chunks), `lost_qualifier` (negation, hedge or "only" missing compared with the cited text, reusing the verifier helpers), `uncited_number` (a number with no citation).
- Emit a `checks` SSE event after `citations`, and save it with the turn.
- Eval: per-case `citation_valid_rate` (share of cited sentences without flags) and `background_share`.

## Tests

- Table-driven: a correct citation, a wrong marker, an altered number, a dropped "not", an uncited number, a pure background sentence, a list item, a code block that is ignored.

## Done when

- `make check` passes. SPEC §8.9 lists the checks.
