# S11: Explain mode in the web chat, gate 1

**Needs:** S10 · **Size:** L · web

## Goal

Users get streamed, readable explanations with clickable citations, and gate 1 is measured.

## Read first

- `web/src/components/ChatWorkspace.tsx`: grep for the SSE handling and message rendering, read those ranges only
- `web/src/components/EvidenceDrawer.tsx`, `web/src/paperSessions.ts`
- `markdown-it` is already a dependency; use it

## Change

- An Explain / Strict switch per chat, remembered per paper in `paperSessions`.
- Render `delta` events as streaming Markdown (sanitized: no raw HTML). Replace `[cN]` with citation chips showing the page; a click opens `EvidenceDrawer` at the chunk's offsets.
- Apply `checks` statuses per sentence: flagged sentences get a warning style and a tooltip with the reasons; background sentences use a quieter style. Render statuses generically so S12 can add `grounding` values without UI changes.
- Saved explain turns re-render from their stored events.

## Tests

- Vitest: streaming renders progressively; chips open the drawer at the right page; flagged and background styles apply; a saved turn re-renders; raw HTML in model text is escaped.

## Gate 1

Run the `arie-eval` skill with `--mode explain` on the baseline paper and the Qasper slice. Record against the strict baseline: `fact_recall_prose`, excerpt fallback rate, median first-token seconds, mean cache hit rate from the second question on. Write pass or fail per criterion (README "Gates") in PROGRESS. Don't switch the default here.

## Done when

- `make web-check` and `make check` pass; gate 1 numbers in PROGRESS.
