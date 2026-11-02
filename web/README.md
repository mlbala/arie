# ARIE web workspace

A chat-focused React 19.2 workspace with a compact library and live pipeline
sidebar. Select a paper, ask a question, and open its page citations in the source
drawer. The five most recent paper conversations retain their state with React
Activity; switching papers cancels the previous response stream.

## Development

Start the API from the repository root:

```sh
uv run uvicorn arie_api.routes:app --host 127.0.0.1 --port 8000 --loop uvloop
```

Then, from `web/`:

```sh
npm install
npm run dev
npm run build
npm test
```

Vite proxies `/v1` requests to `http://127.0.0.1:8000`. Override that target with
`VITE_API_PROXY`; set `VITE_API_URL` when using a different API origin.

## Sync and evidence

**Sync papers** starts a persistent background job across enabled sources. The
initial window is six calendar months, configured in `arie.yaml`. The sidebar
shows discovery, PDF download, extraction, and indexing, with pause/resume and
audit details. The library refreshes as successfully indexed papers become ready.

**Read a specific paper** accepts an arXiv PDF URL such as
`https://arxiv.org/pdf/1706.03762`. Importing a URL has no publication-date limit;
it fetches arXiv metadata, validates and extracts the PDF, and opens the paper
when indexing finishes. Manual imports can run while a batch sync is active.
Only arXiv PDF identifiers are accepted; abstract-page and external-site URLs
are rejected. Existing extracted versions are reused.

Overview questions such as **What is this paper about?** retrieve the abstract,
introduction, and conclusion. The answer is a concise extractive summary with
verified page citations and expandable supporting passages. A provider outage
still permits an extractive overview from the saved PDF. Normalized quotation
matches are mapped back to the saved page offsets before being returned.

During a question the sidebar shows retrieval, reading, and verification. The
client handles progress, verified claims, abstentions, errors, and completion
events. It rejects claims associated with another paper.

Citations address the saved page extraction. The drawer highlights the exact
verified text and offers the original PDF at the cited page. Extracted Markdown
offsets are kept separate from PDF viewer text offsets.

## API types

`npm run generate:api` exports FastAPI's schema to `openapi.json` and regenerates
`src/api/generated.ts`.

## Shortcuts

- `/`, `Cmd+K`, or `Ctrl+K`: open and focus library search.
- `Enter`: send a question; `Shift+Enter`: insert a new line.
- `Escape`: close the library or source drawer.
