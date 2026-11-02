# ARIE web workspace

The local-first research workspace uses React 19.2, the React Compiler 1.0, Vite,
TypeScript strict mode, Tailwind CSS, TanStack Query, and TanStack Virtual.

## Development

```sh
npm install
npm run dev
npm run typecheck
npm run test
```

Vite proxies `/v1` requests to `http://127.0.0.1:8000` by default. Override that
target with `VITE_API_PROXY`; set `VITE_API_URL` when the API is served from a
different origin outside the proxy.

## API types

`npm run generate:api` exports the FastAPI OpenAPI schema to `openapi.json` and
regenerates `src/api/generated.ts` from that schema.

The brief Markdown/KaTeX and PDF.js citation parsing work in dedicated workers.
PDF highlighting requires paper details to include `pdf_url` and citations with
`page_start`, `char_start`, and `char_end` offsets.

## Workspace shortcuts

- `j` / `k`: move through the paper feed
- `/` or `Cmd+K` / `Ctrl+K`: focus paper search
- Hover or keyboard-focus a paper row to prefetch its detail query
