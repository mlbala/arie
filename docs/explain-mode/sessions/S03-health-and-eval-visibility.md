# S03: Honest model health and eval visibility

**Needs:** S01 · **Size:** M · API + web · SPEC §13.4 option 1 (rest)

## Goal

A model that times out on every answer shows as degraded, and the eval panel shows why each case fell back.

## Read first

- SPEC §9 (the `AnswerModels` paragraph) and §11
- `api/arie_api/models.py` (`AnswerModels.status`, `scores`)
- `web/src/components/EvaluationPanel.tsx` and `SystemPane.tsx` with their tests

## Change

- `AnswerModels`: count consecutive timeouts and failures per slot. Three in a row → `degraded` with the last reason; one success → `healthy`. Keep the last call's latency in `status()`.
- `GET /v1/storage` already reports `status()`; make sure the new fields reach it.
- `EvaluationPanel`: per-case rows for the selected run showing variant, mode, fallback reason, reranked, seconds, and the trace's slowest stage. Add a line with fallback counts by reason. Data comes from `GET /v1/papers/{id}/eval/runs/{run}`.
- `SystemPane`: show `degraded` with its reason.
- Run `cd web && npm run generate:api`.

## Tests

- API: healthy → degraded after 3 timeouts → healthy after a success (reuse `FakePool`).
- Vitest: the panel renders fallback reasons and counts; the system pane shows a degraded model.

## Done when

- `make check` and `make web-check` pass. SPEC §9 and §11 updated.
