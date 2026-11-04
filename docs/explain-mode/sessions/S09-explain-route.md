# S09: Explain route, SSE events and eval variant

**Needs:** S08 · **Size:** M

## Goal

Explain mode reachable over the API and measurable in evals. Strict stays the default.

## Read first

- `POST /v1/papers/{id}/questions` in `api/arie_api/routes.py` and how it streams SSE today
- `api/arie_api/runtime/chat.py` (`answer_paper_question`, `save_chat_turn`, history)
- `api/arie_api/paper_eval.py` (the `--mode` flag added in S04)

## Change

- Request field `mode: "strict" | "explain"`, default from config `answer.default_mode` (default `strict`).
- Explain SSE events: `progress`, `answer` (mode `explained_answer`, prompt version, trace), `delta` (text), `citations` (marker, chunk id, page, `char_start`, `char_end`), `done` (usage including cache hit and miss). Strict events unchanged.
- Save explain turns with the same events so a reload renders them. History for explain uses the last 6 turns, markers stripped.
- Eval: `--mode explain` runs cases through explain mode and scores `fact_recall_prose` on the answer text. Store `cache_hit_rate` and first-token seconds per case.
- Run `cd web && npm run generate:api`.

## Tests

- API test with a scripted provider: the event sequence and payload shapes; a saved turn replays; strict requests are unchanged; `mode=explain` with no LLM key abstains clearly.

## Done when

- `make check` passes. SPEC §8.9 and §11 updated.
- Optional manual check if a key is set: one curl to the SSE endpoint, with output trimmed to the first and last events.
