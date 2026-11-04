# S04: Answer-level eval metrics

**Needs:** S01 · **Size:** M

## Goal

The eval scores whether a written answer was right, not only whether retrieval found the passage. Today an answer made of excerpts can pass.

## Read first

- SPEC §10
- `core/arie_core/harness/paper_eval.py` (scorers) and `api/arie_api/paper_eval.py` (runner, storage, `save_feedback`)

## Change

- New per-case fields: `answer_written` (mode is not an excerpt fallback or abstention), `fact_recall_prose` (key facts found in claim or answer text only, quotes excluded), and `passed_written` (grounded, written, and `fact_recall_prose` ≥ 0.5). Keep `passed` and `fact_recall` unchanged so old runs stay comparable.
- Reserve `citation_support` and `win_rate` as `null` for S13.
- Run summary: counts of `passed_written`, mean `fact_recall_prose`, fallbacks by reason, median trace stage times.
- Feedback cases become scoreable: an API to attach 1–3 key facts and an optional source sentence to a feedback case. Once it has key facts, it is scored like a fact case. No UI in this session (S15 adds it).
- Eval runs accept `--mode strict|explain` and store `mode` per result. `explain` errors with "not available yet" until S09.

## Tests

- Scoring: an excerpt fallback that quotes the source sentence passes `passed` but fails `passed_written`.
- A feedback case with key facts gets scored.

## Done when

- `make check` passes; `generate:api` run if routes changed. SPEC §10 updated.
