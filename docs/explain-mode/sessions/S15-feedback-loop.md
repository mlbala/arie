# S15: Feedback loop, gate 4

**Needs:** S13 · **Size:** M · API + web

## Goal

Thumbs ratings become reasons, regression tests and judge labels. Ratings find failures; they are not a quality score.

## Read first

- SPEC §10 (feedback)
- `PUT /v1/papers/{id}/chat/{turn_id}/feedback` in `routes.py` and `save_feedback` in `api/arie_api/paper_eval.py`
- The thumbs UI in `ChatWorkspace.tsx` (grep), `EvaluationPanel.tsx`

## Change

- Feedback gets optional `reasons` chosen from: `wrong`, `not_in_paper`, `too_shallow`, `too_long`, `missing_citation`, `did_not_answer`, plus an optional comment. Store `mode`, `prompt_version` and the trace id with it. Add columns only if missing.
- Implicit feedback: when a follow-up says the last answer was wrong, asks for it again, or repeats the previous question with high word overlap, mark the previous turn `implicit_negative`. No UI.
- Eval panel: list feedback cases with a small form to add 1–3 key facts or pick the source passage (API from S04), which makes them scored cases.
- Judge agreement: for rated turns, compare the judge's verdict with the user's rating; show agreement overall and per task.

## Tests

- Reasons saved and returned; implicit-negative detection on a few phrasings; agreement maths; Vitest for the reason chips and the key-fact form.

## Gate 4

Once 30 or more answers are rated, record the judge agreement in PROGRESS.

## Done when

- `make check` and `make web-check` pass; SPEC §10 updated.
