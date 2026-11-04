# S01: Answer trace and logging

**Needs:** S00 · **Size:** M · Implements SPEC §13.4 option 1 (backend part)

## Goal

Every answer carries a trace that says where its time and tokens went and why it fell back, so later sessions can measure changes.

## Read first

- SPEC §8.1, §8.8, §13.4 (by heading)
- `core/arie_core/harness/answer.py`: `AnswerResult` and the Tier 3 block in `compose_answer`
- `core/arie_core/harness/orchestrator.py`, `harness/session.py`
- `api/arie_api/runtime/chat.py` (`answer_paper_question`, `save_chat_turn`) and where `paper_eval.py` builds a result

## Change

- Add a frozen `AnswerTrace` struct: seconds per stage (`plan`, `vector`, `read`, `rerank`, `write`, `tier3`, `review`, `total`), tokens (`prompt`, `completion`, `cache_hit`, `cache_miss`), `prompt_version` (for example `qa.v1`, from the YAML `name` and `version`), `model`, `mode`, `fallback_reason`.
- Fill it along the existing path. Put it on the answer result, in the SSE `answer` event's `evaluation` block, in the saved chat turn, and in each eval result.
- If `SupervisedPool.submit` can report queue wait separately from compute without a broad refactor, add `tier3_wait` and `rerank_wait`. Otherwise record totals and note it in PROGRESS.
- Log Tier 3 `entailment_timeout`, `entailment_unavailable` and `entailment_rejected` at WARNING with the trace id: one line, no traceback for timeouts.

## Don't

- Change any budget, default or answer behaviour.

## Tests

- With `ScriptedModel`: every stage key is present and non-negative, and `total` ≥ the sum of the stages that ran.
- A forced Tier 3 timeout logs one WARNING (`caplog`) and sets `fallback_reason`.

## Done when

- The trace appears in the SSE answer event, the saved turn and eval results.
- `make check` passes; `cd web && npm run generate:api` run if response models changed.
- SPEC §8.8 describes the trace.
