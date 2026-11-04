# S08: Explain harness core

**Needs:** S06, S07 · **Size:** L · **plan first** (use plan mode; Opus is worth it here if you have headroom)

## Goal

The core of explain mode: build the cached prompt and stream one cited explanation, with no route or UI yet.

## Read first

- `docs/explain-mode/README.md` "Fixed decisions" (prefix order, history, fit rule, budgets)
- ADR 0010, `core/arie_core/pipeline/pack.py`
- `core/arie_core/harness/session.py` (`ModelSession`), `guardrails.py` (`inspect_question`), `intent.py` (`plan_answer`)
- `core/arie_core/harness/orchestrator.py` to mirror its paper-binding pattern (I1)

## Change

- `prompts/explain.v1.yaml` with three parts:
  - `shared_header`: one or two fixed lines, also used later by the judge and guide.
  - `writer_rules`: explain for the reader's level; put `[cN]` after every sentence that uses the paper; every number, result and method claim must be cited; background explanation may stay uncited; never state results of other papers except as this paper reports them; if the paper doesn't cover it, say so; Markdown; target length by depth.
  - `depth_words`: target lengths for brief, standard and detailed.
- `core/arie_core/harness/explain.py`:
  - `build_prompt(header, pack_text, writer_rules, turns, question) -> LLMRequest`, byte-stable for the same pack and turns.
  - `paged_pack(pack, rows, plan, limit)` for papers over the fit limit: abstract, introduction, conclusion, then whole sections ranked by today's `retrieve`.
  - `explain_answer(ctx, ...)`: an async generator. Guard first (abstain as today), depth from `plan_answer`, one streaming call, yields text deltas, then a final item with the raw text, parsed `[cN]` markers mapped to chunk ids, usage and an `AnswerTrace`.
- `ModelSession`: a streaming path with explain budgets (first token 10 s, stall 15 s, total 90 s, output by depth), token accounting from the final usage, and a clear fallback reason on each timeout.

## Tests

- With `ScriptedModel` streaming: deltas arrive in order; markers map to chunk ids; two questions on one paper produce prompts whose bytes match up to the history; another paper's text never appears; guard abstains without a model call; each timeout gives its own reason.

## Done when

- `make check` passes. SPEC gains a short §8.9 "Explain mode".
