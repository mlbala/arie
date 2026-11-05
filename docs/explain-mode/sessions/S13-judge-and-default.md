# S13: LLM judge, eval metrics, default switch, gate 2

**Needs:** S12, S05 · **Size:** M

## Goal

A judge that checks every cited sentence of an explain answer in one call, the two missing eval metrics, and the switch to explain mode by default once gate 2 passes.

## Read first

- `docs/explain-mode/README.md` "Fixed decisions" (prefix order, checks after streaming, judge models)
- [ADR 0011](../../adr/0011-no-local-grounding-model.md) (no local grounding model, so the judge is the only model check)
- `core/arie_core/harness/explain.py` (`build_prompt`), `citations.py`
- `api/arie_api/paper_eval.py` (where S04 reserved `citation_support` and `win_rate`)
- `ArieConfig` in `core/arie_core/registry/manager.py` (for the new `judge` key)

## Change

- `prompts/judge.v1.yaml`, two roles placed after the pack (same shared header, so the pack stays cached):
  - `support`: all of an answer's cited sentences, numbered, each with its `[cN]` markers → one verdict per sentence (`supported`, `partial` or `unsupported`) with one short reason. The chunks are already in the cached pack, so the call sends only the markers, not the chunk text again.
  - `pairwise`: two answers plus the case's key facts → `A`, `B` or `tie`, with one short reason.
- Chat: after the `checks` event, run `support` once per answer on every cited sentence (each sentence with a marker, flagged or not), in one batched call on the writer's provider, through `ModelSession` with its own budget (I3). Emit `judge` events and save them with the turn. No call when the answer cites nothing.
- Config: `judge.chat: all | off`, default `all`, validated in `ArieConfig`. `off` makes no judge call in chat and emits no `judge` events. It is a chat switch only: eval runs always judge, so `citation_support` is always computed.
- Eval:
  - `citation_support` = share of cited sentences that are deterministic-ok and not judged unsupported.
  - `win_rate` = explain vs strict on the same case, judged both ways round; disagreement counts as a tie.
  - Pairwise judge provider: config `eval.judge_provider`, default a different provider than the writer when its key is set.

## Gate 2

Run the `arie-eval` skill in both modes on the baseline paper and the Qasper slice. If gate 1 and gate 2 pass, set `answer.default_mode: explain` in `arie.yaml`. Either way, record the numbers and the decision in PROGRESS.

## Tests

- Judge parsing, including malformed output (a missing, extra or repeated sentence number); `support` judges every cited sentence in exactly one call per answer; `judge.chat: off` makes no judge call; an unknown `judge.chat` value is rejected; pairwise scoring both ways round.

## Done when

- `make check` passes; metrics show in the eval summary and panel; `judge.chat` is in `ArieConfig` and SPEC §3; SPEC §8.9 and §10 updated.
