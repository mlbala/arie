# S13: LLM judge, eval metrics, default switch, gate 2

**Needs:** S12, S05 · **Size:** M

## Goal

A judge for the few sentences the small checker flags, the two missing eval metrics, and the switch to explain mode by default once gate 2 passes.

## Read first

- `docs/explain-mode/README.md` "Fixed decisions" (prefix order, judge models)
- `core/arie_core/harness/explain.py` (`build_prompt`), `citations.py`
- `api/arie_api/paper_eval.py` (where S04 reserved `citation_support` and `win_rate`)

## Change

- `prompts/judge.v1.yaml`, two roles placed after the pack (same shared header, so the pack stays cached):
  - `support`: a sentence plus its cited chunks → `supported`, `partial` or `unsupported`, with one short reason.
  - `pairwise`: two answers plus the case's key facts → `A`, `B` or `tie`, with one short reason.
- Chat: run `support` only on sentences that are `flagged`, on the writer's provider (S12 added no grounding score, ADR 0011). Emit `judge` events and save them.
- Eval:
  - `citation_support` = share of cited sentences that are deterministic-ok and not judged unsupported.
  - `win_rate` = explain vs strict on the same case, judged both ways round; disagreement counts as a tie.
  - Pairwise judge provider: config `eval.judge_provider`, default a different provider than the writer when its key is set.

## Gate 2

Run the `arie-eval` skill in both modes on the baseline paper and the Qasper slice. If gate 1 and gate 2 pass, set `answer.default_mode: explain` in `arie.yaml`. Either way, record the numbers and the decision in PROGRESS.

## Tests

- Judge parsing, including malformed output; `support` runs only on flagged sentences; pairwise scoring both ways round.

## Done when

- `make check` passes; metrics show in the eval summary and panel; SPEC §10 updated.
