# S14: Paper guide and section summaries, gate 3

**Needs:** S10 · **Size:** L · API + web

## Goal

Every paper opens with a cited guide (the NotebookLM feel), and overview questions get section summaries to work from.

## Read first

- `core/arie_core/harness/explain.py` (`build_prompt`) and `citations.py`
- Where a paper becomes `INDEXED` in `api/arie_api/runtime/sync.py`
- How `ChatWorkspace.tsx` shows an empty chat (grep, read that range)

## Change

- `prompts/guide.v1.yaml`, placed after the pack like the judge. One call per paper returns JSON: `tldr`, `problem`, `method_steps`, `key_equations` (with every symbol explained), `results`, `limitations`, `glossary`, `section_summaries` (keyed by outline heading, one or two sentences each), `suggested_questions` (5). Every factual item carries `[cN]` markers. Output cap 2,500 tokens.
- Validate the JSON and run the S10 checks on its sentences; drop items that fail.
- Runtime table `arie_paper_guides(paper_id, revision, prompt_version, json)`. A background task builds it after `INDEXED`, waits while a question is streaming, and retries on the next wake-up after a failure.
- `make guides` builds missing guides for existing papers, one at a time.
- Strict planner: give it the section summaries in place of bare section names when a guide exists.
- Web: show the guide as the opening message of a paper's chat, with citation chips and the suggested questions as buttons. Run `generate:api`.

## Tests

- Guide JSON parsing and validation with a scripted model; the background task skips a paper while a question streams; Vitest for the opening message.

## Gate 3

Run the `arie-eval` skill on overview-type cases with and without guides; record the judge win rate in PROGRESS.

## Done when

- `make check` and `make web-check` pass; SPEC §5 and §8 updated.
