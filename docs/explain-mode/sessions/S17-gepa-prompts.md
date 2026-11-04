# S17 (optional): Prompt optimization with GEPA

**Needs:** S15 · **Size:** M

## Goal

An offline script proposes a better `explain` prompt from eval failures and your feedback reasons. It never changes the live prompt by itself.

## Read first

- `prompts/explain.v1.yaml`, the eval runner's case and result fields
- The `gepa` package's README (github.com/gepa-ai/gepa): how to wrap a system with a metric and text feedback

## Change

- `api/arie_api/optimize_prompts.py` plus `make optimize-prompts PROMPT=explain MAX_CALLS=200`.
- Split eval cases (Qasper slice, the paper's own cases, scored feedback cases) into train and held-out sets, fixed by seed.
- Metric: `fact_recall_prose` + `citation_support` − a penalty for excerpt fallbacks. Text feedback for GEPA's reflection: judge reasons, check flags and the user's feedback reasons.
- Hard cap on model calls (`MAX_CALLS`), printed before starting and enforced.
- Output: `prompts/explain.v<N+1>.yaml` as a candidate plus a short report (train vs held-out scores for old and new).

## Tests

- With a scripted model and a tiny case set: the call cap is enforced, the held-out split never reaches the optimizer, and the output file is written but not activated.

## Done when

- A dry run on 10 cases works; the report is in PROGRESS. The user decides whether to switch `explain` to the new version.
