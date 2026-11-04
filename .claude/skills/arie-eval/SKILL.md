---
name: arie-eval
description: Run ARIE's per-paper or fixed-case evaluation and report a compact summary (modes, fallback reasons, metrics); use after answer-path changes
argument-hint: "[paper id] [extra eval args]"
---

# Run an eval and summarize it

Never paste raw eval JSON or full logs into the conversation. Pipe through the filters below.

## Per paper (needs an ingested paper and a configured LLM key)

The CLI prints one JSON line per case, then one summary line. Keep only the columns that matter:

```sh
make eval-paper PAPER=$0 ARGS="$1" 2>/dev/null \
  | jq -rR 'fromjson? | select(.variant) | [.variant, .mode, .fallback_reason, .passed, .seconds] | @tsv' \
  | sort | uniq -c
make eval-paper PAPER=$0 ARGS="$1" 2>/dev/null | tail -1 | jq '.summary'
```

Running it twice repeats the run; prefer saving once with `| tee /tmp/eval-$0.jsonl` and filtering the file.

If the API is up (`localhost:8000`), the latest saved run is `curl -s localhost:8000/v1/papers/$0/eval | jq '.latest.summary'`.

Field names grow as sessions add metrics (trace, prose fact recall, citation support, win rate). Check `api/arie_api/paper_eval.py` for the current names before filtering on a new field.

## Fixed synthetic cases (no ingested paper needed)

```sh
make eval-llm 2>&1 | tail -40
```

## Report

A table: variant · cases · passed · fact recall · citation support (once present) · fallbacks by reason · median seconds. Compare with the last numbers in `docs/explain-mode/PROGRESS.md` and record the new ones there.
