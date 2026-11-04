# S12: Background grounding checker

**Needs:** S10 · **Size:** L · Measure first

## Goal

A small local model scores each cited sentence against its cited chunks after the answer is shown, off the critical path.

## Read first

- SPEC §4 (slots) and §9 (pools)
- `plugins/entail-nli-onnx` as the pattern for a model plugin that refuses to run outside its worker
- `api/arie_api/models.py` (`ANSWER_SLOTS`, warm-up and status)

## Step 1: measure, then decide

- Candidates: Vectara HHEM-2.1-Open (flan-t5-base, Apache-2.0) and LettuceDetect base (ModernBERT, MIT).
- In the Docker image on 2 vCPUs, time each on 20 real (sentence, cited chunks) pairs from a saved explain answer. Record latency per pair, worker RSS and the image size change.
- Suggested bar: about 5 s for 15 pairs, under 1 GB RSS. If a candidate needs PyTorch and grows the image a lot, **ask the user** before adding it. If neither fits, record that and skip to Step 3 with the LLM judge only (S13).

## Step 2: build

- New slot `grounding` with `score(ctx, pairs) -> list[float]` (the `Entailment` shape, separate so strict mode's Tier 3 is untouched) and the chosen plugin in `plugins/ground-<name>`.
- Its own `ModelPool` at nice 10 so it never slows the next question; warm-up and status like the other answer models.
- After the `checks` event, a background task scores cited sentences, emits `grounding` events while the SSE stream is open (30 s cap) and saves the scores with the turn either way.
- Config: `grounding.min_score`; sentences below it become `flagged` with reason `low_grounding`.

## Step 3: tests

- Contract test for the slot; API test that grounding events arrive and are saved, and that a slow checker never delays `done`.

## Done when

- Measurements and the choice in PROGRESS; `make check` passes; ADR via `arie-adr` for the new slot; SPEC §4 and §9 updated.
