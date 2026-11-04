# S05: Qasper slice and strict baseline

**Needs:** S04 · **Size:** M · Needs network and an LLM key on your machine

## Goal

An outside benchmark next to ARIE's self-generated questions: Qasper's questions were written from a paper's title and abstract, like a real user's, by NLP practitioners (Dasigi et al., NAACL 2021).

## Read first

- `api/arie_api/eval_paper.py` (CLI shape) and how `POST /v1/imports` queues an import (`api/arie_api/runtime/sync.py`)
- The case struct in `core/arie_core/harness/paper_eval.py`

## Change

- `api/arie_api/eval_qasper.py` plus `make eval-qasper N=30`:
  - Download the Qasper dev split once (dataset `allenai/qasper` on Hugging Face; check its card for the current file URL) into `data/eval/qasper/`.
  - Pick the first N papers, sorted by id for repeatability. Qasper ids are arXiv ids, so import each through the existing import path (`https://arxiv.org/pdf/<id>`) and wait until it is `INDEXED`.
  - Convert questions to eval cases with `source: "qasper"`: extractive-span answers become key facts (at most 3, each found verbatim in the paper text, otherwise drop the case); unanswerable questions become absent-topic cases that expect abstention; skip yes/no and free-form answers for now and count them.
- Keep conversion a pure function so it is testable without network.

## Tests

- Conversion on a small hand-written fixture in the shape of a Qasper record: spans kept, a span not in the text dropped, unanswerable mapped, yes/no skipped.

## Done when

- `make eval-qasper N=5` runs locally. Then run N=30 in strict mode with the `arie-eval` skill and record per-run numbers as the strict baseline in PROGRESS: cases, `passed`, `passed_written`, `fact_recall_prose`, fallbacks by reason, median seconds.
- `make check` passes without network.
