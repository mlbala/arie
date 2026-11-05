# ARIE code map

Where things live, so a session can open the right file without searching. Line numbers
drift; grep the symbol, then read a range.

## Answer path (strict mode, today)

| Step | Symbol | File |
| --- | --- | --- |
| Route | `POST /v1/papers/{id}/questions` | `api/arie_api/routes.py` |
| Runtime entry | `ChatHistory.answer_paper_question`, `save_chat_turn` | `api/arie_api/runtime/chat.py` |
| Orchestration | `answer_question`, `_semantic_ranking`, `ResearchAnswer` | `core/arie_core/harness/orchestrator.py` |
| Guard | `inspect_question` | `core/arie_core/harness/guardrails.py` |
| Plan | `plan_research` (LLM), `plan_answer` (keyword fallback) | `harness/planner.py`, `harness/intent.py` |
| Read | `read_for_answer`, `select_rows` | `harness/research.py` |
| Rank and filter | `retrieve`, `is_untrusted_instruction`, `locate_quote`, `select_excerpts` | `harness/reader.py` |
| Rerank | `rerank_reading` | `harness/rerank.py` |
| Write and review | `compose_answer`, `_source_windows`, `review_optional`, `fit_evidence`, `AnswerResult` | `harness/answer.py` |
| Verify | `preserves_anchors`, `attributed_premise`, `tier_3_scores` | `harness/verifier.py` |
| Model calls and budgets | `ModelSession`, `estimate_tokens`, `failure_reason` | `harness/session.py` |
| Prompts | `load_playbook`, `answer_prompts` read `prompts/qa.v1.yaml` | `harness/answer.py` |

## Contracts and plugins

- DTOs: `LLMRequest`, `LLMResponse`, `LLMDelta`, `TokenUsage` (already has `cache_hit`, `cache_miss`) in `core/arie_core/contracts/dtos.py`
- Slot protocols (`LLM.complete/stream`, `Entailment.score`, `Reranker.score`, `Ctx`, `@plugin`): `core/arie_core/plugin_api/__init__.py`
- LLM plugins: `plugins/llm-deepseek`, `plugins/llm-gemini`, `plugins/llm-openai-compatible`
- Provider wiring: `LLMProviders` in `api/arie_api/providers.py`
- Config validation: `ArieConfig` in `core/arie_core/registry/manager.py`

## Ingestion and storage

- Sync and import jobs: `SyncJobs._run_job`, `_supervised_paper` in `api/arie_api/runtime/sync.py`
- Chunking: `chunk_spans` in `core/arie_core/pipeline/chunker.py`
- Download and parse: `IngestionPipeline.download_pdf`, `parse_document` in `core/arie_core/pipeline/pipeline.py` (the unused indexing parts were removed in S02)
- Paper store (`arie_library.document_json`): `api/arie_api/runtime/library.py`
- Vectors: `VectorIndex.semantic_search` in `api/arie_api/runtime/vectors.py`
- Event-loop lag monitor, `supervised` and `ForegroundGate` (S16; held by `RuntimeBase.foreground`): `core/arie_core/sched/__init__.py` (the unused priority gate and rate limiter were removed in S02)

## Models and worker pools

- `AnswerModels` (`warm`, `scores`, `status`) in `api/arie_api/models.py`
- `SupervisedPool` in `worker/arie_worker/pool.py`; `ModelPool` in `worker/arie_worker/models.py`; `ExtractionPool` in `worker/arie_worker/extraction.py`

## Evals and feedback

- Runner and storage: `PaperEvaluations` (`serve`, `save_feedback`) in `api/arie_api/paper_eval.py`
- Question catalog and scoring: `catalog` and scorers in `core/arie_core/harness/paper_eval.py`
- CLI: `api/arie_api/eval_paper.py`; fixed synthetic cases: `api/arie_api/eval_answers.py`
- Qasper slice: CLI `api/arie_api/eval_qasper.py`; pure conversion (`convert_questions`) in `core/arie_core/harness/qasper.py`
- Rating route: `PUT /v1/papers/{id}/chat/{turn_id}/feedback` in `routes.py`
- Prompt optimizer (S17, offline, GEPA): `optimize_prompt`, `ExplainAdapter`, `ExplainRollouts` in `api/arie_api/optimize_prompts.py`; pure parts (`CallBudget`, `CountingLLM`, `split_cases`, `metric`, `feedback_text`, `render_candidate`) in `core/arie_core/harness/optimize.py`

## Web

- Chat and SSE: `web/src/components/ChatWorkspace.tsx`; session cache `web/src/paperSessions.ts`
- Evidence: `EvidenceDrawer.tsx`; evals: `EvaluationPanel.tsx`; health: `SystemPane.tsx`; provider settings: `SettingsDialog.tsx`
- API types: `web/src/api/generated.ts` from `npm run generate:api`

## docs/SPEC.md sections

1 invariants · 3 config · 4 plugins · 5 storage · 6 ingestion · 7 vectors · 8 answering (8.1 is the budget table) · 9 worker pools · 10 eval and feedback · 11 API, health, web · 12 docs vs code gaps · 13 excerpt fallback issue
