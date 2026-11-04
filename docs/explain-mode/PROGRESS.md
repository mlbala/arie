# Explain mode progress

Each session reads "Current state" and the last log entry, and appends one entry.
Keep entries to 12 lines. Put numbers here, not in chat.

## Current state

- Last done: S06 (2026-11-04)
- Next: S07 (S08 needs S06 and S07)
- Default answer mode: strict
- Baseline (strict, before changes): paper 2610.40325, arie.yaml llm = deepseek, one run, 11 cases per variant.
  configured (reranker on): passed 9/11 (0.818), fact recall 0.688, evidence hit 0.625, grounded 1.0, median 6.18 s.
  no_reranker: passed 11/11, fact recall 1.0, evidence hit 1.0, grounded 1.0, median 3.35 s.
- Qasper strict baseline (S05): Qasper dev, first 30 papers, arie.yaml llm = deepseek, embedder = fastembed (BAAI/bge-small-en-v1.5), reranker on, one run, 49 cases (46 fact, 3 abstain).
  passed 22/49 (0.449; 22/46 fact, 0/3 abstain), passed_written 15/46 (0.326), fact recall 0.438, fact recall prose 0.297, grounded 1.0, evidence hit n/a (0 by design), median 4.41 s.
  fallbacks 9: entailment_rejected 5, incomplete_coverage 2, question_not_answered 2. Embedder was fastembed, not Gemini, so compare later runs only with the same embedder.
- Latest gate numbers: none

## Open issues

- Queue wait is not split from compute in the trace (`tier3` and `rerank` are totals). `SupervisedPool.submit` holds its lock for the whole call and a call queued behind an abandoned one waits inside the executor, so a split needs worker-side start times and a new way to report them through the `EntailmentCheck` and `RerankCheck` callbacks. Do it in S03 or later if the totals are not enough.
- Baseline is one LLM run on 11 cases. The configured variant failed 2 fact cases (`question_not_answered`, `entailment_rejected`) that no_reranker passed, and one no_reranker case fell back with `TimeoutError` yet passed. Rerun before drawing conclusions about the reranker.
- `IngestionPipeline.discover`, its `queue` and `max_queue_size` have no caller since S02 removed `run` (`sync.py` only uses `gate`, `download_pdf` and `parse_document`). Left in place because the session file did not list them; `test_research_runtime.py` uses a different `queue` (the sync job queue). Remove in a later cleanup.
- ADR 0006's context says the QoS lanes were "enforced with asyncio semaphores". They never existed in the runtime. Noted in SPEC §12, ADR text left alone; amend it if the ADR is next touched.
- `degraded` is informational. A degraded model is still called, so each answer still waits for it up to its timeout. Skipping it while degraded is SPEC §13.4 option 2 (needs its own decision and maybe an ADR).
- `AnswerModel.scores` counts any cancellation as a timeout, because the harness's `asyncio.timeout` reaches it as one. A request cancelled for another reason (client disconnect, shutdown) counts the same way. Three in a row would show a false `degraded` until the next success.
- `EvaluationPanel` shows only the latest run (or the one just started). `overview.runs` lists earlier runs without results; a run picker would need `getEvalRun` per run.
- Eval results have two `mode` fields: `mode` is the answer's own result mode (`source_excerpts`, ...) and `requested_mode` is the run's `strict|explain`. The session file said "store `mode` per result", which would have overwritten the first. Old saved runs lack the S04 fields (`answer_written`, `fact_recall_prose`, `passed_written`, `requested_mode`) and the panel does not show them yet.
- The panel's fallback count (S03) needs a `fallback_reason`; `summarize` also counts excerpt answers without one as `unspecified`, so the two can differ on runs where the reason was not recorded.
- A worker exception that cannot be unpickled (for example `httpx.HTTPStatusError`) reaches the caller as `BrokenProcessPool` and hides the real cause; the Gemini 429 in S05 was only found by calling the embedder in-process. Make plugin errors picklable or wrap them in the pool.
- Qasper spans now match on letters and digits (case, punctuation, spacing, ligatures ignored) and the paper's own text becomes the key fact. 13 of 94 questions are still dropped; in the N=5 sample those held Qasper citation markers (`BIBREF13`, `lang2011unsupervised`) that never match PDF text. Stripping `BIBREF\d+` before matching would keep more; not done because it was not asked for.
- Qasper fact recall is exact text containment of the span, so a correct answer worded differently scores 0. Read `fact_recall_prose` as a lower bound; the judge (S13) is the fix. Only 3 abstain cases (all answered), too few to read.
- The embedder is now fastembed in `arie.yaml` (Gemini free tier hit 1000 requests a day). That made every library paper stale: about 2 of 420 were re-embedded before the API was stopped, at about 1 paper a minute on 1 thread (about 7 hours in all; `threads: 8` under `embedders.fastembed` would be faster). Until the API runs its backfill, chat on those papers is keyword-only. The 30 Qasper papers have fastembed vectors and joined the library.
- Pack sizes on the real library are not measured. The one attempt (a read-only script over `data/arie.db`) was blocked by the permission classifier, so the ADR makes no claim about how many papers fit under the 60% rule. Run `build_pack` over the INDEXED papers (or call `get_pack` once per paper) and record the token estimates against 64,000 and 128,000 before S08 relies on the whole-pack path. The 3 bytes per token estimate is conservative; real counts will be lower.
- Default `context_tokens` (deepseek 64,000, openai and gemini 128,000, other plugins 32,000) are conservative guesses, not provider-documented limits. Check them against the models in `arie.yaml`.
- The pack text escapes literal markers (`[c12]` becomes `[\c12]`) and has no inverse. S10's quote check must compare against the raw chunk found by id and offsets, never against pack text.
- The pack keeps every chunk that is not credits, a contents listing, references or acknowledgements, so equations and captions stay. The strict path's prose check (`_usable` in `research.py`) was not reused for that reason. `paper_eval.py` still has its own third copy of the section rules.
- The kit re-added `AGENTS.md` (CLAUDE.md imports it) although the README had become the canonical doc. Decide later whether to keep both.

## Log

<!-- Copy this template for each session, newest at the bottom.

### SNN: <title> (YYYY-MM-DD)
- Done:
- Decisions:
- Files:
- Checks: make check ✓/✗ · make web-check ✓/✗/n/a
- Numbers:
- Open:
- Next:
-->

### S00: Install this kit and record a baseline (2026-11-04)
- Done: kit committed (`.claude/` was missing from the repo and copied from `~/Downloads/arie-claude-kit`); `make check` and `make web-check` run; baseline eval run on 2610.40325 with `--compare-reranker`.
- Decisions: kept `AGENTS.md` and committed it with the kit; left the `docs/SPEC.md` and `docs/adr/*` deletions untouched; commits use the Conventional Commits style, sole author, no trailers; work lands on a branch and is merged into main with `merge:`.
- Files: CLAUDE.md, AGENTS.md, .claude/, docs/explain-mode/
- Checks: make check ✓ · make web-check ✓
- Numbers: 232 core + 13 api + 44 contract tests pass; ruff, import-linter (1 contract kept) and pyright clean; web lint, typecheck and 40 vitest tests pass. Eval numbers are under Baseline.
- Open: see Open issues (docs deletions, single-run baseline, AGENTS.md).
- Next: S01 (Needs S00).

### S01: Answer trace and logging (2026-11-04)
- Done: every answer has an `AnswerTrace` (seconds per stage plus total, provider tokens incl. cache hit/miss, prompt version, model, mode, fallback reason). It is on `AnswerResult`, in the SSE `answer` event as `evaluation.trace` (so in the saved turn), in each `PaperEvalResult` and each fixed-case `EvaluationScore`. Tier 3 `entailment_timeout`, `entailment_unavailable` and `entailment_rejected` log one WARNING with the trace id (traceback only for unavailable).
- Decisions: `docs/SPEC.md` and `docs/adr/*` restored with `git restore` on request (they were deleted in the working tree), so the old Open issue is closed; only `SPEC.md` is part of this commit. Queue wait not split (see Open issues). `prompt_version` is `name.vN` from the YAML header, read by `load_playbook`, which now rejects a prompt file without both.
- Files: core/arie_core/harness/{trace,session,answer,orchestrator,planner,evaluation,paper_eval}.py, api/arie_api/routes.py, tests in core/tests and api/tests, docs/SPEC.md §8.1, §8.8, §13.4
- Checks: make check ✓ · make web-check n/a (OpenAPI unchanged, no `generate:api` needed)
- Numbers: 239 core (+7) + 13 api + 44 contract tests pass; no budget, default or behaviour changed.
- Open: queue-wait split; `EvaluationPanel` does not show the trace yet (S03).
- Next: S02.

### S02: Remove dead code listed in SPEC §12 (2026-11-04)
- Done: deleted `QoSPriorityGate`, `HostRateLimiter` and the `TokenBucket` and `QoS_*` names only they used (`sched` keeps `EventLoopLagMonitor`, `supervised`); deleted `StartupReconciler`, `IngestionPipeline.chunk_text`, `index_document`, `run`, `process_next` and the unused imports and export. The `chunk_text` in `reader.py` and `verifier.py` is a keyword argument, not the pipeline method. The four dead `runtime_qos` keys are removed from `RuntimeQoSConfig`, `arie.yaml` and the README; old files still load and `ArieConfig.from_yaml` logs one deprecation warning.
- Decisions: warn in `from_yaml` (startup and reload), not in the validator, so Settings saves that re-validate the file text do not repeat it. `TokenBucket` and `QoS_*` went too because nothing else used them. `search_bm25`, `chunks_fts` and the migrations are kept for paged retrieval (S08). README "four priority lanes" and "FTS5" lines were already fixed; only the stale `sched` sentence and the `runtime_qos` row changed.
- Files: core/arie_core/{sched,pipeline,registry}, core/tests/test_registry_qos_budget.py, arie.yaml, README.md, docs/SPEC.md §3 and §12, .claude/skills/arie-session/reference.md
- Checks: make check ✓ · make web-check n/a (no web or API change)
- Numbers: 312 lines removed, 97 added (net -215; 293 removed and 80 added in Python). 241 core (-1 gate test, +3 config tests) + 13 api + 44 contract tests pass.
- Open: `IngestionPipeline.discover` and `queue` unused; ADR 0006 wording (see Open issues).
- Next: S03.

### S03: Honest model health and eval visibility (2026-11-04)
- Done: `AnswerModel.scores` counts consecutive failures per slot (pool `TimeoutError`, a caller cancellation, or any other exception). Three in a row make `status()` `degraded` with the last reason in `detail`; one success makes it `healthy`. `status()` also has `consecutive_failures`, `last_error` and `last_latency_seconds`; `GET /v1/storage` already spreads `status()`, so it carries them. `EvaluationPanel`: per case it shows mode, fallback reason, reranked, seconds and the slowest trace stage; per variant a line of fallback counts by reason. `SystemPane`: the model cards show the last call's latency, and the "Storage & health" header says "Model degraded" instead of "Healthy".
- Decisions: not-ready refusals are not failures. A refusal's `fallback_reason` (guard code, `not_in_paper`) is shown but not counted as a fallback. Runs saved before S01 have no reasons or trace; the panel says so instead of "no fallbacks". `FakePool` got `delay` and exception results rather than a new double.
- Files: api/arie_api/models.py, api/tests/test_answer_models.py, web/src/{api/client.ts,index.css}, web/src/components/{EvaluationPanel,SystemPane}.tsx and their tests, docs/SPEC.md §9, §10, §11, §13.4
- Checks: make check ✓ · make web-check ✓ · `generate:api` ran, no diff (both routes return untyped documents)
- Numbers: 241 core + 17 api (+4) + 44 contract tests pass; 43 web tests (+3, was 40).
- Open: see Open issues (degraded is informational, cancellation counted as timeout, latest run only, queue-wait split).
- Next: S04.

### S04: Answer-level eval metrics (2026-11-04)
- Done: each result has `answer_written`, `fact_recall_prose` (claim text only, 0 for unwritten answers), `passed_written` (fact cases) and reserved `citation_support`/`win_rate` = null; `passed` and `fact_recall` unchanged. Summary adds `passed_written`, `pass_written_rate`, `mean_fact_recall_prose`, `fallbacks` by reason, `median_stage_seconds`. Feedback cases take 1-3 key facts and an optional source sentence (`PUT .../chat/{turn_id}/feedback/facts`, table `arie_feedback_facts`) and then run as fact cases. Runs take `mode strict|explain` (API field, `--mode`); `explain` is refused as "not available yet" (409; CLI exits 2 before warming models).
- Decisions: key facts must be in the source sentence and the sentence in one chunk (same rule as generated cases); facts only on answers rated down, dropped when the rating is cleared. `requested_mode` is a new field because `mode` was taken (see Open issues). No UI (S15). Excerpt answers without a reason count as `unspecified` fallbacks.
- Files: core/arie_core/harness/paper_eval.py, api/arie_api/{paper_eval,routes,eval_paper}.py, tests in core/tests/test_paper_eval.py and api/tests/test_api_and_mcp.py, web/openapi.json, web/src/api/generated.ts, docs/SPEC.md §5 and §10, README.md
- Checks: make check ✓ · make web-check ✓ · `generate:api` ran (new route and `mode`: +170 generated lines)
- Numbers: 246 core (+5) + 19 api (+2) + 44 contract tests pass; 43 web tests. No eval run (needs a key); S05 records the first answer-level numbers.
- Open: see Open issues (two `mode` fields, panel does not show the new scores).
- Next: S05.

### S05: Qasper slice and strict baseline (2026-11-04)
- Done: `make eval-qasper N=<papers>` (`api/arie_api/eval_qasper.py`, pure conversion in `core/arie_core/harness/qasper.py`): downloads Qasper dev once, imports the first N papers, converts questions to `origin: "qasper"` cases, runs them in strict mode. Spans match on letters and digits only and the key fact is the paper's own text. `save_cases`, `chunks` (public) and `ensure_paper_vectors` added; a paper whose vectors fail 3 times is skipped as `vectors_failed`. Embedder switched to fastembed in `arie.yaml` (separate commit).
- Decisions: abstain only if every annotator says unanswerable; fact case from the first extractive answer (3 spans, 3+ letters or digits); no evidence offsets, so `evidence_hit` is 0 and `passed` rests on recall; exit 0 once a paper is evaluated (a baseline, not a gate). The session file said `source`; the struct field is `origin`.
- Files: core/arie_core/harness/qasper.py, api/arie_api/{eval_qasper,paper_eval,runtime/vectors}.py, tests in core/tests/{test_qasper,test_vector_search}.py and api/tests/test_eval_qasper.py, Makefile, README.md, AGENTS.md, arie.yaml, docs/SPEC.md §10, .claude/skills/arie-session/reference.md
- Checks: make check ✓ (252 core, 23 api, 44 contract) · make web-check n/a
- Numbers: 30 papers: 23 evaluated, 7 had no cases, 0 import or vector failures. 94 questions: 46 fact, 3 abstain, 13 dropped (span not in text), 16 yes/no and 16 free-form skipped. Results under Current state.
- Open: see Open issues (worker errors that cannot be unpickled, remaining dropped spans, exact-match recall, library re-embedding).
- Next: S06.

### S06: ADR 0010, paper pack and context limits (2026-11-04)
- Done: ADR 0010 (explain mode; amends 0003 and 0008; row added, 0003/0008 marked amended). `build_pack`, `pack_fits`, `escape_markers`, `marker_numbers` in `core/arie_core/pipeline/pack.py`. Table `arie_paper_packs`; `PaperPacks.ensure_pack`/`get_pack` (`api/arie_api/runtime/packs.py`) build at INDEXED in `sync.py` (a failure is logged, never fails the paper) and lazily for older papers; paper delete removes the row. Config: `llm_providers.<name>.context_tokens`, `ArieConfig.context_tokens_for`, `explain.max_pack_share` (0.6), both in `arie.yaml`.
- Decisions: `build_pack` takes `(paper_id, title, abstract, revision, chunks)` instead of the document dict (core has no `Document` type). Revision is `<dedup_hash>:<chunk_count>:p1`, so a layout change bumps `PACK_FORMAT`. Kept chunks = `annotate_sections` + `eligible("")` minus heading-only chunks; the strict prose check was not reused (it drops equations and captions). Escape form `[\c12]`, layout `Title / Abstract / Outline / Section: ... / [cN] text`.
- Files: core/arie_core/{pipeline/pack.py,registry/manager.py}, api/arie_api/runtime/{packs,__init__,library,sync,vectors}.py, arie.yaml, tests core/tests/{test_pack,test_pack_storage,test_registry_qos_budget}.py, docs/adr/{0010,README}, AGENTS.md, README.md, docs/SPEC.md §1, §3, §5, §6.2
- Checks: make check ✓ (272 core, 23 api, 44 contract) · make web-check n/a (no route or response model change)
- Numbers: +20 core tests (9 pack, 5 storage, 6 config). No eval run; no answer-path code changed.
- Open: see Open issues (pack sizes unmeasured, context_tokens guesses, no unescape, third copy of the section rules).
- Next: S07.
