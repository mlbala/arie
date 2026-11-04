# Explain mode progress

Each session reads "Current state" and the last log entry, and appends one entry.
Keep entries to 12 lines. Put numbers here, not in chat.

## Current state

- Last done: S01 (2026-11-04)
- Next: S02
- Default answer mode: strict
- Baseline (strict, before changes): paper 2610.40325, arie.yaml llm = deepseek, one run, 11 cases per variant.
  configured (reranker on): passed 9/11 (0.818), fact recall 0.688, evidence hit 0.625, grounded 1.0, median 6.18 s.
  no_reranker: passed 11/11, fact recall 1.0, evidence hit 1.0, grounded 1.0, median 3.35 s.
- Latest gate numbers: none

## Open issues

- Queue wait is not split from compute in the trace (`tier3` and `rerank` are totals). `SupervisedPool.submit` holds its lock for the whole call and a call queued behind an abandoned one waits inside the executor, so a split needs worker-side start times and a new way to report them through the `EntailmentCheck` and `RerankCheck` callbacks. Do it in S03 or later if the totals are not enough.
- Baseline is one LLM run on 11 cases. The configured variant failed 2 fact cases (`question_not_answered`, `entailment_rejected`) that no_reranker passed, and one no_reranker case fell back with `TimeoutError` yet passed. Rerun before drawing conclusions about the reranker.
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
