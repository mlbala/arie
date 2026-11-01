# AGENTS.md — ARIE Engineering Playbook for MAI-Code

## 1. System Identity & Naming Policy
ARIE (Agentic Research Intelligence Engine) is an enterprise-grade, local-first academic research intelligence platform built around a Microkernel + Pluggable Adapter Pattern.
- `arie_core` owns the ingestion state machine, safety gates, Q&A harness, and 3-tier quote verifier.
- All LLMs, embedders, vector stores, databases (SQLite, Postgres, stateless MCP-SQL), file stores, scrapers, and PDF parsers live in `plugins/`.
- **Initial Build Naming Rule:** This is the initial build (`0.1.0`). NEVER mention internal design iteration numbers (such as "v6" or "v4") in code, comments, UI headers, SQL migrations, OpenAPI schemas, or documentation. Always refer to the project simply as **ARIE** (`arie`).

## 2. Five Non-Negotiable Hard Invariants
Every line of code MUST preserve these invariants:
- **I1 (Paper Isolation, P(Contamination) = 0):** Q&A tools (`search_paper`, `read_section`, `get_table`, `get_page`) MUST NOT expose `paper_id` in their LLM-visible JSON schema. Bind `paper_id` via Python lexical closure at session creation AND post-filter every vector/SQL/MCP row (`[r for r in rows if r.paper_id == paper_id]`).
- **I2 (Zero Unverified Claims):** Every factual claim must cite an `Evidence` quote that passes the 3-tier verifier (Tier 1: exact substring; Tier 2: NFKC + ligature/hyphen/table normalization + SQLite offset overwrite; Tier 3: MiniCheck entailment) or abstain.
- **I3 (Bounded Execution):** Q&A harness enforces hard caps in code: max `K <= 3` tool calls (`K + 2` max LLM calls), `<= 12,000` input tokens, `<= 700` output tokens, and `asyncio.timeout(25.0)` wall-clock deadline. Strip tools on the final step.
- **I4 (Clean Store & SSRF Protection):** Validate all scraper outputs via `ScrapedPaperInput` (Pydantic v2, `extra="forbid"`, reject `..` or `\` in `paper_id`). Enforce plugin `allowed_hosts` (no redirects to undeclared hosts). Stream PDFs in 64 KB chunks; abort if `Content-Length` or stream > 25 MB or first 1024 bytes lack `%PDF-`. Commit via atomic `.pdf.part` -> `os.replace()` or atomic SQL transactions.
- **I5 (Zero Vendor SDKs in Core):** `arie_core` imports NO vendor SDKs and NO plugin packages. Plugins import ONLY `arie_core.plugin_api` and never import each other.

## 3. Coding, Typing & Concurrency Rules
- **Python 3.14+ Strict Typing:** Zero `Any` in public slot `Protocol` signatures. Every external call takes `ctx: Ctx` (`deadline: float`, `trace_id: str`, `budget: BudgetHandle`) as its first parameter.
- **DTO Split:** Use `msgspec.Struct(frozen=True)` for internal high-frequency runtime DTOs (`TokenUsage`, `ToolCall`, `LLMResponse`, `Evidence`, `Claim`, `AnswerDraft`). Use `pydantic.BaseModel(frozen=True, extra="forbid")` ONLY for config validation and `ScrapedPaperInput` boundary normalization.
- **Structured Concurrency:** NEVER use bare `asyncio.create_task`. Always use `asyncio.TaskGroup` with a `supervised()` wrapper so one failing paper never cancels a batch sync.
- **Process Isolation:** Never run PDF parsing (`pymupdf4llm`, `docling`), FastEmbed ONNX, or MiniCheck inside `arie-api` or threads (`asyncio.to_thread` is forbidden for MuPDF). Dispatch CPU tasks to `arie-worker` recycled worker processes (`max_tasks_per_child=20`, 60s `SIGKILL` timeout, 1.5 GB RSS cap).
- **4-Tier QoS Lanes:** Route work through `QoS_UserInteractive` (Lane 1: chat/feed), `QoS_UserInitiated` (Lane 2: manual audit overrides, `Semaphore(2)`), `QoS_Utility` (Lane 3: worker pool), and `QoS_Background` (Lane 4: scheduled syncs, `Semaphore(4)`).

## 4. Monorepo Package Map
- `core/arie_core/plugin_api/`: Versioned `@plugin` decorator, slot `Protocol` interfaces, `ScrapedPaperInput`, `normalize_scraper_output()`.
- `core/arie_core/contracts/`: Frozen `msgspec.Struct` runtime DTOs.
- `core/arie_core/persistence/`: SQLite WAL engine (`mmap_size=268435456`), `migrations/0001_arie_initial.sql`, transactional `outbox`.
- `core/arie_core/registry/`: Plugin discovery, Pydantic config validation, health checks, hot-switching.
- `core/arie_core/budget/` & `sched/`: `BudgetHandle`, 4-tier QoS gate, per-host token buckets (`export.arxiv.org` = 1 req / 3.0s).
- `core/arie_core/pipeline/`: Resumable ingestion state machine (`DISCOVERED -> GATED -> DOWNLOADING -> PARSED -> INDEXED -> BRIEFED`), 2-phase 64 KB PDF gatekeeper, section-aware chunker (400-600 tokens).
- `core/arie_core/harness/`: Zero-LLM router, RRF_k=60 hybrid prefetch, 3-zone prompt cache builder (Zone A: ~1200t global, Zone B: ~300t paper card, Zone C: <=4500t dynamic + BATS budget), lexical-closure tools, 3-tier verifier.
- `api/arie_api/`: FastAPI (`uvloop`), `/v1/*` REST + SSE endpoints, stateless MCP 2026-07-28 server (`/mcp/p/<handle>`).
- `worker/arie_worker/`: Background scheduler, reconciler, and recycled multiprocessing pool.
- `prompts/`: Versioned playbooks (`qa.v1.yaml`, `brief.v1.yaml`).
- `contract_tests/`: Slot compliance test suites run via `make test-contracts`.
- `web/`: React 19.2 + React Compiler 1.0, 3-pane workspace, `<Activity mode="hidden">`, Web Worker KaTeX & `pdf.js`.

## 5. Agent Workflow & Verification Loop
1. Read only the files referenced in the user prompt. Do not scan unrelated plugins.
2. Implement code concisely with docstrings stating contract and failure modes.
3. After every code change, run `make check` (`ruff`, `lint-imports`, `pyright`, `pytest`) in the terminal. Fix any errors before finishing the turn.

## 6. Git Commit & Authorship Policy (Strictly Sole User Authorship)
- **Sole Author Only:** All git commits and pushes MUST be authored exclusively by the local repository owner (`Balakrishnan` via configured `git config user.name` and `git config user.email`).
- **Zero Co-Author / AI Trailers:** NEVER append `Co-authored-by:`, `Generated-by:`, `Signed-off-by: GitHub Copilot`, `MAI-Code`, or any AI attribution trailer to commit messages, PR descriptions, or code comments.
- **No Author Overrides:** NEVER pass `-c user.name=...`, `-c user.email=...`, or `--author=...` when running `git commit`.
- **Conventional Commits:** Format all commit messages as clean, single-line or bulleted Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`) with no trailing attribution metadata.
