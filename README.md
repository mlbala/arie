# arie — Agentic Research Intelligence Engine

**ARIE** is an enterprise-grade, local-first academic research intelligence platform built around a **Microkernel + Pluggable Adapter Pattern**. It features:

- **Zero-SDK Microkernel & 9 Pluggable Adapter Slots:** Hot-swap any LLM, embedding model, vector store, SQL database (SQLite WAL, Postgres, or external stateless `mcp-sql`), file storage, PDF parser, or custom paper scraper via `arie.yaml` without touching core code.
- **Automated Multi-Source Ingestion & Boundary Gates:** Pulls AI/CS papers from arXiv, Google DeepMind (via OpenAlex ROR lineage), and custom BYO scrapers (e.g., OpenReview) guarded by a strict Pydantic v2 `ScrapedPaperInput` boundary contract, zero-token relevance filtering, SSRF host allow-lists, and a 64 KB `%PDF-` streaming gatekeeper.
- **Strict Single-Paper Isolation ($P(\text{Contamination}) = 0$):** Binds `paper_id` inside Python tool lexical closures and post-filters every vector and SQL/MCP query so cross-paper contamination is structurally impossible.
- **Bounded Agentic Q&A & 3-Zone Prefix Caching:** Combines zero-LLM routing, hybrid prefetch (FTS5 BM25 + exact vector RRF), a 3-zone prompt cache layout, and a BATS budget-tracked loop hard-capped at $K \le 3$ tool calls, $12,000$ tokens, and $25\text{ s}$.
- **3-Tier Quote & Entailment Verifier Ladder:** Every claim in pre-computed 5-part executive briefs and live Q&A must pass exact byte matching (Tier 1), Unicode NFKC/ligature normalization with authoritative SQLite offset mapping (Tier 2), and local MiniCheck semantic entailment (Tier 3) before reaching the user.
- **Hybrid Multi-Process Runtime & 4-Tier QoS Scheduling:** Isolates non-thread-safe PDF parsing (`PyMuPDF4LLM` / `Docling`) and ONNX inference in recycled worker processes while keeping `arie-api` (`uvloop`) under $\le 10\text{ ms}$ p99 lag via 4 GCD-inspired QoS priority lanes.
- **Chat-focused React 19.2 Workspace:** A wide paper conversation with a compact library and live pipeline sidebar, recent conversations retained with `<Activity mode="hidden">`, and page citations that open highlighted extraction text alongside the original PDF.

## Run locally

From the repository root, install dependencies and start the API:

```sh
uv sync
uv run uvicorn arie_api.routes:app --host 127.0.0.1 --port 8000 --loop uvloop
```

In a second terminal:

```sh
cd web
npm install
npm run dev
```

Open Vite's local URL and choose **Sync options**. Select recent months, a whole
year, one month in a year, a starting month, or an inclusive month range. Choose
the source, metadata page size, and paper limit, then **Sync papers**. **Save
defaults to arie.yaml** persists these choices for the next launch. Dates use UTC;
ongoing years and months stop at the current time. The default is six calendar
months and at most 100 examined papers per run, including skipped or duplicate
records. Reaching the limit pauses the run. **Resume saved run** continues its
original window, settings, and position within the source page. Successful recent
month syncs use a one-day overlap and skip unchanged extracted papers.
You can also pause after the current paper.
Interrupted downloads never become library entries; PDFs appear only after text
extraction and evidence indexing succeed. Sync progress and audit failures persist
in the configured database.

Chat produces a faithful summary with exact supporting quotations and page
citations. One configured LLM fills two sequential roles: a writer and a separate
evidence reviewer. The reviewer checks each proposed statement against its own
supporting passage; the saved PDF text remains the source of truth. Deterministic
checks reject fabricated quotations, changed numbers, lost qualifications and new
absolute claims. If either model stage fails or credentials are rejected, chat
uses labeled source excerpts. The UI explains when the model connection fails.
The two model calls share a 700-token output cap, a conservative 12,000-token input
cap, and the existing 25-second request deadline.
Image-only PDFs that produce no text are reported as extraction failures requiring
OCR; they are not treated as successfully indexed.

To read an older paper directly, paste its arXiv PDF link into **Read a specific
paper** in the sidebar and choose **Import paper**. This import bypasses only the
batch publication-date window; PDF validation and source evidence checks still
apply. The completed paper opens automatically, including papers outside the
first library page. Ask **What is this paper about?** for a concise summary drawn
from the paper's narrative sections, with page citations and supporting quotes.

## Sync configuration and workspace controls

Set `sync.mode` in `arie.yaml`, keeping only the date fields for the chosen mode:

```yaml
sync:
  mode: range
  start_month: "2025-01"
  end_month: "2025-06"
  batch_size: 20
  max_papers: 100
```

For a whole year, use `mode: year` and `year: 2025`. For a single month, use
`mode: month`, `year: 2025`, and `month: 6`. To sync from January 2026 onward,
use `mode: since` and `start_month: "2026-01"`. For a rolling window, use
`mode: relative` and `initial_months: 6`. Do not combine these date selectors.
Each run snapshots its settings, so saving different defaults cannot change an
active or paused run.

**Filter & sort** in the library filters all saved papers by publication month
and research domain, with newest, oldest, or domain ordering. Search covers all
library summaries, ignores case and accents, and ranks partial or misspelled title
matches ahead of topic and abstract matches. Searching does not automatically
choose a result: select a paper before asking a question. Suggested domain tags (LLMs, agents, NLP, vision, and others)
come from exact matches in source categories, titles, abstracts, and the first
five extracted pages. Their metadata records the matching text and its origin;
they are navigation hints. Existing saved papers receive tags at startup. Source
metadata and parser metadata remain attached to each paper.

**Layout** moves the library left or right and places **Read a specific paper**
in the library or the right activity pane. The browser remembers your choice.
**Import & status** opens the compact right pane for current extraction activity,
the selected document's tags and evidence counts, database size, PDF storage and
free disk space. Storage probes are cached for 15 seconds and large directory
scans report partial totals. Remote storage sizes are unavailable unless measured.
The current sync pipeline saves evidence in the main database; its configured
vector provider is shown as inactive rather than reporting an unpopulated index
as healthy.

For local storage, the selected document offers **Delete PDF** (retain extracted
text for chat) and **Delete paper** (remove its PDF, metadata, and evidence).
Confirm the selected action in the pane. Pause sync and wait for active imports
before deleting. Files are staged before the database change and restored if
that change fails. A later sync can rediscover a fully deleted paper; import its
link again to restore a deleted PDF.

If import reports that the running API has no import route, restart the API with
the command above. During local development, add `--reload` so new routes are
loaded after source edits.

## Answer prompts and source-grounding evaluations

Edit `prompts/qa.v1.yaml` to refine the planner, writer and independent evidence-review
instructions. New questions load the current playbook. Overviews request the
contribution, method, evaluation, findings and limitations when supported; absent
sections are omitted. `prompts/brief.v1.yaml` records the same fidelity policy for
briefs. No external knowledge may fill gaps in the saved paper.

Run `make check` for deterministic regressions covering typo search, exact source
offsets, foreign-paper isolation, fabricated quotes, changed numbers, missing
negation, unsupported paraphrases and malformed review verdicts.

Run `make eval-llm` to exercise the configured provider against fixed synthetic
source facts. Cases cover summary coverage, qualified limitations, missing evidence
and source prompt injection. It also checks comparisons, methodology, compound
questions, unsupported premises, input overrides and cross-paper requests.
Reference facts and forbidden distortions are fixed
in code rather than written by the judging model. Each case reports exact-source
grounding, fact coverage, review completion and budgets separately. A fallback is
not counted as a successful model summary. Authentication rejection stops the run
and reports its HTTP status without exposing the key. Configure a valid provider
key before rerunning. Passing this small regression set is not a guarantee that
all future summaries will be correct; citations remain available for inspection.

The backend uses the configured model in three bounded roles: plan what the
question needs, write from exact source passages, then independently check support,
relevance and requested coverage. The planner returns a validated reading plan,
never executable actions. Paper-bound Python functions search and expand sections
or pages under three tool calls. All roles share 12,000 input tokens, 700 output
tokens and a 25-second deadline. Explicit instruction overrides and external reads
are blocked before model calls. History contains previous questions and validated
source locators; previous generated prose never becomes evidence. Provider failure
uses clearly labelled exact source excerpts and is not counted as model-reviewed.
The configured four-process PDF pool and two import consumers are extraction and
queue workers, not additional LLM agents.
