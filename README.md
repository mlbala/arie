# arie — Agentic Research Intelligence Engine

**ARIE** is an enterprise-grade, local-first academic research intelligence platform built around a **Microkernel + Pluggable Adapter Pattern**. It features:

- **Zero-SDK Microkernel & 9 Pluggable Adapter Slots:** Hot-swap any LLM, embedding model, vector store, SQL database (SQLite WAL, Postgres, or external stateless `mcp-sql`), file storage, PDF parser, or custom paper scraper via `arie.yaml` without touching core code.
- **Automated Multi-Source Ingestion & Boundary Gates:** Pulls AI/CS papers from arXiv, Google DeepMind (via OpenAlex ROR lineage), and custom BYO scrapers (e.g., OpenReview) guarded by a strict Pydantic v2 `ScrapedPaperInput` boundary contract, zero-token relevance filtering, SSRF host allow-lists, and a 64 KB `%PDF-` streaming gatekeeper.
- **Strict Single-Paper Isolation ($P(\text{Contamination}) = 0$):** Binds `paper_id` inside Python tool lexical closures and post-filters every vector and SQL/MCP query so cross-paper contamination is structurally impossible.
- **Bounded Agentic Q&A & 3-Zone Prefix Caching:** Combines zero-LLM routing, hybrid prefetch (FTS5 BM25 + exact vector RRF), a 3-zone prompt cache layout, and a BATS budget-tracked loop hard-capped at $K \le 3$ tool calls, $12,000$ tokens, and $25\text{ s}$.
- **3-Tier Quote & Entailment Verifier Ladder:** Every claim in pre-computed 5-part executive briefs and live Q&A must pass exact byte matching (Tier 1), Unicode NFKC/ligature normalization with authoritative SQLite offset mapping (Tier 2), and a local NLI entailment model run in a worker process (Tier 3) before reaching the user.
- **Hybrid Multi-Process Runtime & 4-Tier QoS Scheduling:** Isolates non-thread-safe PDF parsing (`PyMuPDF4LLM`, or `Docling` with OCR) and ONNX inference in recycled worker processes while keeping `arie-api` (`uvloop`) under $\le 10\text{ ms}$ p99 lag via 4 GCD-inspired QoS priority lanes.
- **Chat-focused React 19.2 Workspace:** A wide paper conversation with a compact library and live pipeline sidebar, recent conversations retained with `<Activity mode="hidden">`, and page citations that open highlighted extraction text alongside the original PDF.

## Run with Docker

Requires Docker Engine with Compose v2 (Docker Desktop 4.x or later). Allocate at
least 3 GB of memory to Docker; PDF extraction runs in a pool of worker processes.

1. Create your environment file and add a key for the LLM provider selected in
   `arie.yaml` (`plugins.llm.use`, DeepSeek by default):

   ```sh
   cp .env.example .env
   ```

2. Build and start the API and web workspace:

   ```sh
   docker compose up --build -d
   ```

3. Open <http://localhost:8080>. The API is also reachable at
   <http://127.0.0.1:8000> for MCP clients and scripts.

Useful commands:

```sh
docker compose ps                 # health of both services
docker compose logs -f api        # follow API and extraction logs
docker compose restart api        # reload .env after changing keys
docker compose down               # stop; library data is kept
docker compose down -v            # stop and delete all papers, the database and saved settings
```

**What runs.** The `api` service is the FastAPI backend with its PDF extraction
pool. The `web` service is the built React workspace served by nginx, which proxies
`/v1` (including the live event stream) and `/mcp` to the API. Both images come from
the multi-stage `Dockerfile` (`--target api` and `--target web`).

**Data and settings.** The `arie-data` volume holds the SQLite database, PDFs,
the embedding and claim-checker model caches and the active `arie.yaml`. On first
start the API downloads the claim-checker model (about 280 MB); chat answers with
exact source excerpts until **Storage & health** shows the claim checker as healthy.
The first start also copies the repository's `arie.yaml` into the volume. Settings
saved from the UI (sync defaults, provider choice) update that copy and survive
restarts and rebuilds. To replace it with an edited `arie.yaml` from your checkout:

```sh
docker compose cp arie.yaml api:/app/data/arie.yaml
docker compose restart api
```

The `ARIE_CONFIG` environment variable selects the configuration file; without it,
ARIE reads `arie.yaml` from the repository root.

**Configuration keys** (`.env`):

| Key | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | Provider keys; set the ones you use. Chat falls back to labelled source excerpts without one. |
| `ARIE_MCP_SECRET` | Signs scoped MCP URLs. Set a random value (`openssl rand -hex 32`) for shared deployments. |
| `ARIE_WEB_PORT` | Host port for the web workspace (default `8080`). |
| `ARIE_API_PORT` | Host port for direct API access, bound to `127.0.0.1` (default `8000`). |

**Development overlay.** To edit code with live reload, mount the source over the
images and run the Vite dev server:

```sh
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Python changes under `api/`, `core/`, `worker/` and `plugins/` restart the API
automatically, `prompts/` edits apply to the next question, and changes in `web/src`
hot-reload in the browser. Dependency changes
(`uv.lock`, `web/package-lock.json`) need `--build`.

## Run locally

From the repository root, install dependencies and start the API:

```sh
uv sync
uv run uvicorn arie_api.routes:app --host 127.0.0.1 --port 8000 --loop uvloop
```

The first start downloads the claim-checker model into the Hugging Face cache
(`HF_HOME`, by default `~/.cache/huggingface`).

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
evidence reviewer. Between them, a local claim checker (Tier 3, see below) drops any
drafted statement its own quotation does not entail, so the reviewer sees only
supported claims; the saved PDF text remains the source of truth. Deterministic
checks reject fabricated quotations, changed numbers, lost qualifications and new
absolute claims. If a model stage or the claim checker fails, or credentials are
rejected, chat uses labeled source excerpts and says why.
The two model calls share a 700-token output cap, a conservative 12,000-token input
cap, and the existing 25-second request deadline.
Image-only PDFs that produce no text are reported as extraction failures requiring
OCR; they are not treated as successfully indexed. Select the Docling parser (below)
to extract them with OCR.

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
The vector index reports measured coverage: how many chat-ready papers are
embedded with the configured model, the vector count and its size on disk.

## Vector search and bringing your own vector store

Sync saves evidence chunks in the main database, so a paper is chat-ready before
it is embedded. A background indexer in `arie-api` then embeds each paper in a
recycled `arie-worker` process and writes the vectors to the configured store. It
also backfills existing papers and re-embeds the library when `plugins.embedder.model`
changes, since vectors from different models are not comparable. Answers fuse keyword
and vector rankings (RRF, k=60) inside the selected paper only. Vector-only matches
below `plugins.embedder.min_score` (default 0.6, tuned for `bge-small-en-v1.5`) are
dropped. If the embedder or store fails, answers fall back to keyword retrieval and
the status pane shows `degraded` with the reason. Set either plugin's `use` to `none`
for keyword-only retrieval.

To use your own store, add a class under `plugins/` (or ship a package with an
`arie.plugins` entry point) that implements `arie_core.plugin_api.VectorStore`
(`upsert`, `search`, `delete`, `stats`):

```python
@plugin(slot="vector_store", name="chroma")
class ChromaVectorStore:
    def __init__(self, host: str = "localhost", collection: str = "arie") -> None: ...
```

```yaml
plugins:
  vector_store: {use: chroma, host: chroma.internal, collection: arie}
```

Keys other than `use` go to the constructor. `make test-contracts` runs the
`VectorStore` contract against every discovered store. A plugin that fails to import
is logged at startup with the reason. In Docker, a new plugin package needs `uv lock`
and an image rebuild. The bundled `qdrant` and `pgvector` stores are local SQLite
stand-ins and do not connect to those servers.

For local storage, the selected document offers **Delete PDF** (retain extracted
text for chat) and **Delete paper** (remove its PDF, metadata, and evidence).
Confirm the selected action in the pane. Pause sync and wait for active imports
before deleting. Files are staged before the database change and restored if
that change fails. A later sync can rediscover a fully deleted paper; import its
link again to restore a deleted PDF.

If import reports that the running API has no import route, restart the API with
the command above. During local development, add `--reload` so new routes are
loaded after source edits.

## Claim checker (Tier 3 entailment)

Quote matching proves a quotation exists in the paper, not that a summary of it is
faithful: "the model performs badly", citing "the model performs well", shares nearly
every word with its quote. Tier 3 closes that gap with a local natural-language
inference model, `cross-encoder/nli-deberta-v3-xsmall` (ONNX, CPU), that scores how
strongly each quotation entails its drafted claim:

```yaml
plugins:
  entailment: {use: nli-onnx, model: cross-encoder/nli-deberta-v3-xsmall, min_score: 0.5}
```

It runs in a recycled `arie-worker` process, warms up at startup, and checks a whole
answer in about 0.1 s. Claims scoring below `min_score` are dropped before review. On
paraphrases of source sentences it scores 0.90 or higher, and on polarity flips and
contradictions 0.01 or lower. While the model is loading or unavailable, model-written
claims cannot pass, and chat shows exact source excerpts with the reason. Set
`use: none` to rely on quote matching and the model reviewer only. To use another
model, implement `arie_core.plugin_api.Entailment` (`score`) as an `entailment`
plugin and recalibrate `min_score`. The design is recorded in
[ADR 0003](docs/adr/0003-quote-verifier-ladder.md).

## PDF parsers

`pymupdf4llm` (default) is fast and handles born-digital PDFs. `docling` adds layout
analysis, table structure and OCR, so scanned papers become searchable. It is much
slower and heavier, because it installs PyTorch models:

```sh
uv sync --extra docling
```

```yaml
plugins:
  parser: {use: docling, timeout_sec: 300, do_ocr: true}
```

Keys other than `use` and `timeout_sec` go to the parser's constructor. Each paper is
parsed in a worker process that is killed after `timeout_sec` (default 60) or 1.5 GB of
memory; that paper fails and the rest of the sync continues. The Docker image ships
only `pymupdf4llm`. To use Docling there, add `--extra docling` to both `uv sync`
lines in the `Dockerfile` and give Docker more memory than the 3 GB minimum.

## Answer prompts and source-grounding evaluations

Edit `prompts/qa.v1.yaml` to refine the planner, writer and independent evidence-review
instructions. New questions load the current playbook. Overviews request the
contribution, method, evaluation, findings and limitations when supported; absent
sections are omitted. `prompts/brief.v1.yaml` records the same fidelity policy for
briefs. No external knowledge may fill gaps in the saved paper.

Run `make check` for deterministic regressions covering typo search, exact source
offsets, foreign-paper isolation, fabricated quotes, changed numbers, missing
negation, unsupported paraphrases, claim-checker rejections and malformed review
verdicts.

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

## Development

| Command | Runs |
| --- | --- |
| `make check` | `ruff`, `lint-imports` (core imports no SDK or plugin), strict `pyright`, then the core, API and contract test suites |
| `make test-contracts` | Slot contracts and per-plugin tests: file storage, databases, sources, parsers, vector stores and the claim checker. No network is used. |
| `make web-check` | `oxlint`, TypeScript and Vitest for `web/` |
| `make eval-llm` | Source-grounding evaluation against the configured provider, with the claim checker enabled |

The real claim-checker model test runs only when the model is already cached. To run
it on purpose, point `ARIE_NLI_CACHE` at a Hugging Face cache directory that holds it.

GitHub Actions (`.github/workflows/ci.yml`) runs `make check`, the web checks and
build, and both Docker image builds on every pull request and push to `main`.

Architecture decisions (plugin slots, paper isolation, the verifier ladder, worker
processes) are recorded in [`docs/adr/`](docs/adr/README.md).
