from __future__ import annotations

import asyncio
import heapq
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml
from arie_core.budget import BudgetHandle
from arie_core.contracts import VectorRow
from arie_core.pipeline import IngestionPipeline
from arie_core.pipeline.chunker import chunk_spans
from arie_core.pipeline.topics import TOPICS, topic_tags
from arie_core.plugin_api import (
    Database,
    PagedSourcePlugin,
    PaperURLSource,
    ScrapedPaperInput,
    VectorStore,
    normalize_scraper_output,
)
from arie_core.registry.manager import ArieConfig, RegistryManager
from arie_core.registry.sync import SyncConfig
from arie_core.registry.sync import months_ago as months_ago
from arie_core.sched import supervised
from arie_core.search import score_paper
from arie_worker.embedding import EmbeddingPool
from arie_worker.extraction import ExtractionPool

from arie_api.providers import (
    AUTO_EMBEDDER,
    DEFAULT_LLM,
    LLMProviders,
    config_path,
    embedding_key_configured,
    resolve_embedder,
    save_plugin_selection,
)

Document = dict[str, Any]

logger = logging.getLogger(__name__)

# Chunks per embedding call: about 2 s on CPU, the longest a chat query waits behind it.
EMBED_BATCH = 16
# bge-small-en-v1.5 scores unrelated questions up to ~0.58 and paraphrases from ~0.6.
DEFAULT_MIN_SCORE = 0.6
# Saved chats: the most recent turns of the most recently used papers; older ones are dropped.
CHAT_PAPERS = 10
CHAT_TURNS = 10


@dataclass(frozen=True)
class CallContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


def context(trace_id: str, seconds: float = 120.0) -> CallContext:
    return CallContext(time.monotonic() + seconds, trace_id, BudgetHandle(max_wall_sec=seconds))


class ResearchRuntime:
    """Persist real papers and resumable sync jobs; supervise failures per paper and source."""

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        self.root = root
        self.config = config or ArieConfig.from_yaml(config_path(root))
        self.manager = RegistryManager(project_root=root, config=self.config)
        self.db = cast(Database, self.instance("database"))
        file_config = cast(dict[str, object], self.config.plugins.get("file_storage", {}))
        self.storage_root = root / str(file_config.get("root") or "data/papers")
        parser_config = cast(dict[str, object], self.config.plugins.get("parser", {}))
        self.pool = ExtractionPool(
            root,
            str(parser_config.get("use") or "pymupdf4llm"),
            self.config.runtime_qos.utility_worker_processes,
        )
        self.file_storage = self.instance("file_storage")
        self.jobs: dict[str, Document] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.import_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.lock = asyncio.Lock()
        self.url_source = cast(PaperURLSource, self.manager.instantiate_plugin("source", "arxiv"))
        self.subscribers: set[asyncio.Queue[Document]] = set()
        self.providers = LLMProviders(root, self.config, self.manager, self.lock)
        vector_name = str(
            cast(dict[str, object], self.config.plugins.get("vector_store", {})).get("use")
            or "none"
        )
        self.min_score = DEFAULT_MIN_SCORE
        # Vectors are comparable only within one embedder model; rows are tagged with this.
        self.embedder_id = "none"
        self.embedder_reason = ""
        self.vector_store: VectorStore | None = None
        self.embedding: EmbeddingPool | None = None
        self.vector_wakeup = asyncio.Event()
        self.vector_errors: dict[str, str] = {}
        if vector_name != "none" and resolve_embedder(self.config) is not None:
            self.vector_store = cast(VectorStore, self.instance("vector_store"))
            self.apply_embedder()

    def apply_embedder(self) -> None:
        """Adopt the embedder resolved from config; a new model re-embeds in the background.

        Until a paper is re-embedded, its old vectors no longer match ``embedder_id`` and
        semantic search returns nothing for it, so answers fall back to keyword retrieval.
        """
        choice = resolve_embedder(self.config)
        if self.vector_store is None or choice is None:
            return
        model = choice.settings.get("model")
        embedder_id = f"{choice.name}:{model}" if model else choice.name
        self.embedder_reason = choice.reason
        if embedder_id == self.embedder_id and self.embedding is not None:
            return
        previous = self.embedding
        self.embedding = EmbeddingPool(self.root, choice.name, choice.settings)
        self.embedder_id, self.min_score = embedder_id, choice.min_score
        self.vector_errors.clear()
        if previous is not None:
            previous.close()
        self.vector_wakeup.set()

    async def check_embedder(self, name: str) -> None:
        """Embed one probe text with ``name`` in a disposable worker; raise ``ValueError``."""
        if name not in {AUTO_EMBEDDER, *self.manager.get_available_plugins("embedder")}:
            raise ValueError("Unknown embedder")
        candidate = self.config.model_copy(deep=True)
        candidate.plugins["embedder"] = {"use": name}
        choice = resolve_embedder(candidate)
        if choice is None:
            raise ValueError("Unknown embedder")
        profile = self.config.embedding_providers.get(choice.name)
        if (
            profile
            and profile.api_key_env
            and not embedding_key_configured(choice.name, self.config)
        ):
            raise ValueError(
                f"{profile.api_key_env} is not configured. Add it to .env and restart ARIE."
            )
        pool = EmbeddingPool(self.root, choice.name, choice.settings)
        try:
            # First use may download a local model, so allow more than a chat-sized wait.
            vectors = await pool.embed(["ARIE embedder connection check"], kill_after=120.0)
        except Exception as exc:
            raise ValueError(f"{choice.name} embedding check failed: {exc}") from None
        finally:
            pool.close()
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError(f"{choice.name} returned no embedding")

    async def switch_embedder(self, name: str) -> Document:
        """Verify, save to arie.yaml, then adopt; the library re-embeds in the background."""
        if self.vector_store is None:
            raise ValueError("Vector search is off; set plugins.vector_store in arie.yaml")
        await self.check_embedder(name)
        selection: dict[str, object] = {"use": name}
        async with self.lock:
            await asyncio.to_thread(save_plugin_selection, self.root, "embedder", selection)
            self.config.plugins["embedder"] = selection
            self.apply_embedder()
        return {"slot": "embedder", "plugin": name, "embedder": self.embedder_id}

    def instance(self, slot: str, name: str | None = None) -> object:
        settings = cast(dict[str, object], self.config.plugins.get(slot, {}))
        if slot == "llm":
            provider = name or str(settings.get("use") or DEFAULT_LLM)
            return self.manager.instantiate_plugin(
                slot, provider, **self.providers.settings(provider)
            )
        kwargs = {key: value for key, value in settings.items() if key != "use"}
        if slot in {"database", "vector_store"} and "path" in kwargs:
            kwargs["db_path"] = self.root / str(kwargs.pop("path"))
        if slot == "file_storage" and "root" in kwargs:
            kwargs["root"] = self.root / str(kwargs["root"])
        return self.manager.instantiate_plugin(slot, name, **kwargs)

    async def initialize(self) -> None:
        ctx = context("startup")
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_library (
            paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
            published_at TEXT NOT NULL, summary_json TEXT NOT NULL, document_json TEXT NOT NULL,
            topics_text TEXT NOT NULL DEFAULT '|')""",
        )
        try:
            await self.db.fetch(ctx, "SELECT topics_text FROM arie_library LIMIT 1")
        except Exception:
            await self.db.execute(
                ctx, "ALTER TABLE arie_library ADD COLUMN topics_text TEXT NOT NULL DEFAULT '|'"
            )
        await self.db.execute(
            ctx,
            "CREATE INDEX IF NOT EXISTS arie_library_date ON arie_library(status, published_at)",
        )
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_vector_index (
            paper_id TEXT PRIMARY KEY, model TEXT NOT NULL, revision TEXT NOT NULL,
            vectors INTEGER NOT NULL, updated_at TEXT NOT NULL)""",
        )
        while True:
            untagged = await self.db.fetch(
                ctx, "SELECT document_json FROM arie_library WHERE topics_text = '|' LIMIT 50"
            )
            if not untagged:
                break
            for row in untagged:
                await self.save_paper(cast(Document, json.loads(str(row["document_json"]))))
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_chat_turns (
            turn_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question TEXT NOT NULL,
            events_json TEXT NOT NULL, created_at TEXT NOT NULL)""",
        )
        await self.db.execute(
            ctx,
            """CREATE INDEX IF NOT EXISTS arie_chat_turns_paper
            ON arie_chat_turns(paper_id, created_at)""",
        )
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_sync_runs (
            job_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)""",
        )
        rows = await self.db.fetch(
            ctx, "SELECT payload_json FROM arie_sync_runs ORDER BY updated_at"
        )
        for row in rows:
            job = cast(Document, json.loads(str(row["payload_json"])))
            self.jobs[job["job_id"]] = job
            if job["status"] in {"running", "queued"}:
                job.update(status="interrupted", message="Sync interrupted. Resume to continue.")
                await self.update_job(job)

    async def save_chat_turn(
        self, paper_id: str, question: str, events: Sequence[Document]
    ) -> None:
        """Store one answered turn as the exact events streamed, then enforce both limits.

        Keeps the last ``CHAT_TURNS`` turns of each paper and only the ``CHAT_PAPERS`` papers
        asked about most recently. Saved answers are display history; they never become
        evidence for later answers (I2).
        """
        ctx = context(f"chat:{paper_id}", 5.0)
        await self.db.execute(
            ctx,
            """INSERT INTO arie_chat_turns (turn_id, paper_id, question, events_json, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                paper_id,
                question[:2000],
                json.dumps(list(events)),
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.db.execute(
            ctx,
            """DELETE FROM arie_chat_turns WHERE paper_id = ? AND turn_id NOT IN (
            SELECT turn_id FROM arie_chat_turns WHERE paper_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?)""",
            (paper_id, paper_id, CHAT_TURNS),
        )
        await self.db.execute(
            ctx,
            """DELETE FROM arie_chat_turns WHERE paper_id NOT IN (
            SELECT paper_id FROM arie_chat_turns GROUP BY paper_id
            ORDER BY MAX(created_at) DESC LIMIT ?)""",
            (CHAT_PAPERS,),
        )

    async def chat_history(self, paper_id: str) -> list[Document]:
        """Return a paper's saved turns, oldest first, as ``{question, events}``."""
        rows = await self.db.fetch(
            context(f"chat:{paper_id}", 5.0),
            """SELECT question, events_json FROM arie_chat_turns WHERE paper_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?""",
            (paper_id, CHAT_TURNS),
        )
        return [
            {"question": str(row["question"]), "events": json.loads(str(row["events_json"]))}
            for row in reversed(rows)
        ]

    async def list_papers(
        self,
        offset: int = 0,
        status: str | None = None,
        source: str | None = None,
        month: str | None = None,
        domain: str | None = None,
        search: str | None = None,
        sort: str = "newest",
        *,
        year: int | None = None,
        months: int | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> Document:
        """Filter saved publication dates in UTC; include complete boundary days and months."""
        conditions = ["1 = 1"]
        params: list[object] = []
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        else:
            conditions.append("status IN ('INDEXED', 'BRIEFED')")
        if source:
            conditions.append("source = ?")
            params.append(source)
        if sum(value is not None for value in (month, year, months, start_month)) > 1:
            raise ValueError("Choose one publication window")
        if end_month and not start_month:
            raise ValueError("End month requires a start month")
        window: SyncConfig | None = None
        if month:
            window = SyncConfig(mode="month", year=int(month[:4]), month=int(month[5:]))
        elif year is not None:
            window = SyncConfig(mode="year", year=year)
        elif months is not None:
            window = SyncConfig(initial_months=months)
        elif start_month:
            window = SyncConfig(
                mode="range" if end_month else "since",
                start_month=start_month,
                end_month=end_month,
            )
        if window is not None:
            try:
                start, until = window.window(datetime.now(UTC))
            except ValueError:
                # Valid future-only filters have no published papers yet.
                return {"items": [], "next_cursor": None, "has_more": False}
            conditions.append("published_at >= ? AND published_at < ?")
            params.extend(
                [start.date().isoformat(), (until.date() + timedelta(days=1)).isoformat()]
            )
        if domain:
            if domain not in TOPICS:
                raise ValueError("Unknown research domain")
            conditions.append("topics_text LIKE ?")
            params.append(f"%|{domain}|%")
        ordering = {
            "newest": "published_at DESC, paper_id",
            "oldest": "published_at ASC, paper_id",
            "domain": "topics_text ASC, published_at DESC, paper_id",
        }
        if sort not in ordering:
            raise ValueError("Unknown library sort order")
        if search and search.strip():
            return await self.search_library(conditions, params, search, offset)
        params.extend([21, offset])
        rows = await self.db.fetch(
            context("library"),
            "SELECT summary_json FROM arie_library WHERE "
            + " AND ".join(conditions)
            + f" ORDER BY {ordering[sort]} LIMIT ? OFFSET ?",
            params,
        )
        items = [json.loads(str(row["summary_json"])) for row in rows[:20]]
        return {
            "items": items,
            "next_cursor": str(offset + 20) if len(rows) > 20 else None,
            "has_more": len(rows) > 20,
        }

    async def search_library(
        self,
        conditions: list[str],
        params: list[object],
        query: str,
        offset: int,
    ) -> Document:
        """Search all matching metadata pages; retain only the requested relevance-ranked slice."""
        best: list[tuple[float, str, str, Document]] = []
        after = ""

        def rank(rows: list[dict[str, object]]) -> list[tuple[float, str, str, Document]]:
            hits: list[tuple[float, str, str, Document]] = []
            for row in rows:
                paper = cast(Document, json.loads(str(row["summary_json"])))
                topics = " ".join(str(tag["label"]) for tag in paper.get("tags", []))
                score, reason = score_paper(
                    query, paper["title"], paper.get("abstract", ""), topics
                )
                if score:
                    paper["match_reason"] = reason
                    hits.append((score, paper["published_at"], paper["id"], paper))
            return hits

        while True:
            rows = await self.db.fetch(
                context("library-search"),
                "SELECT paper_id, summary_json FROM arie_library WHERE "
                + " AND ".join(conditions)
                + " AND paper_id > ? ORDER BY paper_id LIMIT 200",
                [*params, after],
            )
            if not rows:
                break
            hits = await asyncio.to_thread(rank, rows)
            best = heapq.nlargest(offset + 21, [*best, *hits], key=lambda item: item[:3])
            after = str(rows[-1]["paper_id"])
        page = best[offset : offset + 21]
        return {
            "items": [paper for _, _, _, paper in page[:20]],
            "has_more": len(page) > 20,
            "next_cursor": str(offset + 20) if len(page) > 20 else None,
        }

    async def get_paper(self, paper_id: str) -> Document | None:
        rows = await self.db.fetch(
            context(paper_id),
            "SELECT paper_id, document_json FROM arie_library WHERE paper_id = ?",
            (paper_id,),
        )
        rows = [row for row in rows if row["paper_id"] == paper_id]
        return cast(Document, json.loads(str(rows[0]["document_json"]))) if rows else None

    async def save_paper(self, paper: Document) -> None:
        summary = {
            key: paper[key]
            for key in (
                "id",
                "title",
                "source",
                "status",
                "published_at",
                "abstract",
                "relevance_gate",
                "overridden_by_user",
                "status_reason",
            )
        }
        paper["tags"] = topic_tags(
            str(paper["title"]),
            str(paper.get("abstract", "")),
            paper.get("metadata", {}),
            [str(page["text"]) for page in paper.get("pages", [])],
        )
        summary["tags"] = paper["tags"]
        summary["pdf_deleted"] = paper.get("pdf_deleted", False)
        summary["chunk_count"] = len(paper.get("chunks", []))
        summary["page_count"] = len(paper.get("pages", []))
        summary["revision"] = paper.get("dedup_hash", "")
        await self.db.execute(
            context(paper["id"]),
            """INSERT INTO arie_library
            (paper_id, source, status, published_at, summary_json, document_json, topics_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET source=excluded.source, status=excluded.status,
            published_at=excluded.published_at, summary_json=excluded.summary_json,
            document_json=excluded.document_json, topics_text=excluded.topics_text""",
            (
                paper["id"],
                paper["source"],
                paper["status"],
                paper["published_at"],
                json.dumps(summary),
                json.dumps(paper),
                "|" + "|".join(sorted(str(tag["tag"]) for tag in paper["tags"])) + "|",
            ),
        )

    async def delete_paper(self, paper_id: str, pdf_only: bool = False) -> None:
        """Delete local data; restore the staged PDF if the database write fails."""
        async with self.lock:
            if any(job["status"] in {"running", "queued"} for job in self.jobs.values()):
                raise RuntimeError("Pause sync and wait for active imports before deleting files")
            paper = await self.get_paper(paper_id)
            if paper is None:
                raise ValueError("Paper not found")
            storage = cast(dict[str, object], self.config.plugins.get("file_storage", {}))
            if storage.get("use") != "local":
                raise RuntimeError("File deletion is currently available for local storage only")
            path = Path(str(paper.get("file_path", "")))
            tombstone: Path | None = None

            def stage() -> Path | None:
                if not paper.get("file_path"):
                    return None
                resolved = path.resolve()
                if not resolved.is_relative_to(self.storage_root.resolve()):
                    raise ValueError("Stored PDF path is outside the paper storage directory")
                if not resolved.is_file():
                    return None
                target = resolved.with_suffix(f".delete-{uuid.uuid4().hex}")
                os.replace(resolved, target)
                return target

            tombstone = await asyncio.to_thread(stage)
            try:
                if pdf_only:
                    paper.update(pdf_deleted=True)
                    paper.pop("file_path", None)
                    paper.pop("pdf_url", None)
                    await self.save_paper(paper)
                else:
                    await self.db.execute(
                        context("delete-paper"),
                        "DELETE FROM arie_library WHERE paper_id = ?",
                        (paper_id,),
                    )
            except BaseException:
                if tombstone is not None:
                    await asyncio.to_thread(os.replace, tombstone, path)
                raise
            if tombstone is not None:
                await asyncio.to_thread(tombstone.unlink, missing_ok=True)
            if not pdf_only:
                await self._forget_vectors(paper_id)
                await self.db.execute(
                    context("delete-chat"),
                    "DELETE FROM arie_chat_turns WHERE paper_id = ?",
                    (paper_id,),
                )

    async def update_job(self, job: Document, **changes: Any) -> None:
        job.update(changes, updated_at=datetime.now(UTC).isoformat())
        self.jobs.pop(job["job_id"], None)
        self.jobs[job["job_id"]] = job
        await self.db.execute(
            context(job["job_id"]),
            """INSERT INTO arie_sync_runs VALUES (?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET updated_at=excluded.updated_at,
            payload_json=excluded.payload_json""",
            (job["job_id"], job["updated_at"], json.dumps(job)),
        )
        for queue in self.subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(dict(job))

    async def save_sync_settings(self, settings: SyncConfig) -> None:
        """Atomically replace only the YAML sync block; leave runtime defaults intact on failure."""
        settings.window(datetime.now(UTC))

        def save() -> None:
            path = config_path(self.root)
            original = path.read_text(encoding="utf-8")
            block = yaml.safe_dump(
                {"sync": settings.model_dump(exclude_none=True)}, sort_keys=False
            )
            pattern = r"(?m)^sync:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*"
            changed, count = re.subn(pattern, lambda _: block, original, count=1)
            if not count:
                changed = original.rstrip() + "\n" + block
            # Validate the complete candidate before replacing a user's configuration.
            ArieConfig.model_validate(yaml.safe_load(changed))
            descriptor, temporary = tempfile.mkstemp(
                prefix=".arie-", suffix=".yaml", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(changed)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)

        async with self.lock:
            await asyncio.to_thread(save)
            self.config.sync = settings

    async def start_sync(
        self,
        source: str = "all",
        settings: SyncConfig | None = None,
        resume_job_id: str | None = None,
    ) -> Document:
        """Snapshot settings per run; resume only matching windows, preserving page offsets."""
        enabled = cast(list[str], self.config.plugins.get("sources", ["arxiv"]))
        if source != "all" and source not in enabled:
            raise ValueError(f"Source {source!r} is not enabled in arie.yaml")
        selected = settings or self.config.sync
        snapshot = selected.model_dump(exclude_none=True)
        async with self.lock:
            active = next(
                (
                    job
                    for job in self.jobs.values()
                    if job.get("kind") != "import" and job["status"] in {"running", "queued"}
                ),
                None,
            )
            if active:
                if resume_job_id == active["job_id"] or (
                    active["source"] == source and active.get("settings", snapshot) == snapshot
                ):
                    return active
                raise RuntimeError("Pause the active sync before starting a different window")
            interrupted = next(
                (
                    job
                    for job in reversed(list(self.jobs.values()))
                    if job.get("kind") != "import"
                    and job["status"] == "interrupted"
                    and (
                        job["job_id"] == resume_job_id
                        if resume_job_id
                        else (job["source"] == source and job.get("settings", snapshot) == snapshot)
                    )
                ),
                None,
            )
            if resume_job_id and interrupted is None:
                raise ValueError("No paused sync with that ID is available")
            job: Document
            if interrupted:
                job = interrupted
                job.update(
                    status="queued",
                    message="Resuming sync",
                    pause_requested=False,
                    limit_reached=False,
                    segment_start=job["processed"],
                )
            else:
                now = datetime.now(UTC)
                since, until = selected.window(now)
                completed = [
                    job
                    for job in self.jobs.values()
                    if job.get("kind") != "import"
                    and job["status"] == "completed"
                    and job["source"] == source
                    and job.get("settings") == snapshot
                ]
                if selected.mode == "relative" and completed:
                    since = max(
                        since, datetime.fromisoformat(completed[-1]["until"]) - timedelta(days=1)
                    )
                job = dict(
                    job_id=f"sync-{uuid.uuid4().hex[:12]}",
                    kind="sync",
                    status="queued",
                    source=source,
                    sources=enabled if source == "all" else [source],
                    settings=snapshot,
                    since=since.isoformat(),
                    until=until.isoformat(),
                    phase="discovery",
                    message="Waiting to discover papers",
                    source_index=0,
                    cursor=None,
                    page_offset=0,
                    segment_start=0,
                    discovered=0,
                    processed=0,
                    indexed=0,
                    skipped=0,
                    failed=0,
                    duplicates=0,
                    source_errors=[],
                    current_paper=None,
                )
                self.jobs[job["job_id"]] = job
            await self.update_job(job)
            self.queue.put_nowait(job["job_id"])
            return job

    async def start_import(self, url: str) -> Document:
        """Queue a validated single-paper import independently of the batch date window."""
        canonical = self.url_source.canonical_pdf_url(url)
        async with self.lock:
            matching = [
                job
                for job in self.jobs.values()
                if job.get("kind") == "import" and job.get("url") == canonical
            ]
            if matching and matching[-1]["status"] in {"running", "queued"}:
                return matching[-1]
            if self.import_queue.full():
                raise RuntimeError("The import queue is full. Wait for a paper to finish.")
            job: Document
            if matching and matching[-1]["status"] == "interrupted":
                job = matching[-1]
                job.update(status="queued", pause_requested=False, error=None)
            else:
                now = datetime.now(UTC).isoformat()
                job = dict(
                    job_id=f"import-{uuid.uuid4().hex[:12]}",
                    kind="import",
                    url=canonical,
                    status="queued",
                    source="arxiv",
                    sources=["arxiv"],
                    since=now,
                    until=now,
                    phase="discovery",
                    message="Fetching arXiv paper metadata",
                    paper_id=None,
                    current_paper=None,
                    discovered=0,
                    processed=0,
                    indexed=0,
                    skipped=0,
                    failed=0,
                    duplicates=0,
                    source_errors=[],
                )
            await self.update_job(job)
            self.import_queue.put_nowait(job["job_id"])
            return job

    async def run_imports(self) -> None:
        """Admit at most two user imports while batch discovery continues separately."""
        async with asyncio.TaskGroup() as group:
            for _ in range(2):
                group.create_task(supervised(self._import_loop)())

    async def _import_loop(self) -> None:
        while True:
            job_id = await self.import_queue.get()
            job = self.jobs[job_id]
            try:
                await self.update_job(job, status="running", phase="discovery")
                paper = await self.url_source.fetch_paper(context(job_id), job["url"])
                paper = normalize_scraper_output([paper], "arxiv")[0]
                await self.update_job(
                    job,
                    paper_id=paper.paper_id,
                    current_paper=paper.title,
                    since=paper.published_at,
                    discovered=1,
                )
                await self._supervised_paper(job, paper, self.url_source.allowed_hosts)
                success = bool(job["indexed"] or job["duplicates"])
                await self.update_job(
                    job,
                    status="completed" if success else "failed",
                    phase="complete",
                    message=(
                        "Paper ready to chat"
                        if success
                        else job.get("error") or "This paper did not pass the ingestion gates"
                    ),
                )
            except asyncio.CancelledError:
                await self.update_job(
                    job,
                    status="interrupted",
                    message="Import interrupted; submit the URL to resume",
                )
                raise
            except Exception as exc:
                await self.update_job(
                    job, status="failed", phase="complete", error=str(exc), message=str(exc)
                )
            finally:
                self.import_queue.task_done()

    async def run(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs[job_id]
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                await self.update_job(
                    job, status="interrupted", message="Sync interrupted; resume available"
                )
                raise
            except Exception as exc:
                await self.update_job(job, status="failed", error=str(exc), message=str(exc))
            finally:
                self.queue.task_done()

    async def _run_job(self, job: Document) -> None:
        settings = SyncConfig.model_validate(job.get("settings", self.config.sync.model_dump()))
        await self.update_job(job, status="running")
        for index in range(job["source_index"], len(job["sources"])):
            source_name = str(job["sources"][index])
            try:
                source = cast(
                    PagedSourcePlugin,
                    self.manager.instantiate_plugin("source", source_name),
                )
                while True:
                    if job.get("pause_requested") or (
                        job["processed"] - job.get("segment_start", 0) >= settings.max_papers
                    ):
                        await self.update_job(
                            job,
                            status="interrupted",
                            limit_reached=not job.get("pause_requested", False),
                            message="Sync paused; resume to process the next batch",
                        )
                        return
                    await self.update_job(
                        job,
                        phase="discovery",
                        current_paper=None,
                        message=f"Discovering {source_name} papers",
                    )
                    page = await source.fetch_page(
                        context(job["job_id"]),
                        job["since"],
                        job["until"],
                        job["cursor"],
                        settings.batch_size,
                    )
                    papers = normalize_scraper_output(page.papers, source_name)
                    offset = job.get("page_offset", 0)
                    if not job.get("page_discovered"):
                        job["discovered"] += len(papers)
                        job["page_discovered"] = True
                    for position, paper in enumerate(papers[offset:], start=offset):
                        if job["processed"] - job.get("segment_start", 0) >= settings.max_papers:
                            await self.update_job(
                                job,
                                status="interrupted",
                                limit_reached=True,
                                message="Paper limit reached; resume to process the next batch",
                            )
                            return
                        await self._supervised_paper(job, paper, source.allowed_hosts)
                        job["page_offset"] = position + 1
                        if job.get("pause_requested"):
                            await self.update_job(
                                job, status="interrupted", message="Sync paused; resume available"
                            )
                            return
                    await self.update_job(
                        job, cursor=page.next_cursor, page_offset=0, page_discovered=False
                    )
                    if page.next_cursor is None:
                        break
            except Exception as exc:
                job["source_errors"].append(f"{source_name}: {exc}")
            await self.update_job(
                job, source_index=index + 1, cursor=None, page_offset=0, page_discovered=False
            )
        status = (
            "completed"
            if not job["source_errors"] and not job["failed"]
            else "completed_with_errors"
        )
        if job["source_errors"] and not job["indexed"]:
            status = "failed"
        message = (
            f"Sync finished: {job['indexed']} ready, {job['duplicates']} already in library, "
            f"{job['skipped']} skipped, {job['failed']} failed"
        )
        await self.update_job(
            job, status=status, phase="complete", current_paper=None, message=message
        )

    async def _supervised_paper(
        self, job: Document, paper: ScrapedPaperInput, hosts: tuple[str, ...]
    ) -> None:
        document: Document | None = None
        try:
            published = datetime.fromisoformat((paper.published_at or "").replace("Z", "+00:00"))
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            if job.get("kind") != "import" and not datetime.fromisoformat(
                job["since"]
            ) <= published <= datetime.fromisoformat(job["until"]):
                job["skipped"] += 1
                return
            existing = await self.get_paper(paper.paper_id)
            if (
                existing
                and existing.get("dedup_hash") == paper.dedup_hash
                and existing["status"] in {"INDEXED", "BRIEFED"}
                and not (job.get("kind") == "import" and existing.get("pdf_deleted"))
            ):
                job["duplicates"] += 1
                return
            if (
                existing
                and existing.get("dedup_hash") == paper.dedup_hash
                and existing["status"] == "QUARANTINED"
            ):
                job["skipped"] += 1
                return
            document = dict(
                id=paper.paper_id,
                title=paper.title,
                source=paper.source,
                abstract=paper.abstract or "",
                published_at=published.isoformat(),
                status="DISCOVERED",
                status_reason="",
                relevance_gate=False,
                overridden_by_user=0,
                dedup_hash=paper.dedup_hash,
                source_pdf_url=paper.pdf_url,
                metadata=paper.metadata,
                authors=paper.authors,
                pages=[],
                chunks=[],
                outline=[],
                brief={"parts": []},
                verified_5_part_brief=[],
            )
            if (
                existing
                and existing.get("dedup_hash") == paper.dedup_hash
                and existing["status"] in {"DOWNLOADING", "PARSED"}
            ):
                document = existing
            pipeline = IngestionPipeline(
                storage_root=self.storage_root,
                file_storage=self.file_storage,
                allowed_hosts=hosts,
                parse_worker=self.pool.parse,
                ctx=context(paper.paper_id),
            )
            if not pipeline.gate(paper):
                document.update(
                    status="SKIPPED", status_reason="RELEVANCE: outside enabled AI topics"
                )
                job["skipped"] += 1
                await self.save_paper(document)
                return
            cached_path = Path(str(document.get("file_path", "")))

            def reusable_pdf() -> bool:
                path = cached_path.resolve()
                return path.is_relative_to(self.storage_root.resolve()) and path.is_file()

            if not await asyncio.to_thread(reusable_pdf):
                document.update(status="GATED", relevance_gate=True)
                await self.save_paper(document)
                await self.update_job(
                    job,
                    phase="download",
                    current_paper=paper.title,
                    message="Downloading and validating PDF",
                )
                path = await pipeline.download_pdf(paper)
                document.update(status="DOWNLOADING", file_path=path)
                await self.save_paper(document)
            else:
                path = str(cached_path)
            pages: list[Document] = document.get("pages", [])
            if not pages:
                await self.update_job(
                    job,
                    phase="extraction",
                    current_paper=paper.title,
                    message="Extracting PDF pages, tables and equations",
                )
                try:
                    parsed = await pipeline.parse_document(paper.paper_id, path)
                except RuntimeError:
                    parsed = await pipeline.parse_document(paper.paper_id, path)
                pages = [{"number": i + 1, "text": text} for i, text in enumerate(parsed.sections)]
                document.update(
                    status="PARSED",
                    pages=pages,
                    extraction_metadata=parsed.metadata,
                    outline=json.loads(parsed.metadata.get("outline_json", "[]")),
                )
                await self.save_paper(document)
            await self.update_job(
                job, phase="indexing", message="Building page-scoped evidence chunks"
            )
            chunks: list[Document] = []
            for page in pages:
                # Every offset addresses the canonical extracted Markdown for this exact page.
                for start, end in chunk_spans(page["text"]):
                    chunks.append(
                        dict(
                            paper_id=paper.paper_id,
                            chunk_id=f"{paper.paper_id}:{len(chunks)}",
                            page_start=page["number"],
                            char_start=start,
                            char_end=end,
                            text=page["text"][start:end],
                        )
                    )
            if not chunks:
                raise ValueError("PDF extraction produced no searchable evidence")
            document.update(
                status="INDEXED",
                chunks=chunks,
                pdf_url=f"/v1/papers/{paper.paper_id}/pdf",
                pdf_deleted=False,
                status_reason="PDF extracted and indexed",
            )
            await self.save_paper(document)
            self.vector_wakeup.set()
            job["indexed"] += 1
        except Exception as exc:
            job["failed"] += 1
            if document is not None:
                document.update(
                    status="QUARANTINED" if "quarantined" in str(exc) else "FAILED",
                    status_reason=str(exc),
                )
                await self.save_paper(document)
            await self.update_job(job, error=f"{paper.title}: {exc}")
        finally:
            job["processed"] += 1
            await self.update_job(job)

    async def events(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[Document] = asyncio.Queue(maxsize=64)
        self.subscribers.add(queue)
        try:
            while True:
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: update\ndata: {json.dumps(job)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            self.subscribers.discard(queue)

    async def _vector_backlog(self, ctx: CallContext) -> tuple[list[str], int]:
        """Return chat-ready papers lacking current vectors, and the chat-ready total."""
        rows = await self.db.fetch(
            ctx,
            """SELECT l.paper_id, l.summary_json, v.model AS vector_model,
            v.revision AS vector_revision FROM arie_library l
            LEFT JOIN arie_vector_index v ON v.paper_id = l.paper_id
            WHERE l.status IN ('INDEXED', 'BRIEFED') ORDER BY l.published_at DESC""",
        )
        pending: list[str] = []
        for row in rows:
            summary = cast(Document, json.loads(str(row["summary_json"])))
            revision = f"{summary.get('revision', '')}:{summary.get('chunk_count', 0)}"
            if (row["vector_model"], row["vector_revision"]) != (self.embedder_id, revision):
                pending.append(str(row["paper_id"]))
        return pending, len(rows)

    async def index_paper_vectors(self, paper_id: str) -> None:
        """Replace one paper's vectors in batches, then record the embedder and revision.

        The marker is written last, so an interrupted run is retried by the next backlog scan.
        """
        # Capture the embedder: a switch mid-run must not tag old-model vectors as new.
        embedding, embedder_id = self.embedding, self.embedder_id
        if self.vector_store is None or embedding is None:
            return
        paper = await self.get_paper(paper_id)
        if paper is None or paper["status"] not in {"INDEXED", "BRIEFED"}:
            return
        chunks: list[Document] = [
            chunk
            for chunk in paper.get("chunks", [])
            if chunk.get("paper_id") == paper_id
            and chunk.get("chunk_id")
            and str(chunk.get("text", "")).strip()
        ]
        ctx = context(f"vectors:{paper_id}", 600.0)
        await self.vector_store.delete(ctx, filters={"paper_id": paper_id})
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = await embedding.embed([str(chunk["text"]) for chunk in batch])
            await self.vector_store.upsert(
                ctx,
                [
                    VectorRow(
                        id=str(chunk["chunk_id"]),
                        vector=vector,
                        metadata={
                            "paper_id": paper_id,
                            "model": embedder_id,
                            "page": str(chunk.get("page_start", "")),
                        },
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
        await self.db.execute(
            ctx,
            """INSERT INTO arie_vector_index (paper_id, model, revision, vectors, updated_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(paper_id) DO UPDATE SET model=excluded.model,
            revision=excluded.revision, vectors=excluded.vectors, updated_at=excluded.updated_at""",
            (
                paper_id,
                embedder_id,
                f"{paper.get('dedup_hash', '')}:{len(paper.get('chunks', []))}",
                len(chunks),
                datetime.now(UTC).isoformat(),
            ),
        )

    async def run_vector_index(self) -> None:
        """Keep vectors in step with the library: backfill, re-embed on model change, retry.

        Wakes at startup, whenever sync indexes a paper, and every 5 minutes while any paper
        failed. A failing paper stays keyword-searchable; its error is shown in the monitor.
        """
        if self.vector_store is None or self.embedding is None:
            return
        warmed = ""
        while True:
            self.vector_wakeup.clear()
            pending, _ = await self._vector_backlog(context("vector-backlog", 10.0))
            try:
                if pending and warmed != self.embedder_id:
                    # The first load may download the model, which can exceed a batch timeout.
                    await self.embedding.embed(["warm-up"], kill_after=600.0)
                    warmed = self.embedder_id
                    self.vector_errors.pop("*", None)
                for paper_id in pending:
                    try:
                        await self.index_paper_vectors(paper_id)
                        self.vector_errors.pop(paper_id, None)
                    except Exception as exc:
                        logger.warning("Vector indexing failed for %s", paper_id, exc_info=True)
                        self.vector_errors[paper_id] = str(exc) or type(exc).__name__
            except Exception as exc:
                logger.warning("Embedder %s failed to load", self.embedder_id, exc_info=True)
                self.vector_errors["*"] = str(exc) or type(exc).__name__
            try:
                timeout = 300.0 if self.vector_errors else None
                await asyncio.wait_for(self.vector_wakeup.wait(), timeout)
            except TimeoutError:
                pass

    async def semantic_search(self, paper_id: str, queries: Sequence[str]) -> list[str]:
        """Rank one paper's chunk IDs by best cosine similarity to any query, best first.

        Returns [] when vectors are off or the paper has no vectors for this embedder.
        Hits below ``min_score`` are dropped and every hit is re-checked against
        ``paper_id`` (I1). Embedder or store errors propagate to the caller.
        """
        embedding, embedder_id, floor = self.embedding, self.embedder_id, self.min_score
        if self.vector_store is None or embedding is None or not queries:
            return []
        ctx = context(f"semantic:{paper_id}", 6.0)
        marker = await self.db.fetch(
            ctx, "SELECT model FROM arie_vector_index WHERE paper_id = ?", (paper_id,)
        )
        if not marker or marker[0]["model"] != embedder_id:
            return []
        best: dict[str, float] = {}
        for vector in await embedding.embed(list(queries), kill_after=6.0):
            hits = await self.vector_store.search(
                ctx, vector, 8, filters={"paper_id": paper_id, "model": embedder_id}
            )
            for hit in hits:
                if hit.payload.get("paper_id") == paper_id and hit.score >= floor:
                    best[hit.id] = max(best.get(hit.id, hit.score), hit.score)
        return sorted(best, key=best.__getitem__, reverse=True)

    async def vector_status(self) -> Document:
        """Report measured index coverage for the configured embedder and vector store."""
        provider = cast(dict[str, object], self.config.plugins.get("vector_store", {})).get("use")
        status: Document = {
            "provider": provider,
            "embedder": self.embedder_id,
            "embedder_reason": self.embedder_reason,
            "status": "disabled",
            "size_bytes": None,
            "vectors": None,
            "papers_indexed": None,
            "papers_total": None,
            "detail": "Vector search is off; set plugins.embedder and plugins.vector_store "
            "in arie.yaml. Answers use keyword retrieval.",
        }
        if self.vector_store is None:
            return status
        ctx = context("vector-health", 3.0)
        stats = await self.vector_store.stats(ctx)
        pending, total = await self._vector_backlog(ctx)
        failed = [paper_id for paper_id in pending if paper_id in self.vector_errors]
        error = self.vector_errors.get("*") or next(
            (self.vector_errors[paper_id] for paper_id in failed), None
        )
        status.update(
            status="degraded" if error else "indexing" if pending else "healthy",
            size_bytes=stats.size_bytes,
            vectors=stats.vectors,
            papers_indexed=total - len(pending),
            papers_total=total,
            detail=f"{total - len(pending)} of {total} papers embedded with {self.embedder_id}"
            + (f"; {len(failed) or 'all'} failing: {error}" if error else ""),
        )
        return status

    async def _forget_vectors(self, paper_id: str) -> None:
        """Best-effort cleanup after a paper is deleted; orphans never match a live paper."""
        if self.vector_store is None:
            return
        ctx = context(f"forget:{paper_id}", 10.0)
        try:
            await self.vector_store.delete(ctx, filters={"paper_id": paper_id})
            await self.db.execute(
                ctx, "DELETE FROM arie_vector_index WHERE paper_id = ?", (paper_id,)
            )
        except Exception:
            logger.warning("Could not remove vectors of deleted paper %s", paper_id, exc_info=True)
        self.vector_errors.pop(paper_id, None)

    def close(self) -> None:
        self.pool.close()
        if self.embedding is not None:
            self.embedding.close()
