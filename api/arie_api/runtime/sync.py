"""Sync and import jobs: discovery, download, extraction and live events."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml
from arie_core.pipeline import IngestionPipeline
from arie_core.pipeline.chunker import chunk_spans
from arie_core.plugin_api import (
    PagedSourcePlugin,
    ScrapedPaperInput,
    normalize_scraper_output,
)
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import SyncConfig
from arie_core.registry.sync import months_ago as months_ago
from arie_core.sched import supervised

from arie_api.providers import (
    config_path,
)
from arie_api.runtime.base import (
    Document,
    context,
    logger,
)
from arie_api.runtime.vectors import VectorIndex


class SyncJobs(VectorIndex):
    """Sync and import jobs: discovery, download, extraction and live events."""

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
                and existing["status"] == "INDEXED"
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
            try:
                await self.ensure_pack(document)
            except Exception:
                # The paper is indexed; the pack is built on first use instead.
                logger.warning(
                    "Could not build the paper pack for %s", paper.paper_id, exc_info=True
                )
            self.vector_wakeup.set()
            # The guide builder (``guides.py``) writes this paper's guide when no answer runs.
            self.guide_pending[paper.paper_id] = None
            self.guide_wakeup.set()
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
