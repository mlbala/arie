"""The paper library: listing, search, loading, saving and deleting papers."""

from __future__ import annotations

import asyncio
import heapq
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from arie_core.pipeline.topics import TOPICS, topic_tags
from arie_core.registry.sync import SyncConfig
from arie_core.registry.sync import months_ago as months_ago
from arie_core.search import score_paper

from arie_api.runtime.base import (
    Document,
    RuntimeBase,
    context,
    logger,
)


class LibraryStore(RuntimeBase):
    """The paper library: listing, search, loading, saving and deleting papers."""

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
            conditions.append("status = 'INDEXED'")
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
                await self.db.execute(
                    context("delete-pack"),
                    "DELETE FROM arie_paper_packs WHERE paper_id = ?",
                    (paper_id,),
                )
                await self.db.execute(
                    context("delete-guide"),
                    "DELETE FROM arie_paper_guides WHERE paper_id = ?",
                    (paper_id,),
                )
                await self.evaluations.forget(paper_id)

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
