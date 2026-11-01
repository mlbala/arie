from __future__ import annotations

import asyncio
import io
import os
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from arie_core.contracts import ChunkRow, ParsedDoc
from arie_core.plugin_api import ScrapedPaperInput

ALLOWED_CATEGORIES = (
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.MA",
    "cs.IR",
    "cs.CV",
    "cs.NE",
    "stat.ML",
)
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_HEADER_BYTES = 1024
CHUNK_SIZE_BYTES = 64 * 1024
R_AI = re.compile(
    r"\b(?:artificial intelligence|machine learning|deep learning|"
    r"neural network|large language model|llm|ai|ml)\b",
    re.IGNORECASE,
)


class IngestionStage(StrEnum):
    NORMALIZED = "normalized"
    DISCOVERED = "discovered"
    GATED = "gated"
    DOWNLOADING = "downloading"
    PARSED = "parsed"
    INDEXED = "indexed"
    BRIEFED = "briefed"
    QUARANTINED = "quarantined"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _DatabaseLike(Protocol):
    async def fetch(
        self,
        ctx: object | None,
        query: str,
        params: tuple[object, ...] = (),
    ) -> list[dict[str, Any]]: ...

    async def execute(
        self,
        ctx: object | None,
        query: str,
        params: tuple[object, ...] = (),
    ) -> int: ...


class StartupReconciler:
    """Reclaims stale leases and cleans orphaned partial-file downloads."""

    def __init__(
        self,
        *,
        db: _DatabaseLike | None = None,
        storage_root: str | Path = "data/papers",
    ) -> None:
        self.db: _DatabaseLike | None = db
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    async def reconcile(self) -> dict[str, int]:
        stats = {"expired_jobs": 0, "orphaned_parts": 0, "replayed_outbox": 0}
        if self.db is None:
            return stats
        db = self.db
        now = _utc_now()
        expired = await db.fetch(
            None,
            "SELECT job_id FROM jobs WHERE status = ? AND lease_until IS NOT NULL "
            "AND lease_until <= ?",
            ("leased", now),
        )
        for row in expired:
            await db.execute(
                None,
                (
                    "UPDATE jobs SET status = ?, lease_until = NULL, claim_token = NULL, "
                    "updated_at = ? WHERE job_id = ?"
                ),
                ("queued", now, row["job_id"]),
            )
            stats["expired_jobs"] += 1

        pending = await db.fetch(
            None,
            "SELECT id, payload_json FROM outbox WHERE status = ? ORDER BY id ASC",
            ("pending",),
        )
        for row in pending:
            await db.execute(
                None,
                "UPDATE outbox SET status = ?, leased_until = NULL, "
                "lease_token = NULL WHERE id = ?",
                ("done", row["id"]),
            )
            stats["replayed_outbox"] += 1

        for partial in sorted(self.storage_root.rglob("*.part")):
            if partial.is_file():
                partial.unlink(missing_ok=True)
                stats["orphaned_parts"] += 1
        return stats


class IngestionPipeline:
    """Resumable ingest pipeline with bounded queueing and explicit stage transitions."""

    def __init__(
        self,
        *,
        storage_root: str | Path = "data/papers",
        file_storage: object | None = None,
        db: object | None = None,
        max_queue_size: int = 4,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.file_storage = file_storage
        self.db = db
        self.queue: asyncio.Queue[ScrapedPaperInput] = asyncio.Queue(maxsize=max_queue_size)
        self._paper_states: dict[str, IngestionStage] = {}
        self._parse_failures: dict[str, int] = {}
        self._quarantined_papers: set[str] = set()
        self._seen: set[str] = set()

    @staticmethod
    def _normalize_paper(raw: ScrapedPaperInput | Mapping[str, object]) -> ScrapedPaperInput:
        if isinstance(raw, ScrapedPaperInput):
            return raw
        data = dict(raw)
        if "source" not in data:
            data["source"] = "arxiv"
        if "paper_id" not in data and "id" in data:
            data["paper_id"] = data["id"]
        return ScrapedPaperInput.model_validate(data)

    def normalize(self, raw: ScrapedPaperInput | Mapping[str, object]) -> ScrapedPaperInput:
        paper = self._normalize_paper(raw)
        self._paper_states.setdefault(paper.paper_id, IngestionStage.NORMALIZED)
        return paper

    def discover(
        self,
        raw_items: Iterable[ScrapedPaperInput | Mapping[str, object]],
    ) -> list[ScrapedPaperInput]:
        discovered: list[ScrapedPaperInput] = []
        seen_keys: set[str] = set()
        for raw in raw_items:
            paper = self.normalize(raw)
            dedup_key = f"{paper.paper_id}:{paper.dedup_hash}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            self._seen.add(dedup_key)
            self._paper_states[paper.paper_id] = IngestionStage.DISCOVERED
            discovered.append(paper)
        return discovered

    def is_ai_relevant(self, paper: ScrapedPaperInput) -> bool:
        metadata = paper.metadata or {}
        category = str(metadata.get("primary_category") or metadata.get("category") or "").strip()
        if category and category not in ALLOWED_CATEGORIES:
            return False
        text = " ".join(part for part in (paper.title, paper.abstract or "") if part)
        if category in ALLOWED_CATEGORIES:
            return True
        return bool(R_AI.search(text))

    def gate(self, paper: ScrapedPaperInput) -> bool:
        if self.is_quarantined(paper.paper_id):
            return False
        relevant = self.is_ai_relevant(paper)
        if relevant:
            self._paper_states[paper.paper_id] = IngestionStage.GATED
        return relevant

    @staticmethod
    def _candidate_hosts(url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":", 1)[0]
        return parsed.scheme, host

    async def _fetch_response(self, url: str) -> httpx.Response:
        async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
            return await client.get(url)

    async def download_pdf(self, paper: ScrapedPaperInput) -> str:
        self._paper_states[paper.paper_id] = IngestionStage.DOWNLOADING
        url = str(paper.pdf_url)
        scheme, host = self._candidate_hosts(url)
        if scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme for {paper.paper_id!r}: {scheme!r}")
        if host not in {"export.arxiv.org", "arxiv.org"}:
            raise ValueError(f"Host {host!r} is not allowed for {paper.paper_id!r}")

        response = await self._fetch_response(url)
        if response.is_redirect:
            raise ValueError("Redirects are rejected by the download gate")

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                length_value = int(content_length)
            except (TypeError, ValueError) as exc:
                # pragma: no cover - preserved for clear validation
                raise ValueError("Invalid Content-Length header") from exc
            if length_value > MAX_PDF_BYTES:
                raise ValueError("Content-Length exceeds the 25 MB limit")

        pending = bytearray()
        total = 0
        async for chunk in response.aiter_bytes(CHUNK_SIZE_BYTES):
            chunk_bytes = bytes(chunk)
            if total + len(chunk_bytes) > MAX_PDF_BYTES:
                raise ValueError("Downloaded PDF exceeds the 25 MB limit")
            pending.extend(chunk_bytes)
            total += len(chunk_bytes)

        if not pending[:MAX_HEADER_BYTES].startswith(b"%PDF-"):
            raise ValueError("Corrupted PDF header: missing %PDF-")

        return await self._persist_bytes(paper.paper_id, bytes(pending))

    async def _persist_bytes(self, paper_id: str, content: bytes) -> str:
        if self.file_storage is not None:
            put_stream = getattr(self.file_storage, "put_stream", None)
            write = getattr(self.file_storage, "write", None)
            if put_stream is not None:
                await put_stream(
                    None,
                    f"{paper_id}.pdf",
                    io.BytesIO(content),
                    expected_sha256=__import__("hashlib").sha256(content).hexdigest(),
                    expected_size=len(content),
                )
                return str(self.storage_root / f"{paper_id}.pdf")
            if write is not None:
                await write(None, f"{paper_id}.pdf", content)
                return str(self.storage_root / f"{paper_id}.pdf")

        target = self.storage_root / f"{paper_id}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(f"{target.suffix}.part")
        temp_target.write_bytes(content)
        os.replace(temp_target, target)
        return str(target)

    async def _run_worker_parse(self, file_path: str) -> ParsedDoc:
        content = await asyncio.to_thread(Path(file_path).read_bytes)
        text = content.decode("utf-8", errors="replace").strip() or "empty document"
        sections = [text]
        return ParsedDoc(
            paper_id=Path(file_path).stem,
            title="Parsed document",
            source="arxiv",
            sections=sections,
            metadata={"path": file_path},
        )

    def is_quarantined(self, paper_id: str) -> bool:
        return paper_id in self._quarantined_papers

    async def parse_document(self, paper_id: str, file_path: str) -> ParsedDoc:
        if self.is_quarantined(paper_id):
            raise RuntimeError(f"Paper {paper_id!r} has been quarantined")
        try:
            parsed = await self._run_worker_parse(file_path)
            if parsed.paper_id != paper_id:
                parsed = ParsedDoc(
                    paper_id=paper_id,
                    title=parsed.title,
                    source=parsed.source,
                    sections=parsed.sections,
                    metadata=parsed.metadata,
                )
            self._paper_states[paper_id] = IngestionStage.PARSED
            self._parse_failures.pop(paper_id, None)
            return parsed
        except Exception as exc:
            failure_count = self._parse_failures.get(paper_id, 0) + 1
            self._parse_failures[paper_id] = failure_count
            if failure_count >= 2:
                self._quarantined_papers.add(paper_id)
                self._paper_states[paper_id] = IngestionStage.QUARANTINED
                raise RuntimeError(
                    f"parse failed for {paper_id!r}; quarantined after two parse failures"
                ) from exc
            raise RuntimeError(f"parse failed for {paper_id!r}") from exc

    def stage_for(self, paper_id: str) -> IngestionStage:
        return self._paper_states.get(paper_id, IngestionStage.NORMALIZED)

    def chunk_text(
        self,
        text: str,
        *,
        min_tokens: int = 400,
        max_tokens: int = 600,
    ) -> list[str]:
        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]
        chunks: list[str] = []
        buffer: list[str] = []
        for paragraph in paragraphs:
            buffer.append(paragraph)
            word_count = sum(len(part.split()) for part in buffer)
            if word_count >= min_tokens or len(" ".join(buffer).split()) >= max_tokens:
                candidate = "\n\n".join(buffer)
                chunks.append(candidate)
                buffer = []
        if buffer:
            chunks.append("\n\n".join(buffer))
        if not chunks:
            chunks = [text.strip() or "empty document"]
        return chunks

    async def index_document(self, parsed: ParsedDoc) -> list[ChunkRow]:
        rows: list[ChunkRow] = []
        sections = parsed.sections or ["body"]
        for index, section in enumerate(sections):
            for chunk_index, chunk in enumerate(self.chunk_text(section)):
                rows.append(
                    ChunkRow(
                        chunk_id=f"{parsed.paper_id}:{index}:{chunk_index}",
                        paper_id=parsed.paper_id,
                        section=section if section else "body",
                        text=chunk,
                        order=chunk_index,
                        token_count=max(1, len(chunk.split())),
                    )
                )
        self._paper_states[parsed.paper_id] = IngestionStage.INDEXED
        return rows

    async def brief_document(self, paper_id: str, parsed: ParsedDoc) -> str:
        self._paper_states[paper_id] = IngestionStage.BRIEFED
        return f"Brief for {paper_id}: {parsed.title}"

    async def run(
        self,
        raw_items: Iterable[ScrapedPaperInput | Mapping[str, object]],
    ) -> list[ScrapedPaperInput]:
        accepted: list[ScrapedPaperInput] = []
        for item in self.discover(raw_items):
            if self.gate(item):
                accepted.append(item)
                await self.queue.put(item)
        return accepted

    async def process_next(self) -> ScrapedPaperInput:
        paper = await self.queue.get()
        try:
            file_path = await self.download_pdf(paper)
            parsed = await self.parse_document(paper.paper_id, file_path)
            await self.index_document(parsed)
            await self.brief_document(paper.paper_id, parsed)
            return paper
        finally:
            self.queue.task_done()


__all__ = [
    "ALLOWED_CATEGORIES",
    "R_AI",
    "IngestionPipeline",
    "IngestionStage",
    "StartupReconciler",
]
