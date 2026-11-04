from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

import httpx

from arie_core.contracts import ParsedDoc
from arie_core.plugin_api import Ctx, FileStorage, ScrapedPaperInput

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
    QUARANTINED = "quarantined"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class IngestionPipeline:
    """Resumable ingest pipeline with bounded queueing and explicit stage transitions."""

    def __init__(
        self,
        *,
        storage_root: str | Path = "data/papers",
        file_storage: object | None = None,
        db: object | None = None,
        max_queue_size: int = 4,
        allowed_hosts: tuple[str, ...] = ("export.arxiv.org", "arxiv.org"),
        parse_worker: Callable[[str], Awaitable[ParsedDoc]] | None = None,
        ctx: Ctx | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.allowed_hosts = allowed_hosts
        self.parse_worker = parse_worker
        self.ctx = ctx
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
        client = httpx.AsyncClient(follow_redirects=False, timeout=30.0)
        try:
            response = await client.send(client.build_request("GET", url), stream=True)
        except BaseException:
            await client.aclose()
            raise
        response.extensions["arie_client"] = client
        return response

    async def download_pdf(self, paper: ScrapedPaperInput) -> str:
        """Stream a bounded PDF to a temporary file; failures never promote partial data."""
        self._paper_states[paper.paper_id] = IngestionStage.DOWNLOADING
        url = str(paper.pdf_url)
        parsed_url = urlparse(url)
        scheme, host = self._candidate_hosts(url)
        if scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {scheme!r}")
        if (
            host not in self.allowed_hosts
            or parsed_url.username
            or parsed_url.password
            or parsed_url.port not in {None, 80, 443}
        ):
            raise ValueError(f"Host {host!r} is not allowed")
        date = datetime.fromisoformat(
            (paper.published_at or _utc_now()).replace("Z", "+00:00")
        ).strftime("%Y/%m/%d")
        key = f"{date}/{paper.storage_hash}.pdf"
        target = self.storage_root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".download.part")
        response = await self._fetch_response(url)
        try:
            if response.is_redirect:
                raise ValueError("Redirects are rejected by the download gate")
            if isinstance(response, httpx.Response):  # pyright: ignore[reportUnnecessaryIsInstance]
                response.raise_for_status()
            length = response.headers.get("Content-Length")
            expected = int(length) if length is not None else None
            if expected is not None and (expected < 0 or expected > MAX_PDF_BYTES):
                raise ValueError("Content-Length exceeds the 25 MB limit")
            total = 0
            header = bytearray()
            digest = hashlib.sha256()
            with partial.open("wb") as handle:
                async for chunk in response.aiter_bytes(CHUNK_SIZE_BYTES):
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise ValueError("Downloaded PDF exceeds the 25 MB limit")
                    if len(header) < MAX_HEADER_BYTES:
                        header.extend(chunk[: MAX_HEADER_BYTES - len(header)])
                    if len(header) >= MAX_HEADER_BYTES and b"%PDF-" not in header:
                        raise ValueError("Corrupted PDF header: missing %PDF-")
                    handle.write(chunk)
                    digest.update(chunk)
                if b"%PDF-" not in header:
                    raise ValueError("Corrupted PDF header: missing %PDF-")
                if expected is not None and expected != total:
                    raise ValueError("Incomplete PDF download")
                handle.flush()
                os.fsync(handle.fileno())
            if self.file_storage is not None:
                put_stream = getattr(self.file_storage, "put_stream", None)
                if put_stream is None:
                    # The basic storage slot is bounded by the same 25 MB gate.
                    storage = cast(FileStorage, self.file_storage)
                    await storage.write(cast(Ctx, self.ctx), key, partial.read_bytes())
                else:
                    with partial.open("rb") as stream:
                        await put_stream(
                            self.ctx,
                            key,
                            cast(BinaryIO, stream),
                            expected_sha256=digest.hexdigest(),
                            expected_size=total,
                        )
            os.replace(partial, target)
            return str(target)
        finally:
            partial.unlink(missing_ok=True)
            if isinstance(response, httpx.Response):  # pyright: ignore[reportUnnecessaryIsInstance]
                await response.aclose()
                client = response.extensions.get("arie_client")
                if isinstance(client, httpx.AsyncClient):
                    await client.aclose()

    async def _run_worker_parse(self, file_path: str) -> ParsedDoc:
        if self.parse_worker is None:
            raise RuntimeError("No isolated PDF parser worker is configured")
        return await self.parse_worker(file_path)

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


__all__ = [
    "ALLOWED_CATEGORIES",
    "R_AI",
    "IngestionPipeline",
    "IngestionStage",
]
