from __future__ import annotations

from pathlib import Path

import pytest
from arie_core.contracts import ParsedDoc
from arie_core.pipeline import IngestionPipeline
from arie_core.pipeline.pipeline import MAX_PDF_BYTES
from arie_core.plugin_api import ScrapedPaperInput


class DummyResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        redirect: bool = False,
    ) -> None:
        self.headers = headers or {}
        self._chunks = chunks or []
        self._redirect = redirect

    @property
    def is_redirect(self) -> bool:
        return self._redirect

    async def aiter_bytes(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_download_rejects_oversized_content_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path)
    paper = ScrapedPaperInput(
        paper_id="oversize-1",
        title="Oversize",
        source="arxiv",
        pdf_url="https://export.arxiv.org/pdf/oversize-1.pdf",
    )

    async def fake_fetch(_url: str) -> DummyResponse:
        return DummyResponse(headers={"Content-Length": str(MAX_PDF_BYTES + 1)})

    monkeypatch.setattr(pipeline, "_fetch_response", fake_fetch)

    with pytest.raises(ValueError, match="Content-Length exceeds the 25 MB limit"):
        await pipeline.download_pdf(paper)


@pytest.mark.asyncio
async def test_download_rejects_corrupted_pdf_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path)
    paper = ScrapedPaperInput(
        paper_id="bad-header",
        title="Bad Header",
        source="arxiv",
        pdf_url="https://export.arxiv.org/pdf/bad-header.pdf",
    )

    async def fake_fetch(_url: str) -> DummyResponse:
        return DummyResponse(headers={"Content-Length": "5"}, chunks=[b"hello"])

    monkeypatch.setattr(pipeline, "_fetch_response", fake_fetch)

    with pytest.raises(ValueError, match="Corrupted PDF header"):
        await pipeline.download_pdf(paper)


@pytest.mark.asyncio
async def test_download_rejects_ssrf_redirect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path)
    paper = ScrapedPaperInput(
        paper_id="redirect-1",
        title="Redirect",
        source="arxiv",
        pdf_url="https://export.arxiv.org/pdf/redirect-1.pdf",
    )

    async def fake_fetch(_url: str) -> DummyResponse:
        return DummyResponse(
            headers={"Location": "https://evil.example/payload.pdf"},
            redirect=True,
        )

    monkeypatch.setattr(pipeline, "_fetch_response", fake_fetch)

    with pytest.raises(ValueError, match="Redirects are rejected"):
        await pipeline.download_pdf(paper)


@pytest.mark.asyncio
async def test_parse_document_quarantines_after_two_failures(tmp_path: Path) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path)
    paper_id = "quarantine-1"

    async def fail_parse(_file_path: str) -> ParsedDoc:
        raise RuntimeError("worker died")

    pipeline._run_worker_parse = fail_parse  # type: ignore[assignment]

    for _ in range(2):
        with pytest.raises(RuntimeError, match="parse"):
            await pipeline.parse_document(paper_id, str(tmp_path / "paper.pdf"))

    assert pipeline.is_quarantined(paper_id)
