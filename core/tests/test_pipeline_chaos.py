from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from arie_core.contracts import Figure, ParsedDoc
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
async def test_download_honours_a_lower_configured_size_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cap = 10 * 1024 * 1024
    pipeline = IngestionPipeline(storage_root=tmp_path, max_pdf_bytes=cap)
    paper = ScrapedPaperInput(
        paper_id="big-1",
        title="Big",
        source="arxiv",
        pdf_url="https://export.arxiv.org/pdf/big-1.pdf",
    )
    # No Content-Length: the streamed size is checked as it arrives.
    body = [b"%PDF-1.4\n" + b"0" * (cap - 9), b"0" * 1024]

    async def fake_fetch(_url: str) -> DummyResponse:
        return DummyResponse(chunks=body)

    monkeypatch.setattr(pipeline, "_fetch_response", fake_fetch)

    with pytest.raises(ValueError, match="Downloaded PDF exceeds the 10 MB limit"):
        await pipeline.download_pdf(paper)
    assert not await asyncio.to_thread(lambda: list(tmp_path.rglob("*.pdf")))
    # A cap above the hard limit is clamped to it.
    assert (
        IngestionPipeline(storage_root=tmp_path, max_pdf_bytes=MAX_PDF_BYTES * 2).max_pdf_bytes
        == MAX_PDF_BYTES
    )


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


@pytest.mark.asyncio
async def test_parse_document_renames_the_paper_and_keeps_every_parsed_field(
    tmp_path: Path,
) -> None:
    """Parsers name the paper after the stored file; only the id changes (figures were lost)."""
    pipeline = IngestionPipeline(storage_root=tmp_path)
    figure = Figure("2", 1, (10.0, 20.0, 110.0, 120.0), "Figure 2: Accuracy.", 0, 19)
    parsed = ParsedDoc(
        paper_id="b3dfc257979290a7",
        title="Cedar",
        source="",
        sections=["Figure 2: Accuracy."],
        metadata={"figure_count": "1"},
        figures=[figure],
    )

    async def parse(_file_path: str) -> ParsedDoc:
        return parsed

    pipeline._run_worker_parse = parse  # type: ignore[assignment]
    result = await pipeline.parse_document("1611.04798", str(tmp_path / "b3dfc257979290a7.pdf"))
    assert result.paper_id == "1611.04798"
    assert result.figures == [figure]
    assert (result.title, result.sections, result.metadata) == (
        parsed.title,
        parsed.sections,
        parsed.metadata,
    )
