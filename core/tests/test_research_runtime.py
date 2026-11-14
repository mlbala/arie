from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from arie_api.routes import create_app
from arie_api.runtime import ResearchRuntime, context, months_ago
from arie_core.contracts import LLMResponse, TokenUsage
from arie_core.harness.reader import EvidenceChunk, retrieve, select_excerpts
from arie_core.pipeline import IngestionPipeline
from arie_core.plugin_api import Ctx, LLMRequest, ScrapedPaperInput, SourcePage
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import SyncConfig
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def test_calendar_window_clamps_month_ends() -> None:
    assert months_ago(datetime(2026, 3, 31, tzinfo=UTC), 6) == datetime(2025, 9, 30, tzinfo=UTC)
    assert months_ago(datetime(2024, 8, 31, tzinfo=UTC), 6) == datetime(2024, 2, 29, tzinfo=UTC)


@pytest.mark.asyncio
async def test_reader_rejects_other_papers_and_fabricated_model_quotes() -> None:
    row = EvidenceChunk("selected", "c1", "The method uses isolated worker processes.", 2, 10, 52)
    foreign = EvidenceChunk("other", "c2", "The method uses secret vendor data.", 1, 0, 39)
    rows = retrieve("selected", "method", [foreign, row])
    assert rows == [row]

    class FabricatingLLM:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx
            assert request.tools == [] and request.max_output_tokens <= 700
            assert "secret vendor" not in request.prompt
            return LLMResponse('{"quotes":["Made-up unsupported evidence."]}', TokenUsage(1, 1))

        async def stream(self, ctx: Ctx, request: LLMRequest):
            del ctx, request
            yield

    from arie_core.plugin_api import LLM

    assert await select_excerpts(context("test"), "method", rows, cast(LLM, FabricatingLLM())) == []
    excerpts = await select_excerpts(context("test"), "method", rows)
    assert excerpts and excerpts[0].text == excerpts[0].quote
    assert excerpts[0].char_start == 10


@pytest.mark.asyncio
async def test_pdf_stream_aborts_early_and_closes_on_bad_header(tmp_path: Path) -> None:
    read = 0

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal read
            for _ in range(100):
                read += 1
                yield b"x" * 65536

    response = httpx.Response(
        200, stream=Stream(), request=httpx.Request("GET", "https://arxiv.org")
    )
    pipeline = IngestionPipeline(storage_root=tmp_path)

    async def fetch(url: str) -> httpx.Response:
        del url
        return response

    pipeline._fetch_response = fetch  # pyright: ignore[reportPrivateUsage]
    paper = ScrapedPaperInput(
        paper_id="bad", title="AI", source="arxiv", pdf_url="https://arxiv.org/pdf/bad"
    )
    with pytest.raises(ValueError, match="Corrupted PDF header"):
        await pipeline.download_pdf(paper)
    assert read == 1 and response.is_closed
    assert await asyncio.to_thread(
        lambda: not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob("*.pdf"))
    )


def test_sync_extracts_real_pdf_persists_and_serves_cited_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "input.pdf"
    # Generate a compressed PDF in a separate process; raw-byte decoding cannot pass this test.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pymupdf, sys
with pymupdf.open() as doc:
    page = doc.new_page()
    page.insert_text((72,72), 'Isolated AI Research', fontsize=18)
    page.insert_text((72,112), 'The method uses isolated worker processes to extract PDF evidence.')
    page.insert_text((72,142), 'Results show that source quotes preserve reliable page citations.')
    page = doc.new_page()
    page.insert_text((72,72), 'Limitations', fontsize=18)
    page.insert_text((72,112), 'The method requires readable text; scanned images are unsupported.')
    doc.save(sys.argv[1], deflate=True)
""",
            str(pdf),
        ],
        check=True,
    )
    content = pdf.read_bytes()
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
        reranker={"use": "none"},
        sources=["arxiv"],
    )
    # The default 6-month window and no retention, whatever arie.yaml sets.
    config = config.model_copy(update={"plugins": plugins, "sync": SyncConfig()})
    runtime = ResearchRuntime(ROOT, config)
    published = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    pages_seen: list[str | None] = []
    downloads: list[str] = []

    def paper(identifier: str) -> ScrapedPaperInput:
        return ScrapedPaperInput(
            paper_id=identifier,
            title=f"AI worker method {identifier}",
            source="arxiv",
            pdf_url=f"https://arxiv.org/pdf/{identifier}",
            published_at=published,
            metadata={"primary_category": "cs.AI"},
        )

    class Source:
        allowed_hosts = ("arxiv.org",)

        async def fetch_page(
            self, ctx: Ctx, since: str, until: str, cursor: str | None = None, limit: int = 20
        ) -> SourcePage:
            del ctx, until, limit
            assert datetime.fromisoformat(since) <= datetime.now(UTC) - timedelta(days=180)
            pages_seen.append(cursor)
            return SourcePage(
                [paper("broken")] if cursor is None else [paper("good")],
                "next" if cursor is None else None,
            )

    class UnavailableLLM:
        api_key = "test-credential"

        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx, request
            response = httpx.Response(401, request=httpx.Request("POST", "https://provider.test"))
            response.raise_for_status()
            raise AssertionError("Unreachable")

    original = runtime.manager.instantiate_plugin

    def instantiate(slot: str, plugin_name: str | None = None, **kwargs: object) -> object:
        if slot == "source":
            return Source()
        if slot == "llm":
            return UnavailableLLM()
        return original(slot, plugin_name, **kwargs)

    monkeypatch.setattr(runtime.manager, "instantiate_plugin", instantiate)

    async def fetch(self: IngestionPipeline, url: str) -> httpx.Response:
        del self
        downloads.append(url)
        return httpx.Response(
            200,
            content=b"not a pdf" if "broken" in url else content,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(IngestionPipeline, "_fetch_response", fetch)
    with cast(httpx.Client, TestClient(create_app(runtime))) as client:
        assert client.get("/v1/papers").json()["items"] == []
        start = client.post("/v1/sync", json={})
        assert start.status_code == 202
        job_id = start.json()["job_id"]
        assert client.post("/v1/sync", json={}).json()["job_id"] == job_id
        job: dict[str, Any] = {}
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline:
            job = client.get(f"/v1/sync/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.1)
        assert job["indexed"] == 1, job
        assert job["failed"] == 1 and job["status"] == "completed_with_errors"
        assert pages_seen == [None, "next"]
        library = client.get("/v1/papers").json()["items"]
        assert [item["id"] for item in library] == ["good"]
        assert library[0]["page_count"] == 2
        assert client.get("/v1/papers/good/pdf").content.startswith(b"%PDF-")
        answer = client.post(
            "/v1/papers/good/questions", json={"question": "How does the method use workers?"}
        )
        events = [
            json.loads(block.split("data: ")[1])
            for block in answer.text.split("\n\n")
            if "event: claim" in block
        ]
        assert events and "source_excerpts" in answer.text, answer.text
        for event in events:
            claim = event["claim"]
            citation = claim["citation"]
            text = client.get(f"/v1/papers/good/pages/{citation['page_start']}").json()["text"]
            assert text[citation["char_start"] : citation["char_end"]] == claim["quote"]
            assert claim["text"] == claim["quote"] and event["paper_id"] == "good"
        assert (
            "event: abstention"
            in client.post("/v1/papers/good/questions", json={"question": "quantum bananas?"}).text
        )
        assert client.post("/v1/sync", json={"source": "disabled"}).status_code == 400
        assert (
            client.post("/v1/papers/good/questions", json={"question": "x" * 2001}).status_code
            == 422
        )
        assert client.get("/v1/papers/good/pages/99").status_code == 404
    # New service instance must read the completed document from disk, without demo seeds.
    restored = ResearchRuntime(ROOT, config)
    with cast(httpx.Client, TestClient(create_app(restored))) as client:
        assert client.get("/v1/papers").json()["items"][0]["id"] == "good"
        assert client.get("/v1/sync").json()["job"]["job_id"] == job_id


def test_chunk_spans_preserve_long_paragraphs_tables_and_offsets() -> None:
    from arie_core.pipeline.chunker import chunk_spans

    paragraph = "Detailed evidence from the paper. " * 250
    table = "| Method | Score |\n| --- | --- |\n" + "| Isolated | 0.95 |\n" * 200
    text = f"# Methods\n\n{paragraph}\n\n{table}\n\n$$E = mc^2$$\n"
    spans = chunk_spans(text)
    recovered = "".join(text[start:end] for start, end in spans)
    assert "".join(recovered.split()) == "".join(text.split())
    assert any(text[start:end] == table.rstrip() for start, end in spans)
    assert any("$$E = mc^2$$" in text[start:end] for start, end in spans)


@pytest.mark.asyncio
async def test_interrupted_sync_resumes_saved_window_and_cursor(tmp_path: Path) -> None:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "resume.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
        reranker={"use": "none"},
    )
    config = config.model_copy(update={"plugins": plugins})
    first = ResearchRuntime(ROOT, config)
    await first.initialize()
    job = await first.start_sync()
    since = job["since"]
    await first.update_job(job, status="running", cursor="saved-page", pause_requested=True)
    first.close()
    restored = ResearchRuntime(ROOT, config)
    try:
        await restored.initialize()
        assert restored.jobs[job["job_id"]]["status"] == "interrupted"
        resumed = await restored.start_sync()
        assert resumed["job_id"] == job["job_id"]
        assert resumed["cursor"] == "saved-page" and resumed["since"] == since
        assert resumed["status"] == "queued" and resumed["pause_requested"] is False
        assert restored.queue.qsize() == 1
    finally:
        restored.close()
