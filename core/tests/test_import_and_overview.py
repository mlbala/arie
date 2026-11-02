from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import httpx
import pytest
from arie_api.routes import create_app
from arie_api.runtime import ResearchRuntime, context
from arie_core.contracts import LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.reader import EvidenceChunk, is_overview, retrieve, select_excerpts
from arie_core.pipeline import IngestionPipeline
from arie_core.plugin_api import LLM, Ctx, wait_for_source_host
from arie_core.registry.manager import ArieConfig
from arie_source_arxiv import ArxivSourcePlugin
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "question",
    [
        "What is the paper about?",
        "what is this paper about",
        "Explain this paper",
        "Give me a summary",
        "Summarize the paper",
        "What is its main contribution?",
    ],
)
@pytest.mark.asyncio
async def test_overview_uses_pdf_narrative_even_with_no_keyword_or_provider(question: str) -> None:
    abstract = (
        "## Abstract\n\nWe propose a new research model based solely on attention. "
        "The model is more parallelizable and requires less training time."
    )
    rows = [
        EvidenceChunk("selected", "abstract", abstract, 1, 90, 90 + len(abstract)),
        EvidenceChunk("other", "foreign", "We propose a model using secret data.", 1, 0, 43),
        EvidenceChunk(
            "selected", "refs", "## References\n\nOur model is a fictional model.", 9, 0, 50
        ),
    ]
    assert is_overview(question)
    selected = retrieve("selected", question, rows)
    assert selected and all(row.paper_id == "selected" for row in selected)
    assert all(row.chunk_id != "refs" for row in selected)
    answer = await select_excerpts(context("overview"), question, selected)
    assert answer
    assert "The paper proposes" in answer[0].text
    assert any("attention" in item.text for item in answer)
    for item in answer:
        assert abstract[item.char_start - 90 : item.char_end - 90] == item.quote
        assert item.page_start == 1
    assert sum(len(item.text.encode("utf-8")) for item in answer) <= 700
    assert retrieve("selected", "quantum bananas", rows) == []


@pytest.mark.asyncio
async def test_model_quote_normalization_preserves_authoritative_offsets() -> None:
    text = "We propose an efﬁcient\nmodel that uses verified source passages."
    row = EvidenceChunk("paper", "c", text, 3, 40, 40 + len(text))

    class Model:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx
            assert request.tools == [] and request.max_output_tokens == 700
            return LLMResponse(
                json.dumps(
                    {
                        "quotes": [
                            "We propose an efficient model that uses verified source passages.",
                            "Our model achieved 100 percent accuracy on every dataset.",
                        ]
                    }
                ),
                TokenUsage(1, 1),
            )

    answer = await select_excerpts(context("normalization"), "method", [row], cast(LLM, Model()))
    assert len(answer) == 1
    assert answer[0].quote == text
    assert answer[0].char_start == 40 and answer[0].char_end == 40 + len(text)
    assert "100 percent" not in answer[0].text


@pytest.mark.parametrize(
    "url, canonical",
    [
        ("http://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1706.03762"),
        ("https://www.arxiv.org/pdf/1706.03762", "https://arxiv.org/pdf/1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v2#page=3", "https://arxiv.org/pdf/1706.03762v2"),
        ("https://arxiv.org/pdf/cs/9901001.pdf", "https://arxiv.org/pdf/cs/9901001"),
    ],
)
def test_arxiv_pdf_url_accepts_old_and_versioned_identifiers(url: str, canonical: str) -> None:
    assert ArxivSourcePlugin.canonical_pdf_url(url) == canonical


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/pdf/1706.03762",
        "https://arxiv.org.evil.example/pdf/1706.03762",
        "https://arxiv.org@127.0.0.1/pdf/1706.03762",
        "https://arxiv.org:444/pdf/1706.03762",
        "https://arxiv.org/abs/1706.03762",
        "https://arxiv.org/pdf/../1706.03762",
        "https://arxiv.org/pdf/%2e%2e/1706.03762",
        "https://arxiv.org/pdf/1706.03762?redirect=https://evil.example",
        "file:///etc/passwd",
        "https://arxiv.org/pdf/1713.12345",
        "https://arxiv.org/pdf/no-id",
    ],
)
def test_arxiv_pdf_url_rejects_non_pdf_and_ssrf_inputs(url: str) -> None:
    with pytest.raises(ValueError):
        ArxivSourcePlugin.canonical_pdf_url(url)


def test_old_url_import_runs_during_batch_and_answers_overview_with_page_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "source.pdf"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pymupdf, sys
with pymupdf.open() as doc:
    page = doc.new_page()
    page.insert_text((72,72), 'Research Reader', fontsize=18)
    page.insert_text((72,110), 'Abstract', fontsize=15)
    page.insert_text((72,145), 'We propose the Context Reader model for verified source answers.')
    page.insert_text((72,170), 'The model uses attention to retrieve relevant paper evidence.')
    page = doc.new_page()
    page.insert_text((72,72), 'Conclusion', fontsize=18)
    page.insert_text((72,110), 'The model preserves page citations for source inspection.')
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
        database={"use": "sqlite", "path": str(tmp_path / "import.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))
    metadata_requests: list[dict[str, str | int]] = []
    source = ArxivSourcePlugin()

    async def query(ctx: Ctx, params: dict[str, str | int]) -> ET.Element:
        del ctx
        metadata_requests.append(params)
        return ET.fromstring("""<feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom"><entry>
          <id>http://arxiv.org/abs/1706.03762v1</id><title>Research Reader</title>
          <summary>A paper about attention and evidence.</summary>
          <published>2017-06-12T00:00:00Z</published>
          <arxiv:primary_category term="cs.CL" /></entry></feed>""")

    monkeypatch.setattr(source, "_query", query)
    runtime.url_source = source
    downloads: list[str] = []

    async def fetch(self: IngestionPipeline, url: str) -> httpx.Response:
        del self
        downloads.append(url)
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(IngestionPipeline, "_fetch_response", fetch)
    original = runtime.manager.instantiate_plugin

    class SlowBatch:
        async def fetch_page(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            await asyncio.sleep(120)

    def instantiate(slot: str, plugin_name: str | None = None, **kwargs: object) -> object:
        if slot == "source":
            return SlowBatch()
        if slot == "llm":
            return object()
        return original(slot, plugin_name, **kwargs)

    monkeypatch.setattr(runtime.manager, "instantiate_plugin", instantiate)
    with cast(httpx.Client, TestClient(create_app(runtime))) as client:
        for url in ["https://evil.example/pdf/1706.03762", "https://arxiv.org/abs/1706.03762"]:
            assert client.post("/v1/imports", json={"url": url}).status_code == 400
        assert (
            client.post(
                "/v1/imports",
                json={"url": "https://arxiv.org/pdf/1706.03762", "host": "evil.example"},
            ).status_code
            == 422
        )
        assert metadata_requests == [] and downloads == []
        batch = client.post("/v1/sync", json={}).json()
        started = client.post("/v1/imports", json={"url": "https://arxiv.org/pdf/1706.03762.pdf"})
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        same = client.post("/v1/imports", json={"url": "https://arxiv.org/pdf/1706.03762"}).json()
        assert same["job_id"] == job_id
        job = started.json()
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and job["status"] in {"queued", "running"}:
            time.sleep(0.1)
            job = client.get(f"/v1/imports/{job_id}").json()
        assert job["status"] == "completed" and job["indexed"] == 1, job
        assert job["paper_id"] == "1706.03762"
        assert metadata_requests == [{"id_list": "1706.03762"}]
        assert len(downloads) == 1
        assert client.get("/v1/sync").json()["job"]["job_id"] == batch["job_id"]
        paper = client.get("/v1/papers/1706.03762").json()
        assert paper["published_at"].startswith("2017-") and paper["page_count"] == 2
        answer = client.post(
            "/v1/papers/1706.03762/questions", json={"question": "what is the paper about"}
        )
        assert "event: abstention" not in answer.text, answer.text
        assert '"heading": "Paper summary"' in answer.text
        assert "Context Reader" in answer.text
        claims = [
            json.loads(block.split("data: ")[1])["claim"]
            for block in answer.text.split("\n\n")
            if "event: claim" in block
        ]
        assert claims
        for claim in claims:
            citation = claim["citation"]
            page = client.get(f"/v1/papers/1706.03762/pages/{citation['page_start']}").json()
            assert page["text"][citation["char_start"] : citation["char_end"]] == claim["quote"]
        assert client.get("/v1/imports/no-such-job").status_code == 404


@pytest.mark.asyncio
async def test_source_spacing_is_shared_and_does_not_wait_past_deadline() -> None:
    ctx = context("host-deadline", 0.1)
    await wait_for_source_host(ctx, "export.arxiv.org")
    with pytest.raises(TimeoutError, match="deadline"):
        await wait_for_source_host(ctx, "export.arxiv.org")
    await wait_for_source_host(ctx, "independent-source.test")
