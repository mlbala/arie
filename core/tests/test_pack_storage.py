"""Paper packs and guides in the runtime: stored per revision, built at INDEXED."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import msgspec
import pikepdf
import pytest
from arie_api.models import ANSWER_SLOTS, AnswerModel
from arie_api.runtime import ResearchRuntime, context
from arie_api.runtime import base as runtime_base
from arie_api.runtime import packs as packs_module
from arie_core.contracts import Figure, ParsedDoc
from arie_core.pipeline.pack import PaperPack, escape_markers, marker_numbers
from arie_core.plugin_api import ScrapedPaperInput
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import SyncConfig
from arie_worker.figure_ocr import OcrTexts
from arie_worker.models import LocalModel
from arie_worker.pdf_check import check_pdf
from conftest import ScriptedModel

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [
    "## Introduction\nLarge language models use agent tools to answer questions.",
    "Our method is simple and it improves accuracy by four points.",
    "## References\n[1] A. Author. A paper about models. 2020.",
]


def runtime_at(tmp_path: Path) -> ResearchRuntime:
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
    return ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))


def document(identifier: str, *, status: str = "INDEXED", count: int = 3) -> dict[str, Any]:
    chunks = [
        dict(
            paper_id=identifier,
            chunk_id=f"{identifier}:{i}",
            page_start=1,
            char_start=i * 100,
            char_end=i * 100 + len(TEXTS[i]),
            text=TEXTS[i],
        )
        for i in range(count)
    ]
    return dict(
        id=identifier,
        title="Agent tools for questions",
        abstract="Research with reproducible evidence.",
        source="arxiv",
        status=status,
        published_at="2025-06-01T00:00:00+00:00",
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        dedup_hash="hash-1",
        metadata={"primary_category": "cs.AI"},
        pages=[{"number": 1, "text": "\n\n".join(TEXTS)}],
        chunks=chunks,
    )


class BuildCounter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        original = packs_module.build_pack

        def counted(*args: Any, **kwargs: Any) -> PaperPack:
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(packs_module, "build_pack", counted)


async def stored_rows(runtime: ResearchRuntime) -> list[dict[str, Any]]:
    rows = await runtime.db.fetch(
        packs_module.context("test"),
        "SELECT paper_id, revision, text, tokens FROM arie_paper_packs ORDER BY paper_id",
    )
    return [dict(row) for row in rows]


@pytest.mark.asyncio
async def test_ensure_pack_stores_once_per_revision_and_rebuilds_when_the_paper_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    counter = BuildCounter(monkeypatch)
    try:
        paper = document("first")
        pack = await runtime.ensure_pack(paper)
        assert pack.chunk_numbers == (0, 1) and "[c2]" not in pack.text  # references excluded
        assert pack.revision == "hash-1:3:p2"
        rows = await stored_rows(runtime)
        assert [(r["paper_id"], r["revision"], r["tokens"]) for r in rows] == [
            ("first", "hash-1:3:p2", pack.token_estimate)
        ]
        assert rows[0]["text"] == pack.text

        again = await runtime.ensure_pack(paper)
        assert again == pack and counter.calls == 1

        changed = await runtime.ensure_pack(document("first", count=2))
        assert counter.calls == 2 and changed.revision == "hash-1:2:p2"
        assert [r["revision"] for r in await stored_rows(runtime)] == ["hash-1:2:p2"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_get_pack_builds_lazily_for_an_older_paper_then_reads_without_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    counter = BuildCounter(monkeypatch)
    try:
        await runtime.save_paper(document("old"))
        await runtime.save_paper(document("parsed", status="PARSED"))
        assert await stored_rows(runtime) == []  # saving alone builds nothing

        pack = await runtime.get_pack("old")
        assert pack is not None and pack.chunk_numbers == (0, 1) and counter.calls == 1
        assert marker_numbers(pack.text) == pack.chunk_numbers

        async def no_document(paper_id: str) -> None:
            raise AssertionError(f"loaded the document of {paper_id}")

        monkeypatch.setattr(runtime, "get_paper", no_document)
        assert await runtime.get_pack("old") == pack and counter.calls == 1

        assert await runtime.get_pack("parsed") is None
        assert await runtime.get_pack("missing") is None
        assert [r["paper_id"] for r in await stored_rows(runtime)] == ["old"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_database_from_before_packs_still_loads_and_builds_lazily(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        await runtime.save_paper(document("old"))
        await runtime.db.execute(packs_module.context("test"), "DROP TABLE arie_paper_packs")
    finally:
        runtime.close()

    reopened = runtime_at(tmp_path)
    await reopened.initialize()  # recreates the table instead of failing
    try:
        assert await stored_rows(reopened) == []
        pack = await reopened.get_pack("old")
        assert pack is not None and pack.chunk_numbers == (0, 1)
        assert [r["paper_id"] for r in await stored_rows(reopened)] == ["old"]
        await reopened.delete_paper("old")
        assert await stored_rows(reopened) == []
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_each_pack_holds_only_its_own_paper_and_delete_removes_it(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        second = document("second")
        second["chunks"][0]["text"] = "## Introduction\nA different paper uses its own method."
        await runtime.save_paper(document("first"))
        await runtime.save_paper(second)
        first_pack = await runtime.get_pack("first")
        second_pack = await runtime.get_pack("second")
        assert first_pack and second_pack
        assert "different paper" in second_pack.text and "different paper" not in first_pack.text
        assert "agent tools" in first_pack.text and "agent tools" not in second_pack.text

        await runtime.delete_paper("first", pdf_only=True)
        assert [r["paper_id"] for r in await stored_rows(runtime)] == ["first", "second"]
        await runtime.delete_paper("first")
        assert [r["paper_id"] for r in await stored_rows(runtime)] == ["second"]
    finally:
        runtime.close()


def resumable(tmp_path: Path) -> tuple[ScrapedPaperInput, dict[str, Any]]:
    """A paper parsed in an earlier run, so ingestion resumes at chunking with no download."""
    scraped = ScrapedPaperInput(
        paper_id="resumed",
        title="Agent tools for questions",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/resumed",
        abstract="Research with reproducible evidence.",
        published_at="2025-06-01T00:00:00Z",
        metadata={"primary_category": "cs.AI"},
    )
    pdf = tmp_path / "papers" / "resumed.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF- test file")
    parsed = document("resumed", status="PARSED")
    parsed.update(
        dedup_hash=scraped.dedup_hash, chunks=[], file_path=str(pdf), authors=[], outline=[]
    )
    return scraped, parsed


async def ingest(runtime: ResearchRuntime, scraped: ScrapedPaperInput) -> dict[str, Any]:
    job = await runtime.start_sync(settings=SyncConfig(mode="year", year=2025))
    await runtime._supervised_paper(job, scraped, ("arxiv.org",))  # pyright: ignore[reportPrivateUsage]
    return job


@pytest.mark.asyncio
async def test_a_paper_that_reaches_indexed_gets_its_pack_during_sync(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    scraped, parsed = resumable(tmp_path)
    try:
        await runtime.save_paper(parsed)
        job = await ingest(runtime, scraped)
        assert job["indexed"] == 1 and job["failed"] == 0
        saved = await runtime.get_paper("resumed")
        assert saved and saved["status"] == "INDEXED" and saved["chunks"]
        rows = await stored_rows(runtime)
        assert [r["paper_id"] for r in rows] == ["resumed"]
        assert rows[0]["revision"] == f"{scraped.dedup_hash}:{len(saved['chunks'])}:p2"
        assert "Title: Agent tools for questions" in rows[0]["text"] and "[c0]" in rows[0]["text"]
        assert "resumed" not in rows[0]["text"]  # the paper id never reaches the pack
        # Every marker leads back to the saved chunk and to the exact page text it quotes.
        pages = {page["number"]: page["text"] for page in saved["pages"]}
        by_id = {chunk["chunk_id"]: chunk for chunk in saved["chunks"]}
        numbers = marker_numbers(rows[0]["text"])
        assert numbers
        for number in numbers:
            chunk = by_id[f"resumed:{number}"]
            quoted = pages[chunk["page_start"]][chunk["char_start"] : chunk["char_end"]]
            assert quoted == chunk["text"]
            assert escape_markers(quoted.strip()) in rows[0]["text"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_sync_saves_the_hidden_text_count_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    scraped, parsed = resumable(tmp_path)
    parsed.update(pages=[])  # parse again, as a fresh download would

    async def parse_pdf(file_path: str) -> ParsedDoc:
        del file_path
        metadata = {"page_count": "1", "hidden_text_chars": "42"}
        return ParsedDoc("resumed", "", "", ["\n\n".join(TEXTS)], metadata)

    monkeypatch.setattr(runtime, "parse_pdf", parse_pdf)
    try:
        await runtime.save_paper(parsed)
        job = await ingest(runtime, scraped)
        assert job["indexed"] == 1 and job["failed"] == 0  # hidden text never fails a paper
        saved = await runtime.get_paper("resumed")
        assert saved and saved["status"] == "INDEXED" and saved["hidden_text_chars"] == 42
        warnings = [r.getMessage() for r in caplog.records if "hidden text" in r.getMessage()]
        assert warnings == ["Removed 42 hidden text characters from resumed"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_sync_stores_figures_and_tags_caption_and_mention_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    scraped, parsed = resumable(tmp_path)
    parsed.update(pages=[])
    page = "As Figure 1 shows, accuracy rises.\n\nFigure 1: Accuracy by size.\n\nNo figure here."
    start = page.index("Figure 1:")
    figure = Figure("1", 1, (1.0, 2.0, 3.0, 4.0), "Figure 1: Accuracy by size.", start, start + 27)

    async def parse_pdf(file_path: str) -> ParsedDoc:
        del file_path
        return ParsedDoc("resumed", "", "", [page], {"page_count": "1"}, [figure])

    monkeypatch.setattr(runtime, "parse_pdf", parse_pdf)
    try:
        await runtime.save_paper(parsed)
        job = await ingest(runtime, scraped)
        assert job["indexed"] == 1 and job["failed"] == 0
        saved = await runtime.get_paper("resumed")
        assert saved and msgspec.convert(saved["figures"], list[Figure]) == [figure]
        tags = [(c.get("figure"), c.get("mentions")) for c in saved["chunks"]]
        assert tags == [(None, ["1"]), ("1", None), (None, None)]
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("figure_ocr", [False, True])
async def test_figure_ocr_adds_one_ocr_chunk_only_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, figure_ocr: bool
) -> None:
    monkeypatch.setattr(runtime_base, "ocr_available", lambda: True)
    checked: list[Path] = []

    def verify_model(path: Path) -> Path:
        checked.append(path)
        return path

    monkeypatch.setattr(runtime_base, "verify_model", verify_model)
    base = runtime_at(tmp_path)
    checked.clear()  # the base runtime checked the model too (on by default, S26)
    ingestion = base.config.ingestion.model_copy(update={"figure_ocr": figure_ocr})
    runtime = ResearchRuntime(ROOT, base.config.model_copy(update={"ingestion": ingestion}))
    base.close()
    await runtime.initialize()
    scraped, parsed = resumable(tmp_path)
    parsed.update(pages=[])
    page = "As Figure 1 shows, accuracy rises.\n\nFigure 1: Accuracy by size.\n\nNo figure here."
    start = page.index("Figure 1:")
    figure = Figure("1", 1, (1.0, 2.0, 3.0, 4.0), "Figure 1: Accuracy by size.", start, start + 27)
    calls: list[tuple[list[tuple[int, Any]], str]] = []

    async def parse_pdf(file_path: str) -> ParsedDoc:
        del file_path
        return ParsedDoc("resumed", "", "", [page], {"page_count": "1"}, [figure])

    async def ocr_figures(file_path: str, figures: list[tuple[int, Any]], model: str) -> OcrTexts:
        del file_path
        calls.append((figures, model))
        return OcrTexts({0: "ACCURACY\n92.5"})

    monkeypatch.setattr(runtime, "parse_pdf", parse_pdf)
    monkeypatch.setattr(runtime.pool, "ocr_figures", ocr_figures)
    try:
        await runtime.save_paper(parsed)
        job = await ingest(runtime, scraped)
        assert job["indexed"] == 1 and job["failed"] == 0
        saved = await runtime.get_paper("resumed")
        assert saved is not None
        page_chunks = [
            {"paper_id": "resumed", "chunk_id": f"resumed:{i}", "page_start": 1, **span}
            for i, span in enumerate(
                [
                    {"char_start": 0, "char_end": 34, "text": page[:34], "mentions": ["1"]},
                    {"char_start": 36, "char_end": 63, "text": page[36:63], "figure": "1"},
                    {"char_start": 65, "char_end": len(page), "text": page[65:]},
                ]
            )
        ]
        if not figure_ocr:
            # Off: OCR never runs and the stored chunks and figures are those of S24.
            assert calls == [] and checked == [] and saved["chunks"] == page_chunks
            assert "ocr_text" not in saved["figures"][0]
            # OCR chunks stored while it was on are not evidence once it is off.
            stale = {**page_chunks[0], "chunk_id": "resumed:3", "ocr": True, "figure": "1"}
            rows = await runtime.evidence({**saved, "chunks": [*page_chunks, stale]})
            assert [row.chunk_id for row in rows] == ["resumed:0", "resumed:1", "resumed:2"]
            return
        model = ROOT / "models" / "ocr" / "en_PP-OCRv4_rec_infer.onnx"
        assert checked == [model]  # checked once, at startup
        assert calls == [([(1, (1.0, 2.0, 3.0, 4.0))], str(model))]
        assert saved["chunks"][:3] == page_chunks
        assert saved["chunks"][3:] == [
            {
                "paper_id": "resumed",
                "chunk_id": "resumed:3",
                "page_start": 1,
                "char_start": 0,
                "char_end": 13,
                "text": "ACCURACY\n92.5",
                "figure": "1",
                "figure_text": True,
                "ocr": True,
            }
        ]
        rows = await runtime.evidence(saved)
        assert [row.chunk_id for row in rows if row.ocr] == ["resumed:3"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_pack_failure_never_fails_an_indexed_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    scraped, parsed = resumable(tmp_path)

    async def broken(paper: dict[str, Any]) -> PaperPack:
        raise RuntimeError(f"cannot pack {paper['id']}")

    monkeypatch.setattr(runtime, "ensure_pack", broken)
    try:
        await runtime.save_paper(parsed)
        job = await ingest(runtime, scraped)
        assert job["indexed"] == 1 and job["failed"] == 0
        saved = await runtime.get_paper("resumed")
        assert saved and saved["status"] == "INDEXED"
        assert "Could not build the paper pack for resumed" in caplog.text
        assert await stored_rows(runtime) == []
    finally:
        runtime.close()


GUIDE_REPLY = json.dumps(
    {
        "tldr": "Large language models use agent tools to answer questions [c0].",
        "results": ["Accuracy rose by 12 points [c1]."],
        "section_summaries": {"Introduction": "The method improves accuracy by four points [c1]."},
        "suggested_questions": ["Which tools are used?"],
    }
)


def keyed(runtime: ResearchRuntime, monkeypatch: pytest.MonkeyPatch, *replies: str) -> Any:
    model = ScriptedModel(streams=[[reply] for reply in replies])

    def llm(provider: str | None = None) -> ScriptedModel:
        del provider
        return model

    monkeypatch.setattr(runtime, "_keyed_llm", llm)
    return model


@pytest.mark.asyncio
async def test_the_guide_builder_waits_while_an_answer_is_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    model = keyed(runtime, monkeypatch, GUIDE_REPLY)
    try:
        await runtime.save_paper(document("first"))
        runtime.queue_guide("first")
        with runtime.foreground.hold():
            builder = asyncio.create_task(runtime.guide_pass())
            await asyncio.sleep(0.05)
            assert not builder.done() and model.requests == []  # skipped while answering
        await asyncio.wait_for(builder, 5.0)
        assert len(model.requests) == 1 and runtime.guide_pending == {}

        guide = await runtime.get_guide("first")
        assert guide is not None and guide.revision == "hash-1:3:p2"
        assert guide.results == ()  # "12" is in no chunk, so the result was dropped
        assert guide.summaries() == {"Introduction": "The method improves accuracy by four points."}
        view = await runtime.guide_view(document("first"))
        assert view["status"] == "ready"
        assert [c["chunk_id"] for c in view["citations"]] == ["first:0", "first:1"]
        # A new revision of the paper makes the stored guide stale.
        await runtime.save_paper(document("first", count=2))
        assert await runtime.get_guide("first") is None
        assert await runtime.missing_guides() == ["first"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_failed_guide_stays_queued_and_is_retried_on_the_next_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    keyed(runtime, monkeypatch, "not json", GUIDE_REPLY)
    try:
        await runtime.save_paper(document("first"))
        runtime.queue_guide("first")
        await runtime.guide_pass()
        assert "first" in runtime.guide_pending
        assert runtime.guide_errors == {"first": "malformed_output"}
        view = await runtime.guide_view(document("first"))
        assert view == {"paper_id": "first", "status": "pending", "error": "malformed_output"}

        await runtime.guide_pass()
        assert runtime.guide_pending == {} and runtime.guide_errors == {}
        assert await runtime.get_guide("first") is not None
        await runtime.delete_paper("first")
        rows = await runtime.db.fetch(context("test"), "SELECT paper_id FROM arie_paper_guides")
        assert rows == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_without_a_key_the_builder_waits_and_sync_queues_indexed_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()

    def no_llm(provider: str | None = None) -> None:
        del provider

    monkeypatch.setattr(runtime, "_keyed_llm", no_llm)
    scraped, parsed = resumable(tmp_path)
    try:
        await runtime.save_paper(parsed)
        await ingest(runtime, scraped)
        assert runtime.guide_pending == {"resumed": None} and runtime.guide_wakeup.is_set()
        await runtime.guide_pass()
        assert runtime.guide_pending == {"resumed": None}
        assert runtime.guide_errors == {"*": "model_unavailable"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_pdf_parsing_waits_for_a_question_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    parsed: list[str] = []

    async def parse(file_path: str) -> Any:
        parsed.append(file_path)
        return "parsed"

    monkeypatch.setattr(runtime.pool, "parse", parse)
    try:
        assert await asyncio.wait_for(runtime.parse_pdf("idle.pdf"), 1.0) == "parsed"
        with runtime.foreground.hold():
            task = asyncio.create_task(runtime.parse_pdf("busy.pdf"))
            await asyncio.sleep(0.05)
            assert parsed == ["idle.pdf"] and not task.done()  # held back, not killed
        assert await asyncio.wait_for(task, 1.0) == "parsed"
        assert parsed == ["idle.pdf", "busy.pdf"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_pdf_with_javascript_is_quarantined_once_and_its_file_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    scraped, stored = resumable(tmp_path)
    stored.update(status="DOWNLOADING", pages=[])
    pdf = pikepdf.new()
    pdf.add_blank_page()
    pdf.Root.OpenAction = pikepdf.Dictionary(
        S=pikepdf.Name.JavaScript, JS=pikepdf.String("app.alert(1)")
    )
    pdf.save(stored["file_path"])
    calls: list[str] = []

    async def parse(file_path: str) -> Any:
        # The worker's order, in process: the check first, the parser only if it passes.
        calls.append(file_path)
        check_pdf(file_path, runtime.config.ingestion.pdf_check)
        raise AssertionError("the parser must not run")

    monkeypatch.setattr(runtime.pool, "parse", parse)
    try:
        await runtime.save_paper(stored)
        job = await ingest(runtime, scraped)
        assert job["failed"] == 1 and job["indexed"] == 0
        assert calls == [stored["file_path"]]  # no second parse attempt
        saved = await runtime.get_paper("resumed")
        assert saved and saved["status"] == "QUARANTINED"
        assert saved["status_reason"] == "quarantined: javascript"
        assert saved["pdf_deleted"] is True and "file_path" not in saved
        assert not await asyncio.to_thread(Path(stored["file_path"]).exists)
    finally:
        runtime.close()


class BusyWorker:
    """Stands in for a model pool whose worker still runs ``abandoned_calls`` abandoned calls."""

    def __init__(self, abandoned_calls: int) -> None:
        self.abandoned_calls = abandoned_calls

    async def call(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return [0.5, 0.4, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize("abandoned", [1, 0])
async def test_a_question_skips_reranking_only_while_the_reranker_worker_is_busy(
    tmp_path: Path, abandoned: int
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        reranker = AnswerModel(
            ANSWER_SLOTS[1], LocalModel(BusyWorker(abandoned), "reranker", "fake", {}), {}  # pyright: ignore[reportArgumentType]
        )
        reranker.ready = True
        runtime.models.models["reranker"] = reranker
        paper = document("first")
        answer = await runtime.answer_paper_question(
            context("chat:first", 10.0), paper, "How does the method improve accuracy?"
        )
        assert answer.answer.trace.rerank_skipped_busy is bool(abandoned)
        if abandoned:
            assert not answer.reading.reranked
        skipped = await runtime.answer_paper_question(
            context("chat:first", 10.0), paper, "How does the method improve accuracy?",
            rerank=False,
        )
        assert not skipped.answer.trace.rerank_skipped_busy  # asked not to rerank: not "busy"
    finally:
        runtime.close()
