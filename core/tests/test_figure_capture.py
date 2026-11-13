"""Capture by default (S26): OCR on, a missing model never stops ARIE, older papers catch up."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import msgspec
import pytest
from arie_api.monitor import StorageMonitor
from arie_api.runtime import ResearchRuntime
from arie_api.runtime import base as runtime_base
from arie_core.contracts import Figure, ParsedDoc
from arie_core.pipeline.figures import CAPTURE_KEY, CAPTURE_STEPS, capture_due
from arie_core.registry.manager import ArieConfig
from arie_worker.figure_crops import Crops
from arie_worker.figure_ocr import OcrTexts

ROOT = Path(__file__).resolve().parents[2]
PAGE = "As Figure 1 shows, accuracy rises.\n\nFigure 1: Accuracy by size.\n\nNo figure here."
CAPTION = PAGE.index("Figure 1:")
FIGURE = Figure("1", 1, (1.0, 2.0, 3.0, 4.0), "Figure 1: Accuracy by size.", CAPTION, CAPTION + 27)
CURRENT = dict(CAPTURE_STEPS)


def runtime_at(tmp_path: Path, **ingestion: object) -> ResearchRuntime:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        vector_store={"use": "none"},
        guard={"use": "none"},
        embedder={"use": "none"},
        entailment={"use": "none"},
        reranker={"use": "none"},
    )
    update: dict[str, object] = {"plugins": plugins}
    if ingestion:
        update["ingestion"] = config.ingestion.model_copy(update=ingestion)
    return ResearchRuntime(ROOT, config.model_copy(update=update))


def paper(tmp_path: Path, identifier: str, **extra: object) -> dict[str, Any]:
    pdf = tmp_path / "papers" / f"{identifier}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF- test file")
    chunks = [
        dict(
            paper_id=identifier,
            chunk_id=f"{identifier}:0",
            page_start=1,
            char_start=0,
            char_end=len(PAGE),
            text=PAGE,
        )
    ]
    return dict(
        id=identifier,
        title="Agent tools",
        abstract="",
        source="arxiv",
        status="INDEXED",
        published_at="2025-06-01T00:00:00+00:00",
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        dedup_hash=f"hash-{identifier}",
        metadata={},
        pages=[{"number": 1, "text": PAGE}],
        chunks=chunks,
        file_path=str(pdf),
        **extra,
    )


class FakeWorker:
    """The PDF pool's figure calls, recorded; no worker process starts."""

    def __init__(self, runtime: ResearchRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, str]] = []
        monkeypatch.setattr(runtime.pool, "parse", self.parse)
        monkeypatch.setattr(runtime.pool, "render_crops", self.render_crops)
        monkeypatch.setattr(runtime.pool, "ocr_figures", self.ocr_figures)

    async def parse(self, file_path: str) -> ParsedDoc:
        self.calls.append(("parse", Path(file_path).stem))
        return ParsedDoc("x", "", "", [PAGE], {"page_count": "1"}, [FIGURE])

    async def render_crops(self, file_path: str, figures: list[Any], *_: object) -> Crops:
        self.calls.append(("crops", Path(file_path).stem))
        return Crops({i: b"\x89PNG" for i in range(len(figures))})

    async def ocr_figures(self, file_path: str, figures: list[Any], model: str) -> OcrTexts:
        del model
        self.calls.append(("ocr", Path(file_path).stem))
        return OcrTexts({i: "ACCURACY 92.5" for i in range(len(figures))})


@pytest.fixture
def ocr_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_base, "ocr_available", lambda: True)
    def verify_model(path: Path) -> Path:
        return path

    monkeypatch.setattr(runtime_base, "verify_model", verify_model)


async def catch_up(runtime: ResearchRuntime) -> None:
    """Run the background catch-up until its first pass ends."""
    runtime.capture_idle.clear()
    task = asyncio.create_task(runtime.run_figure_capture())
    try:
        async with asyncio.timeout(5):
            await runtime.capture_idle.wait()
    finally:
        task.cancel()


def test_capture_is_on_by_default_and_due_steps_follow_the_record() -> None:
    assert ArieConfig().ingestion.figure_ocr is True
    figures: list[dict[str, object]] = [{"number": "1", "crop": "a/0.png"}]
    assert capture_due({}, ocr=True) == ["figures", "crops", "ocr"]  # indexed before S22
    assert capture_due({"figures": []}, ocr=True) == ["ocr"]  # before S26, no crops needed
    assert capture_due({"figures": [{"number": "1"}]}, ocr=True) == ["crops", "ocr"]
    assert capture_due({"figures": figures}, ocr=False) == []
    assert capture_due({CAPTURE_KEY: CURRENT}, ocr=True) == []
    assert capture_due({CAPTURE_KEY: {"figures": 1, "crops": 1}}, ocr=True) == ["ocr"]
    assert capture_due({CAPTURE_KEY: {"figures": 0, "crops": 1, "ocr": 1}}, ocr=False) == [
        "figures",
        "crops",
    ]


def test_a_missing_ocr_model_skips_ocr_and_never_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    model = tmp_path / "rec.onnx"
    monkeypatch.setattr(runtime_base, "ocr_available", lambda: False)
    runtime = runtime_at(tmp_path, figure_ocr_model=str(model))
    assert runtime.ocr_problem == "OCR engine missing, run uv sync"
    runtime.close()
    monkeypatch.setattr(runtime_base, "ocr_available", lambda: True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=runtime_base.logger.name):
        runtime = runtime_at(tmp_path, figure_ocr_model=str(model))
        runtime.check_ocr()  # the same problem again: no second warning
    try:
        assert runtime.ocr_problem == "model missing, run make ocr-model"
        assert not runtime.ocr_ready()
        warnings = [r.getMessage() for r in caplog.records if "Figure OCR" in r.getMessage()]
        assert len(warnings) == 1 and "make ocr-model" in warnings[0]
        status = runtime.figure_capture_status()
        assert status["ocr"] == "unavailable"
        model.write_bytes(b"a different model")
        assert runtime.check_ocr() == "model checksum mismatch, run make ocr-model"
    finally:
        runtime.close()
    off = runtime_at(tmp_path, figure_ocr=False, figure_ocr_model=str(model))
    assert off.ocr_problem is None and not off.ocr_ready()
    off.close()


@pytest.mark.asyncio
async def test_without_the_model_figures_and_crops_are_still_captured_and_health_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_base, "ocr_available", lambda: True)
    runtime = runtime_at(tmp_path, figure_ocr_model=str(tmp_path / "missing.onnx"))
    worker = FakeWorker(runtime, monkeypatch)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper(tmp_path, "old"))
        await runtime.save_paper(paper(tmp_path, "gone", pdf_deleted=True))
        await catch_up(runtime)
        assert worker.calls == [("parse", "old"), ("crops", "old")]
        saved = await runtime.get_paper("old")
        assert saved is not None and saved[CAPTURE_KEY] == {"figures": 1, "crops": 1}
        assert saved["figures"][0]["crop"] == "old.figures/0.png"
        assert not any(chunk.get("ocr") for chunk in saved["chunks"])
        health = await StorageMonitor(runtime).snapshot()
        figures = health["figures"]
        assert figures["status"] == "degraded" and figures["papers_no_pdf"] == 1
        assert figures["papers_captured"] == 1 and figures["papers_total"] == 2
        assert "Figure OCR: model missing, run make ocr-model." in figures["detail"]
        assert "1 have no stored PDF" in figures["detail"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_catch_up_runs_only_the_steps_a_paper_is_behind_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ocr_present: None
) -> None:
    del ocr_present
    runtime = runtime_at(tmp_path)
    worker = FakeWorker(runtime, monkeypatch)
    await runtime.initialize()
    stored = msgspec.to_builtins([FIGURE])
    try:
        await runtime.save_paper(paper(tmp_path, "bare"))
        await runtime.save_paper(paper(tmp_path, "nocrop", figures=stored))
        await runtime.save_paper(paper(tmp_path, "done", figures=stored, **{CAPTURE_KEY: CURRENT}))
        await catch_up(runtime)
        assert sorted(worker.calls) == [
            ("crops", "bare"),
            ("crops", "nocrop"),
            ("ocr", "bare"),
            ("ocr", "nocrop"),
            ("parse", "bare"),
        ]
        for identifier in ("bare", "nocrop"):
            saved = await runtime.get_paper(identifier)
            assert saved is not None and saved[CAPTURE_KEY] == CURRENT
            assert [c.get("ocr") for c in saved["chunks"]] == [None, True]
            assert saved["chunks"][1]["text"] == "ACCURACY 92.5"
        # Only the re-parsed paper's page chunk is tagged; it holds the caption.
        bare = await runtime.get_paper("bare")
        assert bare is not None and bare["chunks"][0]["figure"] == "1"
        status = runtime.figure_capture_status()
        assert status["status"] == "healthy" and status["papers_captured"] == 3
        # A second pass finds nothing to do.
        worker.calls.clear()
        await catch_up(runtime)
        assert worker.calls == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_catch_up_waits_while_a_question_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ocr_present: None
) -> None:
    del ocr_present
    runtime = runtime_at(tmp_path)
    worker = FakeWorker(runtime, monkeypatch)
    await runtime.initialize()
    await runtime.save_paper(paper(tmp_path, "bare"))
    task: asyncio.Task[None] | None = None
    try:
        with runtime.foreground.hold():
            task = asyncio.create_task(runtime.run_figure_capture())
            await asyncio.sleep(0.2)
            assert worker.calls == [] and runtime.capture_pending == 1
        async with asyncio.timeout(5):
            await runtime.capture_idle.wait()
        assert runtime.capture_pending == 0
        assert [step for step, _ in worker.calls] == ["parse", "crops", "ocr"]
    finally:
        if task is not None:
            task.cancel()
        runtime.close()


@pytest.mark.asyncio
async def test_with_ocr_off_everything_but_ocr_text_is_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ocr_present: None
) -> None:
    del ocr_present
    runtime = runtime_at(tmp_path, figure_ocr=False)
    worker = FakeWorker(runtime, monkeypatch)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper(tmp_path, "bare"))
        await catch_up(runtime)
        assert worker.calls == [("parse", "bare"), ("crops", "bare")]
        saved = await runtime.get_paper("bare")
        assert saved is not None and saved[CAPTURE_KEY] == {"figures": 1, "crops": 1}
        assert len(saved["chunks"]) == 1 and "ocr_text" not in saved["figures"][0]
        status = runtime.figure_capture_status()
        assert status["status"] == "healthy" and status["ocr"] == "off"
        assert "Figure OCR is off." in status["detail"]
    finally:
        runtime.close()
