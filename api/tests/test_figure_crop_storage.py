"""Figure crops are rendered in the PDF worker, stored next to the PDF and deleted with it (S24)."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
import pytest
from arie_api.routes import create_app
from arie_api.runtime import Document, ResearchRuntime
from arie_core.pipeline.figures import CAPTURE_KEY, CAPTURE_STEPS
from arie_core.registry.manager import ArieConfig
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
# pymupdf ships no complete types; the tests only build small PDFs with it.
pymupdf: Any = importlib.import_module("pymupdf")
BOX = [100.0, 120.0, 400.0, 300.0]


class _Client(Protocol):
    def get(self, url: str) -> httpx.Response: ...
    def post(self, url: str, *, json: object) -> httpx.Response: ...


def runtime_at(tmp_path: Path) -> ResearchRuntime:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        vector_store={"use": "none"},
        guard={"use": "none"},
    )
    return ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))


def stored_paper(tmp_path: Path) -> Document:
    pdf = tmp_path / "papers" / "2026" / "01" / "01" / "h1.pdf"
    pdf.parent.mkdir(parents=True)
    document = pymupdf.open()
    page = document.new_page()
    page.draw_rect(pymupdf.Rect(*BOX), color=(0, 0, 0), fill=(0.8, 0.2, 0.2))
    document.save(str(pdf))
    document.close()
    return dict(
        id="p1",
        title="Cedar",
        source="arxiv",
        abstract="",
        status="INDEXED",
        published_at="2026-01-01T00:00:00+00:00",
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        file_path=str(pdf),
        pages=[{"number": 1, "text": "Figure 2: Accuracy."}],
        chunks=[],
        figures=[
            {"number": "2", "page": 1, "box": BOX, "caption": "Figure 2: Accuracy."},
            # Off the page: skipped, so this record gets no crop.
            {"number": "3", "page": 1, "box": [900.0, 900.0, 950.0, 950.0], "caption": ""},
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pdf_only", [True, False])
async def test_crops_are_stored_in_the_paper_folder_and_removed_by_delete(
    tmp_path: Path, pdf_only: bool
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        paper = stored_paper(tmp_path)
        counts = await runtime.store_figure_crops(paper)
        assert counts == {"crops": 1, "bad_box": 1}
        figure, offpage = paper["figures"]
        assert figure["crop"] == "2026/01/01/h1.figures/0.png" and "crop" not in offpage
        crop = tmp_path / "papers" / figure["crop"]
        assert crop.read_bytes().startswith(b"\x89PNG")
        assert not list(crop.parent.glob("*.part"))
        await runtime.save_paper(paper)

        await runtime.delete_paper("p1", pdf_only=pdf_only)
        assert not crop.exists() and not crop.parent.exists()
        stored = await runtime.get_paper("p1")
        if pdf_only:
            assert stored is not None
            assert all("crop" not in f for f in stored["figures"])
        else:
            assert stored is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_rendering_again_replaces_the_old_crops(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        paper = stored_paper(tmp_path)
        await runtime.store_figure_crops(paper)
        folder = tmp_path / "papers" / "2026" / "01" / "01" / "h1.figures"
        paper["figures"] = paper["figures"][1:]  # a re-parse that lost Figure 2
        stale = folder / "0.png"
        paper["figures"][0]["crop"] = "2026/01/01/h1.figures/0.png"
        assert await runtime.store_figure_crops(paper) == {"crops": 0, "bad_box": 1}
        assert not stale.exists()
    finally:
        runtime.close()


def _events(text: str, kind: str) -> list[dict[str, Any]]:
    return [
        json.loads(block.split("data: ")[1])
        for block in text.split("\n\n")
        if block.startswith(f"event: {kind}\n")
    ]


def test_a_strict_fallback_on_a_figure_caption_names_the_figure_and_links_its_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = runtime_at(tmp_path)
    initialize, instance = runtime.initialize, runtime.instance
    caption = "Figure 2: Accuracy by model size on Cedar."

    async def seed() -> None:
        await initialize()
        paper = stored_paper(tmp_path)
        paper["pages"] = [{"number": 1, "text": caption}]
        paper["chunks"] = [
            dict(
                paper_id="p1",
                chunk_id="p1:0",
                text=caption,
                page_start=1,
                char_start=0,
                char_end=len(caption),
                figure="2",
            )
        ]
        await runtime.store_figure_crops(paper)
        # Captured already (S26), so the background figure catch-up leaves it alone.
        paper[CAPTURE_KEY] = dict(CAPTURE_STEPS)
        await runtime.save_paper(paper)

    async def idle() -> None:
        await asyncio.Event().wait()

    def without_model(slot: str, name: str | None = None) -> object:
        return object() if slot == "llm" else instance(slot, name)

    monkeypatch.setattr(runtime, "initialize", seed)
    monkeypatch.setattr(runtime, "instance", without_model)
    monkeypatch.setattr(runtime, "run", idle)
    monkeypatch.setattr(runtime, "run_imports", idle)
    with TestClient(create_app(runtime)) as connected:
        client = cast(_Client, connected)
        response = client.post(
            "/v1/papers/p1/questions",
            json={"question": "How does accuracy change with model size?", "mode": "strict"},
        )
        [hint] = _events(response.text, "figure_hint")
        assert hint == {
            "paper_id": "p1",
            "number": "2",
            "page": 1,
            "message": "The closest passage refers to Figure 2 (p. 1).",
            "crop_url": "/v1/papers/p1/figures/2.png",
        }
        # The caption alone does not answer: the abstention keeps its text, the hint adds a fact.
        [abstention] = _events(response.text, "abstention")
        assert abstention["message"].startswith("I could not find verified evidence")
        [answer] = _events(response.text, "answer")
        assert answer["evaluation"]["trace"]["figure_hint"] == "2"
        assert client.get(hint["crop_url"]).content.startswith(b"\x89PNG")
        [turn] = client.get("/v1/papers/p1/chat").json()["turns"]
        assert "figure_hint" in [event["event"] for event in turn["events"]]
