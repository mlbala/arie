"""Parser plugins: page-level text, explicit OCR failures, and worker-only execution."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from arie_core.plugin_api import Figure, ParsedDoc
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]


def parser(name: str, **kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin("parser", name, **kwargs)


def write_pdf(path: Path, pages: list[str]) -> str:
    pymupdf = importlib.import_module("pymupdf")
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=12)
    document.save(str(path))
    return str(path)


@pytest.mark.asyncio
async def test_pymupdf4llm_returns_one_section_per_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = write_pdf(tmp_path / "2510.01234.pdf", ["Gated memory retrieval.", "Results on Cedar."])
    with pytest.raises(RuntimeError, match="inside arie-worker"):
        await parser("pymupdf4llm").parse(None, pdf)
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    parsed: ParsedDoc = await parser("pymupdf4llm").parse(None, pdf)
    assert parsed.paper_id == "2510.01234" and parsed.metadata["page_count"] == "2"
    assert "Gated memory retrieval." in parsed.sections[0]
    assert "Results on Cedar." in parsed.sections[1]


@pytest.mark.asyncio
async def test_pymupdf4llm_returns_captioned_figures_with_page_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pymupdf = importlib.import_module("pymupdf")
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "A first page of plain text.", fontsize=11)
    page = document.new_page()
    page.insert_text((72, 72), "As Figure 1 shows, accuracy rises with size.", fontsize=11)
    page.draw_rect(pymupdf.Rect(100, 120, 400, 300), color=(0, 0, 0), fill=(0.8, 0.2, 0.2))
    page.draw_line((120, 140), (380, 280))
    page.insert_text((100, 320), "Figure 1: Accuracy versus model size.", fontsize=10)
    page.insert_text((72, 360), "More body text follows here.", fontsize=11)
    pdf = tmp_path / "2510.01235.pdf"
    document.save(str(pdf))
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    parsed: ParsedDoc = await parser("pymupdf4llm").parse(None, str(pdf))
    assert isinstance(parsed.figures, list) and all(isinstance(f, Figure) for f in parsed.figures)
    assert len(parsed.figures) == 1
    figure = parsed.figures[0]
    assert (figure.number, figure.page, figure.caption) == (
        "1",
        2,
        "Figure 1: Accuracy versus model size.",
    )
    assert parsed.sections[1][figure.caption_start : figure.caption_end] == figure.caption
    x0, y0, x1, y1 = figure.box
    assert x0 < 110 and y0 < 130 and x1 > 390 and y1 > 290
    assert parsed.metadata["figure_boxes_uncaptioned"] == "0"
    # A page of text alone has no figures.
    plain = await parser("pymupdf4llm").parse(None, write_pdf(tmp_path / "p.pdf", ["Text."]))
    assert plain.figures == []


@pytest.mark.asyncio
async def test_pymupdf4llm_reports_image_only_pdfs_as_needing_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    with pytest.raises(ValueError, match="OCR"):
        await parser("pymupdf4llm").parse(None, write_pdf(tmp_path / "blank.pdf", [""]))


def fake_docling(monkeypatch: pytest.MonkeyPatch, pages: list[str]) -> dict[str, object]:
    """Install a minimal stand-in for the Docling API surface the plugin uses."""
    seen: dict[str, object] = {}

    class PdfPipelineOptions:
        do_ocr = False
        do_table_structure = False

    class Document:
        def num_pages(self) -> int:
            return len(pages)

        def export_to_markdown(self, *, page_no: int | None = None) -> str:
            assert page_no is not None
            return pages[page_no - 1]

    class DocumentConverter:
        def __init__(self, format_options: dict[str, Any]) -> None:
            seen["options"] = format_options["pdf"].pipeline_options
            seen["built"] = cast(int, seen.get("built", 0)) + 1

        def convert(self, source: str) -> Any:
            seen["source"] = source
            return types.SimpleNamespace(document=Document())

    def pdf_format_option(pipeline_options: PdfPipelineOptions) -> types.SimpleNamespace:
        return types.SimpleNamespace(pipeline_options=pipeline_options)

    modules = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.base_models": types.SimpleNamespace(
            InputFormat=types.SimpleNamespace(PDF="pdf")
        ),
        "docling.datamodel.pipeline_options": types.SimpleNamespace(
            PdfPipelineOptions=PdfPipelineOptions
        ),
        "docling.document_converter": types.SimpleNamespace(
            DocumentConverter=DocumentConverter,
            PdfFormatOption=pdf_format_option,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return seen


@pytest.mark.asyncio
async def test_docling_extracts_pages_with_ocr_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = fake_docling(monkeypatch, ["## Attention Is All You Need\n\nIntro", "Results"])
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    parsed: ParsedDoc = await parser("docling").parse(None, "/papers/1706.03762.pdf")
    assert parsed.title == "Attention Is All You Need"
    assert parsed.sections == ["## Attention Is All You Need\n\nIntro", "Results"]
    assert parsed.metadata["parser"] == "docling" and parsed.metadata["ocr"] == "true"
    options: Any = seen["options"]
    assert options.do_ocr is True and options.do_table_structure is True


@pytest.mark.asyncio
async def test_docling_loads_its_models_once_per_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = fake_docling(monkeypatch, ["## Title\n\nText"])
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    plugin = parser("docling")
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        await plugin.parse(None, f"/papers/{name}")
    assert seen["built"] == 1 and seen["source"] == "/papers/c.pdf"


@pytest.mark.asyncio
async def test_docling_fails_explicitly_when_no_text_survives_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_docling(monkeypatch, ["", "  "])
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    with pytest.raises(ValueError, match="even with OCR"):
        await parser("docling", do_ocr=True).parse(None, "/papers/scan.pdf")
