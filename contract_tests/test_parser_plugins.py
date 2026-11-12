"""Parser plugins: page-level text, explicit OCR failures, and worker-only execution."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from arie_core.harness.reader import locate_quote
from arie_core.pipeline.chunker import chunk_spans
from arie_core.pipeline.figures import chunk_figure_tags
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


def diagram_pdf(path: Path) -> str:
    """One page: a sentence, a vector diagram with three labelled boxes, its caption."""
    pymupdf = importlib.import_module("pymupdf")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "As Figure 1 shows, the system has three parts.", fontsize=11)
    page.draw_rect(pymupdf.Rect(100, 120, 450, 300), color=(0, 0, 0), fill=(0.9, 0.9, 0.95))
    for x, label in ((115, "Encoder"), (225, "Retriever"), (345, "Decoder")):
        page.draw_rect(pymupdf.Rect(x, 180, x + 90, 230), color=(0, 0, 0), fill=(0.6, 0.8, 0.6))
        page.insert_text((x + 8, 210), label, fontsize=10)
    page.draw_line((205, 205), (225, 205))
    page.draw_line((315, 205), (345, 205))
    page.insert_text((100, 320), "Figure 1: The retrieval pipeline.", fontsize=10)
    page.insert_text((72, 360), "More body text follows here.", fontsize=11)
    document.save(str(path))
    return str(path)


def raw_pages(pdf: str) -> list[str]:
    """pymupdf4llm's own page text, before the plugin looks at it."""
    module = importlib.import_module("pymupdf4llm")
    pages = cast(list[dict[str, object]], module.to_markdown(pdf, page_chunks=True))
    return [str(page["text"]) for page in pages]


def tagged_chunks(text: str, figures: list[Figure]) -> list[tuple[str, dict[str, object]]]:
    return [
        (text[start:end], chunk_figure_tags(text[start:end], 1, start, end, figures))
        for start, end in chunk_spans(text)
    ]


@pytest.mark.asyncio
async def test_pymupdf4llm_tags_words_drawn_inside_a_figure_without_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    pdf = diagram_pdf(tmp_path / "2510.01236.pdf")
    parsed: ParsedDoc = await parser("pymupdf4llm").parse(None, pdf)
    [figure] = parsed.figures
    text = parsed.sections[0]
    # The page text is pymupdf4llm's, unchanged: the labels are there once, not appended.
    assert parsed.sections == raw_pages(pdf)
    assert text.count("Encoder") == 1 and text.count("Decoder") == 1
    [(start, end)] = figure.text_spans
    assert text[start:end] == "Encoder Retriever Decoder"
    chunks = tagged_chunks(text, parsed.figures)
    drawn = [(chunk, tags) for chunk, tags in chunks if tags.get("figure_text")]
    assert len(drawn) == 1
    chunk, tags = drawn[0]
    assert tags["figure"] == "1" and "Retriever" in chunk
    assert locate_quote(chunk, "Encoder") is not None
    # The caption keeps its own chunk, tagged as the caption, not as drawn words.
    assert ("Figure 1: The retrieval pipeline.", {"figure": "1"}) in [
        (chunk.strip(), tags) for chunk, tags in chunks
    ]


@pytest.mark.asyncio
async def test_pymupdf4llm_leaves_text_and_chunks_of_a_page_without_figures_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARIE_PDF_WORKER", "1")
    pdf = write_pdf(tmp_path / "plain.pdf", ["Gated memory retrieval uses a bank of slots."])
    parsed: ParsedDoc = await parser("pymupdf4llm").parse(None, pdf)
    assert parsed.figures == []
    assert parsed.sections == raw_pages(pdf)
    assert chunk_spans(parsed.sections[0]) == chunk_spans(raw_pages(pdf)[0])
    assert all(tags == {} for _, tags in tagged_chunks(parsed.sections[0], parsed.figures))


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
