"""Figure crops and the "see Figure N" hint (S24): ARIE shows a figure, never describes it."""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from typing import Any

import msgspec
import pytest
from arie_api.runtime import context
from arie_core.harness.figure_hint import FigureHint, figure_hint
from arie_core.harness.orchestrator import answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.pipeline.figures import chunk_figures
from arie_worker.figure_crops import CropLimits, render_crops

# pymupdf ships no complete types; the tests only build small PDFs with it.
pymupdf: Any = importlib.import_module("pymupdf")
BOX = (100.0, 120.0, 400.0, 300.0)


def figure_pdf(path: Path) -> str:
    """The S22 contract-test page: a drawn figure with its caption below it."""
    document = pymupdf.open()
    page = document.new_page()
    page.draw_rect(pymupdf.Rect(*BOX), color=(0, 0, 0), fill=(0.8, 0.2, 0.2))
    page.insert_text((100, 330), "Figure 2: Accuracy by model size on Cedar.", fontsize=11)
    document.save(str(path))
    document.close()
    return str(path)


def png_size(png: bytes) -> tuple[int, int]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    return width, height


def test_a_figure_renders_to_a_png_within_the_limits(tmp_path: Path) -> None:
    crops = render_crops(figure_pdf(tmp_path / "f.pdf"), [(1, BOX)])
    assert crops.skipped == {}
    png = crops.images[0]
    # 300 x 180 points at 150 dpi (MuPDF may round the clip outward by a pixel).
    width, height = png_size(png)
    assert abs(width - 625) <= 1 and abs(height - 375) <= 1
    assert len(png) <= CropLimits().max_bytes


def test_a_full_page_figure_is_scaled_once_to_the_longest_side(tmp_path: Path) -> None:
    pdf = figure_pdf(tmp_path / "f.pdf")
    # A letter page at 150 dpi is 1,650 px tall: one smaller scale, not skipped.
    crops = render_crops(pdf, [(1, (-20.0, -20.0, 700.0, 900.0))])
    width, height = png_size(crops.images[0])
    assert max(width, height) <= 1600 and height >= 1590


def test_oversize_extra_and_broken_figures_are_skipped_not_shrunk(tmp_path: Path) -> None:
    pdf = figure_pdf(tmp_path / "f.pdf")
    tiny = CropLimits(max_bytes=200)
    assert render_crops(pdf, [(1, BOX)], tiny).skipped == {"too_large": 1}
    two = render_crops(pdf, [(1, BOX), (1, BOX)], CropLimits(max_crops=1))
    assert list(two.images) == [0] and two.skipped == {"too_many": 1}
    broken = render_crops(pdf, [(3, BOX), (1, (700.0, 800.0, 900.0, 950.0)), (0, BOX)])
    assert broken.images == {} and broken.skipped == {"bad_box": 3}


def row(chunk_id: str, text: str, page: int, **tags: object) -> EvidenceChunk:
    return msgspec.convert(
        dict(
            paper_id="p",
            chunk_id=chunk_id,
            text=text,
            page_start=page,
            char_start=0,
            char_end=len(text),
            **tags,
        ),
        EvidenceChunk,
    )


CAPTION = row("c3", "Figure 2: Accuracy by model size on Cedar.", 5, figure="2")
DRAWN = row("c4", "Small Base Large", 5, figure="2", figure_text=True)
MENTION = row("c1", "Accuracy rises with size, see Figure 2.", 4, mentions=["2"])
PLAIN = row("c0", "Cedar uses a gated memory bank.", 1)


def test_an_abstention_whose_top_row_is_the_caption_names_the_figure_and_page() -> None:
    hint = figure_hint("abstention", "p", [CAPTION, PLAIN])
    assert hint is not None and hint == FigureHint("2", 5)
    assert hint.message == "Not stated in the text; see Figure 2 (p. 5)."
    assert figure_hint("source_excerpts", "p", [DRAWN, PLAIN]) == FigureHint("2", 5)


def test_no_hint_without_a_top_figure_row_or_for_an_answer() -> None:
    assert figure_hint("abstention", "p", [PLAIN, CAPTION]) is None
    assert figure_hint("abstention", "p", [MENTION, CAPTION]) is None
    assert figure_hint("abstention", "p", []) is None
    assert figure_hint("answer", "p", [CAPTION]) is None
    assert figure_hint("extractive_summary", "p", [CAPTION]) is None
    # Another paper's caption is never the top row of this one.
    other = msgspec.structs.replace(CAPTION, paper_id="q")
    assert figure_hint("abstention", "p", [other, PLAIN]) is None


@pytest.mark.asyncio
async def test_a_strict_fallback_on_a_caption_carries_the_hint_and_keeps_its_reason() -> None:
    question = "How does accuracy change with model size on Cedar?"
    hinted = await answer_question(context("hint"), "p", question, [CAPTION])
    assert hinted.answer.mode == "source_excerpts"
    assert hinted.figure_hint == FigureHint("2", 5)
    assert hinted.answer.trace.figure_hint == "2"
    plain = await answer_question(context("plain"), "p", question, [PLAIN])
    assert plain.figure_hint is None and plain.answer.trace.figure_hint == ""
    assert plain.answer.fallback_reason == hinted.answer.fallback_reason


def test_a_citation_lists_the_figure_it_captions_then_the_ones_it_names() -> None:
    assert chunk_figures("2", ("3", "2")) == ["2", "3"]
    assert chunk_figures("", ["4"]) == ["4"]
    assert chunk_figures(None, None) == []
