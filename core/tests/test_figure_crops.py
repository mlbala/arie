"""Figure crops and the "see Figure N" hint (S24): ARIE shows a figure, never describes it."""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from typing import Any

import msgspec
import pytest
from arie_api.runtime import context
from arie_core.harness.figure_hint import (
    FigureHint,
    figure_captions,
    figure_hint,
    figure_pages,
    is_about,
    shared_words,
)
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


ROWS = [PLAIN, MENTION, CAPTION, DRAWN]
PAGES = figure_pages(ROWS)
CAPTIONS = figure_captions(ROWS)
ABOUT = "How does accuracy change with model size?"  # shares "accuracy", "model", "size"
NAMED = "What does Fig. 2 show?"
UNRELATED = "Which optimizer do they use?"


def hint(question: str, rows: list[EvidenceChunk], mode: str = "abstention") -> FigureHint | None:
    return figure_hint(mode, "p", question, rows, PAGES, CAPTIONS)


def test_pages_and_captions_come_from_caption_chunks() -> None:
    assert PAGES == {"2": 5}
    assert CAPTIONS == {"2": "Figure 2: Accuracy by model size on Cedar."}
    text = "Results are below.\n\nFig. 4. Loss per epoch.\n\nNext section."
    long = row("c9", text, 7, figure="4")
    assert figure_captions([long]) == {"4": "Fig. 4. Loss per epoch."}


def test_a_question_about_the_figure_gets_a_hint_that_states_where_the_passage_points() -> None:
    found = hint(ABOUT, [CAPTION, PLAIN])
    assert found == FigureHint("2", 5)
    assert found is not None
    assert found.message == "The closest passage refers to Figure 2 (p. 5)."
    assert hint(ABOUT, [DRAWN, PLAIN], "source_excerpts") == FigureHint("2", 5)
    # A question that names the figure needs no shared words.
    assert hint(NAMED, [PLAIN, MENTION]) == FigureHint("2", 5)


def test_shared_words_count_content_words_by_word_or_root() -> None:
    caption = "Figure 2: Accuracy by model size on Cedar."
    assert shared_words(ABOUT, caption) == 3
    # Stop words, "figure" and numbers never count; long words match by their root.
    assert shared_words("What is in the figure 2?", caption) == 0
    assert shared_words("Is the accuracies of models reported?", caption) == 1
    assert is_about("Report the accuracies by size", "2", caption)
    assert not is_about("Report accuracy", "2", caption)
    assert not is_about(UNRELATED, "2", "")


def test_no_hint_when_the_question_is_not_about_the_picked_figure() -> None:
    assert hint(UNRELATED, [CAPTION, PLAIN]) is None
    # The picked passage decides: a lower figure passage is not tried instead.
    other = row("c7", "Figure 3: Training cost.", 6, figure="3")
    assert hint(ABOUT, [other, CAPTION]) is None


def test_a_mentioned_figure_without_a_caption_chunk_fires_only_when_named() -> None:
    lone = row("c8", "Results improve, see Figure 7.", 4, mentions=["7"])
    found = hint("What does Figure 7 show?", [lone])
    assert found == FigureHint("7", 0)
    assert found is not None and found.message == "The closest passage refers to Figure 7."
    assert hint("Do results improve?", [lone]) is None
    # Two figures mentioned: the first one is picked.
    both = row("c5", "Figures 3 and 2 compare both runs.", 6, mentions=["3", "2"])
    assert hint("What does Figure 3 compare?", [both, CAPTION]) == FigureHint("3", 0)


def test_no_hint_below_the_top_three_or_for_an_answer() -> None:
    plain = [msgspec.structs.replace(PLAIN, chunk_id=f"c{i}") for i in range(3)]
    assert hint(NAMED, [*plain, CAPTION]) is None
    assert hint(NAMED, []) is None
    assert hint(NAMED, [CAPTION], "answer") is None
    assert hint(NAMED, [CAPTION], "extractive_summary") is None
    # Another paper's rows never count, and do not take a place in the top 3.
    other = msgspec.structs.replace(CAPTION, paper_id="q")
    assert hint(NAMED, [other, PLAIN]) is None
    assert hint(NAMED, [other, *plain[:2], MENTION]) == FigureHint("2", 5)


@pytest.mark.asyncio
async def test_a_strict_fallback_on_a_caption_carries_the_hint_and_keeps_its_reason() -> None:
    question = "How does accuracy change with model size on Cedar?"
    hinted = await answer_question(context("hint"), "p", question, [CAPTION])
    assert hinted.answer.mode == "source_excerpts"
    assert hinted.figure_hint == FigureHint("2", 5)
    assert hinted.answer.trace.figure_hint == "2"
    unrelated = await answer_question(context("off"), "p", "Which optimizer is used?", [CAPTION])
    assert unrelated.figure_hint is None and unrelated.answer.trace.figure_hint == ""
    plain = await answer_question(context("plain"), "p", question, [PLAIN])
    assert plain.figure_hint is None and plain.answer.trace.figure_hint == ""
    assert plain.answer.fallback_reason == hinted.answer.fallback_reason


def test_a_citation_lists_the_figure_it_captions_then_the_ones_it_names() -> None:
    assert chunk_figures("2", ("3", "2")) == ["2", "3"]
    assert chunk_figures("", ["4"]) == ["4"]
    assert chunk_figures(None, None) == []
