"""OCR text for bitmap figures (S25): words printed inside a figure the PDF draws as an image.

Off by default (``ingestion.figure_ocr``). Each figure whose record has no drawn words
(``Figure.text_spans`` is empty) is rendered from its box, clipped to the page, and read by
RapidOCR, one line per detected text box in the engine's reading order. Lines under
``min_score`` are dropped, and the text is cut at ``max_chars``. This runs only in the PDF
worker pool, under the parse time and memory limits (ADR 0005, 0013); the engine is an
optional extra (``uv sync --extra ocr``) loaded once per worker process.

The result is a transcription of the paper's image, not the paper's text layer (ADR 0015,
S25 amendment): strict answers may show it only as a labelled excerpt, never as a claim.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import pymupdf

from arie_worker.figure_crops import Box, clip_box, open_pdf

# RapidOCR's result: one ``[box, text, score]`` per detected line, or None when there is none.
type OcrLines = list[tuple[object, str, float]] | None
type OcrEngine = Callable[[bytes], tuple[OcrLines, object]]

_ENGINE: list[OcrEngine] = []
_LATIN = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True, slots=True)
class OcrLimits:
    dpi: int = 216
    max_side: int = 2000
    max_figures: int = 60
    max_chars: int = 2000
    min_score: float = 0.5


@dataclass(frozen=True, slots=True)
class OcrTexts:
    """OCR text by the figure's position in the input, and why the others have none."""

    texts: dict[int, str] = field(default_factory=lambda: dict[int, str]())
    skipped: dict[str, int] = field(default_factory=lambda: dict[str, int]())


def ocr_available() -> bool:
    """Whether the optional OCR engine is installed; never imports it (API process)."""
    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def load_engine() -> OcrEngine:
    """The RapidOCR engine of this worker process, built on first use."""
    if not _ENGINE:
        module = importlib.import_module("rapidocr_onnxruntime")
        _ENGINE.append(cast(OcrEngine, module.RapidOCR()))
    return _ENGINE[0]


def lines_text(lines: OcrLines, limits: OcrLimits) -> str:
    """The engine's lines over ``min_score`` with an ASCII letter or digit, one per line, cut at
    ``max_chars``."""
    kept: list[str] = []
    for item in lines or []:
        text, score = " ".join(str(item[1]).split()), float(item[2])
        # The bundled model reads Chinese too; a line with no ASCII letter or digit is noise.
        if score >= limits.min_score and _LATIN.search(text):
            kept.append(text)
    return "\n".join(kept)[: limits.max_chars].strip()


def ocr_figures(
    file_path: str,
    figures: Sequence[tuple[int, Box]],
    limits: OcrLimits | None = None,
    engine: OcrEngine | None = None,
) -> OcrTexts:
    """OCR each ``(page, box)`` within ``limits``; a failure skips that figure only."""
    os.environ["ARIE_PDF_WORKER"] = "1"
    limits = limits or OcrLimits()
    engine = engine or load_engine()
    texts: dict[int, str] = {}
    skipped: Counter[str] = Counter()
    doc = open_pdf(file_path)
    try:
        for index, (page_number, box) in enumerate(figures):
            if index >= limits.max_figures:
                skipped["too_many"] += 1
                continue
            if not 1 <= page_number <= doc.page_count:
                skipped["bad_box"] += 1
                continue
            page = doc[page_number - 1]
            clip = clip_box(page, box)
            if clip is None:
                skipped["bad_box"] += 1
                continue
            longest = max(clip[2] - clip[0], clip[3] - clip[1])
            zoom = min(limits.dpi / 72, limits.max_side / longest)
            try:
                png = page.get_pixmap(
                    matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(*clip), alpha=False
                ).tobytes("png")
                lines, _ = engine(png)
            except Exception:
                # One unreadable image must not cost the paper its other figures.
                skipped["ocr_failed"] += 1
                continue
            text = lines_text(lines, limits)
            if text:
                texts[index] = text
            else:
                skipped["no_text"] += 1
    finally:
        doc.close()
    return OcrTexts(texts, dict(skipped))


__all__ = ["OcrEngine", "OcrLimits", "OcrTexts", "lines_text", "ocr_available", "ocr_figures"]
