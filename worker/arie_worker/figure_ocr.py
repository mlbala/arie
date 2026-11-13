"""OCR text for bitmap figures (S25): words printed inside a figure the PDF draws as an image.

Off by default (``ingestion.figure_ocr``). Each figure whose record has no drawn words
(``Figure.text_spans`` is empty) is rendered from its box, clipped to the page, and read by
RapidOCR, one line per detected text box in the engine's reading order. The render uses the
crop settings (150 dpi, longest side at most 1,600 px), and text detection works on that
render without enlarging it. Recognition uses the pinned English
model (``arie_worker.ocr_model``, ADR 0018), so the engine emits only ASCII; letters from
other scripts are still stripped as a backstop. Lines under ``min_score`` or with no ASCII
letter or digit are dropped, and the text is cut at ``max_chars``. This runs only in the PDF
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
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pymupdf

from arie_worker.figure_crops import Box, clip_box, open_pdf
from arie_worker.ocr_model import OcrModelError, verify_model

# RapidOCR's result: one ``[box, text, score]`` per detected line, or None when there is none.
type OcrLines = list[tuple[object, str, float]] | None
type OcrEngine = Callable[[bytes], tuple[OcrLines, object]]

_ENGINES: dict[str, OcrEngine] = {}
# Text detection never enlarges an image past the render's longest side (OcrLimits.max_side).
DETECT_MAX_SIDE = 1600
_LATIN = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True, slots=True)
class OcrLimits:
    # The crop render settings (``CropLimits``): ADR 0018 measured them against 216 dpi/2,000 px.
    dpi: int = 150
    max_side: int = 1600
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


def load_engine(model: str) -> OcrEngine:
    """This worker's RapidOCR engine with the English recognition ``model``, built on first use.

    The model file is checked against its pinned SHA-256 before it is loaded.
    """
    if not model:
        raise OcrModelError("no OCR recognition model given (ingestion.figure_ocr_model)")
    if model not in _ENGINES:
        verify_model(Path(model))
        module = importlib.import_module("rapidocr_onnxruntime")
        # Detection sees the render as it is: RapidOCR's default enlarges the short side to
        # 736 px, which took a worker to 888 MB on the 78-figure set; capped, 371 MB (ADR 0018).
        engine = module.RapidOCR(
            rec_model_path=model, det_limit_type="max", det_limit_side_len=DETECT_MAX_SIDE
        )
        _ENGINES[model] = cast(OcrEngine, engine)
    return _ENGINES[model]


def _foreign(char: str) -> bool:
    """A wide (CJK) character, or a letter outside the Latin and Greek scripts."""
    if unicodedata.east_asian_width(char) in ("W", "F"):
        return True
    script = unicodedata.name(char, "")
    return unicodedata.category(char).startswith("L") and not script.startswith(("LATIN", "GREEK"))


def latin_text(text: str) -> str:
    """``text`` without characters from other scripts, whitespace collapsed."""
    return " ".join("".join(char for char in text if not _foreign(char)).split())


def lines_text(lines: OcrLines, limits: OcrLimits) -> str:
    """The engine's lines over ``min_score``, stripped of other scripts, that keep an ASCII
    letter or digit; one per line, cut at ``max_chars``."""
    kept: list[str] = []
    for item in lines or []:
        text, score = latin_text(str(item[1])), float(item[2])
        if score >= limits.min_score and _LATIN.search(text):
            kept.append(text)
    return "\n".join(kept)[: limits.max_chars].strip()


def ocr_figures(
    file_path: str,
    figures: Sequence[tuple[int, Box]],
    limits: OcrLimits | None = None,
    engine: OcrEngine | None = None,
    model: str = "",
) -> OcrTexts:
    """OCR each ``(page, box)`` within ``limits``; a failure skips that figure only.

    ``engine`` is for tests; otherwise this worker's engine with recognition ``model`` is used.
    """
    os.environ["ARIE_PDF_WORKER"] = "1"
    limits = limits or OcrLimits()
    engine = engine or load_engine(model)
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


__all__ = [
    "OcrEngine",
    "OcrLimits",
    "OcrTexts",
    "latin_text",
    "lines_text",
    "load_engine",
    "ocr_available",
    "ocr_figures",
]
