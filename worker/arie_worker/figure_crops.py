"""Figure crops (S24): the paper's own figure, rendered from the stored PDF in a worker.

A crop is the figure's box (``Figure.box``, PDF points) clipped to its page and rendered
once at ``dpi``, scaled down once if its longest side would pass ``max_side`` pixels. A
crop whose PNG is over ``max_bytes``, past the first ``max_crops`` of a paper, or whose box
is empty or off the page is skipped, never shrunk again. Rendering runs only in the PDF
worker pool, never in the API process (ADR 0005, 0013). ARIE shows the crop; it never
describes it.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import pymupdf

type Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CropLimits:
    dpi: int = 150
    max_side: int = 1600
    max_bytes: int = 400 * 1024
    max_crops: int = 60


@dataclass(frozen=True, slots=True)
class Crops:
    """PNG bytes by the figure's position in the input, and why the others were skipped."""

    images: dict[int, bytes] = field(default_factory=lambda: dict[int, bytes]())
    skipped: dict[str, int] = field(default_factory=lambda: dict[str, int]())


class _Pixmap(Protocol):
    width: int
    height: int

    def tobytes(self, output: str) -> bytes: ...


class _Rect(Protocol):
    @property
    def x0(self) -> float: ...
    @property
    def y0(self) -> float: ...
    @property
    def x1(self) -> float: ...
    @property
    def y1(self) -> float: ...


class _Page(Protocol):
    @property
    def rect(self) -> _Rect: ...
    def get_pixmap(self, *, matrix: pymupdf.Matrix, clip: pymupdf.Rect, alpha: bool) -> _Pixmap: ...


class _Document(Protocol):
    page_count: int

    def __getitem__(self, index: int) -> _Page: ...
    def close(self) -> None: ...


open_pdf = cast(Callable[[str], _Document], pymupdf.open)


def clip_box(page: _Page, box: Box) -> Box | None:
    """``box`` clipped to the page, or None when less than a point is left on either side."""
    edge = page.rect
    x0, y0 = max(box[0], edge.x0), max(box[1], edge.y0)
    x1, y1 = min(box[2], edge.x1), min(box[3], edge.y1)
    return (x0, y0, x1, y1) if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def render_crops(
    file_path: str, figures: Sequence[tuple[int, Box]], limits: CropLimits | None = None
) -> Crops:
    """Render each ``(page, box)`` within ``limits``; a failure skips that figure only."""
    os.environ["ARIE_PDF_WORKER"] = "1"
    limits = limits or CropLimits()
    images: dict[int, bytes] = {}
    skipped: Counter[str] = Counter()
    doc = open_pdf(file_path)
    try:
        for index, (page_number, box) in enumerate(figures):
            if len(images) >= limits.max_crops:
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
            # One scale, never a retry: 150 dpi, or less so the longest side fits. The 2 px
            # margin absorbs MuPDF rounding a fractional clip outward.
            longest = max(clip[2] - clip[0], clip[3] - clip[1])
            zoom = min(limits.dpi / 72, (limits.max_side - 2) / longest)
            try:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(*clip), alpha=False
                )
                png = pixmap.tobytes("png")
            except RuntimeError:
                skipped["render_failed"] += 1
                continue
            if max(pixmap.width, pixmap.height) > limits.max_side or len(png) > limits.max_bytes:
                skipped["too_large"] += 1
                continue
            images[index] = png
    finally:
        doc.close()
    return Crops(images, dict(skipped))


__all__ = ["Box", "CropLimits", "Crops", "clip_box", "open_pdf", "render_crops"]
