"""Hidden text: text a reader cannot see is redacted from a copy before parsing (SPEC 6.2).

A text span is hidden when it is invisible (render mode 3), fully transparent, under
2 pt, near-white (every channel at least 0.95) with no image or darker filled drawing under
it, or entirely outside the page's crop box. Its glyph boxes are redacted, text only, in a
copy, which the parser then reads (ADR 0013). The stored PDF is never changed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pymupdf

MIN_FONT_SIZE = 2.0
NEAR_WHITE = 0.95
# MuPDF names every Type 3 font "Type3..."; such a glyph may set its own colour, so the
# colour and opacity rules skip it (pymupdf4llm does the same).
TYPE3_PREFIX = "Type3"
INVISIBLE = 3  # ``get_texttrace`` span type for render mode 3
RULES = ("invisible", "transparent", "tiny", "off_page", "white")
# ``apply_redactions`` options: leave images and drawings, remove text.
_IMAGES_KEEP = 0
_DRAWINGS_KEEP = 0
_TEXT_REMOVE = 0

type Box = tuple[float, float, float, float]
type Trace = list[dict[str, object]]


class _Rect(Protocol):
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...


class _Page(Protocol):
    """The part of ``pymupdf.Page`` this module uses (PyMuPDF's own hints are partial)."""

    @property
    def cropbox(self) -> _Rect: ...
    def get_texttrace(self) -> Trace: ...
    def get_image_info(self) -> Trace: ...
    def get_drawings(self) -> Trace: ...
    def add_redact_annot(self, quad: Box, *, fill: bool) -> object: ...
    def apply_redactions(self, images: int, graphics: int, text: int) -> bool: ...


class _Document(Protocol):
    def __iter__(self) -> Iterator[_Page]: ...
    def save(self, filename: str) -> None: ...
    def close(self) -> None: ...


_open_pdf = cast(Callable[[str], _Document], pymupdf.open)


@dataclass(frozen=True, slots=True)
class Span:
    """One hidden text span: the rule it hit, its text and its glyph boxes."""

    rule: str
    text: str
    glyphs: tuple[Box, ...]


@dataclass(frozen=True, slots=True)
class HiddenFacts:
    """Hidden characters removed from one PDF, per rule, and visible characters lost too."""

    chars: Counter[str]
    visible_lost: int

    @property
    def hidden_chars(self) -> int:
        return self.chars.total()


def _box(value: object) -> Box:
    x0, y0, x1, y1 = cast(Sequence[float], value)
    return (x0, y0, x1, y1)


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _rgb(color: tuple[float, ...]) -> tuple[float, ...]:
    if len(color) == 1:
        return (color[0],) * 3
    if len(color) == 4:
        c, m, y, k = color
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return color


def _near_white(color: tuple[float, ...] | None) -> bool:
    return bool(color) and all(channel >= NEAR_WHITE for channel in _rgb(color or ()))


class _Backdrop:
    """Images and non-white filled drawings on a page, read only if a near-white span needs it."""

    def __init__(self, page: _Page) -> None:
        self.page = page
        self.loaded = False
        self.images: list[Box] = []
        self.fills: list[tuple[Box, int]] = []

    def under(self, box: Box, seqno: int) -> bool:
        """True if an image, or a filled drawing painted before the span, overlaps ``box``."""
        if not self.loaded:
            self.loaded = True
            self.images = [_box(info["bbox"]) for info in self.page.get_image_info()]
            for path in self.page.get_drawings():
                fill = cast(tuple[float, ...] | None, path.get("fill"))
                opacity = cast(float | None, path.get("fill_opacity"))
                if fill and not _near_white(fill) and (opacity is None or opacity > 0):
                    self.fills.append((_box(path["rect"]), cast(int, path["seqno"])))
        return any(_overlaps(box, image) for image in self.images) or any(
            seq < seqno and _overlaps(box, rect) for rect, seq in self.fills
        )


def _rule(span: dict[str, object], page_box: Box, backdrop: _Backdrop) -> str | None:
    box = _box(span["bbox"])
    type3 = str(span.get("font", "")).startswith(TYPE3_PREFIX)
    if span["type"] == INVISIBLE:
        return "invisible"
    if not type3 and cast(float, span["opacity"]) <= 0:
        return "transparent"
    if cast(float, span["size"]) < MIN_FONT_SIZE:
        return "tiny"
    if not _overlaps(box, page_box):
        return "off_page"
    color = cast(tuple[float, ...], span["color"])
    if not type3 and _near_white(color) and not backdrop.under(box, cast(int, span["seqno"])):
        return "white"
    return None


def hidden_spans(page: _Page) -> list[Span]:
    """The page's hidden text spans, in drawing order."""
    # Trace boxes are unrotated and start at the crop box's top-left corner.
    crop = page.cropbox
    page_box: Box = (0.0, 0.0, crop.width, crop.height)
    backdrop = _Backdrop(page)
    spans: list[Span] = []
    for span in page.get_texttrace():
        rule = _rule(span, page_box, backdrop)
        if rule is None:
            continue
        chars = cast(tuple[tuple[int, int, object, object], ...], span["chars"])
        text = "".join(chr(char[0]) for char in chars)
        spans.append(Span(rule, text, tuple(_box(char[3]) for char in chars)))
    return spans


def _chars(page: _Page) -> Counter[str]:
    return Counter(
        chr(char[0])
        for span in page.get_texttrace()
        for char in cast(tuple[tuple[int, ...], ...], span["chars"])
    )


def _strip(doc: _Document) -> HiddenFacts:
    """Redact every hidden glyph box (text only) in ``doc`` and measure what went."""
    removed: Counter[str] = Counter()
    lost = 0
    for page in doc:
        spans = hidden_spans(page)
        if not spans:
            continue
        visible = _chars(page)
        for span in spans:
            removed[span.rule] += len(span.text)
            visible.subtract(span.text)
            for glyph in span.glyphs:
                if glyph[0] < glyph[2] and glyph[1] < glyph[3]:
                    page.add_redact_annot(glyph, fill=False)
        page.apply_redactions(images=_IMAGES_KEEP, graphics=_DRAWINGS_KEEP, text=_TEXT_REMOVE)
        lost += (visible - _chars(page)).total()
    return HiddenFacts(removed, lost)


def strip_hidden(src: str | Path, dst: str | Path) -> int:
    """Write ``src`` without its hidden text to ``dst``; return the hidden character count.

    Nothing is written when the count is 0, so the caller parses ``src`` itself.
    """
    doc = _open_pdf(str(src))
    try:
        facts = _strip(doc)
        if facts.hidden_chars:
            doc.save(str(dst))
    finally:
        doc.close()
    return facts.hidden_chars


def measure_hidden(path: str | Path) -> HiddenFacts:
    """Strip ``path`` in memory and report what went; nothing is written (``make pdf-report``)."""
    doc = _open_pdf(str(path))
    try:
        return _strip(doc)
    finally:
        doc.close()


__all__ = ["RULES", "HiddenFacts", "Span", "hidden_spans", "measure_hidden", "strip_hidden"]
