from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from arie_core.plugin_api import Ctx, Figure, ParsedDoc, plugin
from arie_core.plugin_api.figures import (
    MAX_FIGURES,
    Box,
    caption_number,
    match_captions,
    overlap_area,
    picture_text_spans,
)


def _box(value: object) -> Box | None:
    if not isinstance(value, list | tuple):
        return None
    items = list(cast(Sequence[object], value))
    if len(items) != 4 or not all(isinstance(v, int | float) for v in items):
        return None
    x0, y0, x1, y1 = (float(cast(float, v)) for v in items)
    return (x0, y0, x1, y1)


def _span(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple):
        return None
    items = list(cast(Sequence[object], value))
    if len(items) != 2 or not all(isinstance(v, int) for v in items):
        return None
    return cast(int, items[0]), cast(int, items[1])


def page_figures(number: int, page: Mapping[str, object]) -> tuple[list[Figure], int]:
    """Captioned figures of one page from pymupdf4llm's layout boxes, and the count of
    picture boxes without a caption. Caption offsets address this page's ``text``.

    The words a picture box draws are already in ``text`` as a picture-text block at the
    box's ``pos``. Each block goes to the figure whose box overlaps that picture box most;
    a block whose picture overlaps no figure stays untagged. Nothing is appended (S23).
    """
    text = str(page.get("text", ""))
    boxes = page.get("page_boxes")
    if not isinstance(boxes, list):
        return [], 0
    pictures: list[Box] = []
    blocks: list[tuple[Box, tuple[int, int]]] = []
    captions: list[tuple[str, Box, int, int]] = []
    for raw in cast(list[object], boxes):
        if not isinstance(raw, dict):
            continue
        item = cast(dict[str, object], raw)
        bbox = _box(item.get("bbox"))
        if bbox is None:
            continue
        pos = _span(item.get("pos"))
        if item.get("class") == "picture":
            pictures.append(bbox)
            if pos is not None:
                blocks.extend((bbox, span) for span in picture_text_spans(text, *pos))
            continue
        if item.get("class") not in {"caption", "text"} or pos is None:
            continue
        start, end = pos
        block = text[start:end]
        label = caption_number(block)
        if label:
            start += len(block) - len(block.lstrip())
            end = start + len(block.strip())
            captions.append((label, bbox, start, end))
    paired, uncaptioned = match_captions(pictures, [caption[1] for caption in captions])
    owned: dict[int, list[tuple[int, int]]] = {index: [] for index in paired}
    for picture, span in blocks:
        best = max(paired, key=lambda index: overlap_area(paired[index], picture), default=None)
        if best is not None and overlap_area(paired[best], picture) > 0:
            owned[best].append(span)
    figures: list[Figure] = []
    for index, box in sorted(paired.items()):
        label, _, start, end = captions[index]
        spans = tuple(sorted(owned[index]))
        figures.append(Figure(label, number, box, text[start:end], start, end, spans))
    return figures, uncaptioned


@plugin(slot="parser", name="pymupdf4llm")
class PyMuPDF4LLMParserPlugin:
    """Extract page Markdown in worker processes; empty or image-only PDFs fail explicitly."""

    async def parse(self, ctx: Ctx, file_path: str) -> ParsedDoc:
        del ctx
        if os.environ.get("ARIE_PDF_WORKER") != "1":
            raise RuntimeError("PDF extraction must run inside arie-worker")
        module = importlib.import_module("pymupdf4llm")
        extract = cast(Callable[..., object], module.to_markdown)
        result = extract(file_path, page_chunks=True, show_progress=False)
        if not isinstance(result, list):
            raise ValueError("Parser did not return page-level extraction")
        pages = cast(list[dict[str, object]], result)
        texts = [str(page.get("text", "")) for page in pages]
        if not any(text.strip() for text in texts):
            raise ValueError("No text could be extracted; this PDF may require OCR")
        headings = [
            match.group(1).strip()
            for text in texts
            for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        ]
        figures: list[Figure] = []
        uncaptioned = 0
        for number, page in enumerate(pages, start=1):
            found, missing = page_figures(number, page)
            uncaptioned += missing
            seen = {figure.number for figure in figures}
            figures.extend(figure for figure in found if figure.number not in seen)
        return ParsedDoc(
            paper_id=Path(file_path).stem,
            title=headings[0] if headings else "Research paper",
            source="",
            sections=texts,
            metadata={
                "page_count": str(len(texts)),
                "outline_json": json.dumps(headings[:80]),
                "figure_count": str(min(len(figures), MAX_FIGURES)),
                "figure_boxes_uncaptioned": str(uncaptioned),
            },
            figures=figures[:MAX_FIGURES],
        )
