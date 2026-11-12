"""Figure captions and mentions as plain text facts (ADR 0015).

Everything here reads the paper's own words and layout boxes. Nothing describes an image:
a figure is known only by its number, where it is drawn and its caption.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

import msgspec

from arie_core.contracts import Figure
from arie_core.plugin_api.figures import caption_number, figure_mentions


def locate_caption(page_text: str, number: str, caption: str) -> tuple[int, int]:
    """Offsets of a caption in a page's text: the exact caption, else the paragraph that opens
    with that figure's label; ``(-1, -1)`` when neither is there."""
    if caption:
        start = page_text.find(caption)
        if start >= 0:
            return start, start + len(caption)
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", page_text):
        if caption_number(paragraph.group()) == number:
            offset = len(paragraph.group()) - len(paragraph.group().lstrip())
            return paragraph.start() + offset, paragraph.end()
    return -1, -1


def chunk_figure_tags(
    text: str, page: int, char_start: int, char_end: int, figures: Sequence[Figure]
) -> dict[str, object]:
    """``figure`` (the caption this chunk holds) and ``mentions`` (other figures it names).

    Returns only the keys that are set, so a chunk without figures stays as it was.
    """
    figure = ""
    for item in figures:
        if item.page == page and char_start <= item.caption_start < char_end:
            figure = item.number
            break
    if not figure:
        number = caption_number(text)
        if number and any(item.page == page and item.number == number for item in figures):
            figure = number
    mentions = [number for number in figure_mentions(text) if number != figure]
    tags: dict[str, object] = {}
    if figure:
        tags["figure"] = figure
    if mentions:
        tags["mentions"] = mentions
    return tags


def figure_records(rows: Sequence[Mapping[str, object]]) -> list[Figure]:
    """Stored figure dicts (``paper.figures``) back as ``Figure``s; malformed rows are skipped."""
    figures: list[Figure] = []
    for row in rows:
        try:
            figures.append(msgspec.convert(row, Figure))
        except msgspec.ValidationError:
            continue
    return figures


def attach_figures(document: dict[str, object], parsed: Sequence[Figure]) -> int:
    """Add a fresh parse's figures to a stored paper without changing its pages or chunks.

    Captions are found again in the stored page text, so offsets address what is stored
    (-1 when a caption is not there). Chunks gain ``figure``/``mentions`` tags; their text,
    ids and offsets stay, so vectors, guard scores and packs remain valid. Returns the number
    of captions not found in the stored text.
    """
    pages = cast(list[dict[str, object]], document.get("pages") or [])
    texts = {int(cast(int, page["number"])): str(page["text"]) for page in pages}
    figures: list[Figure] = []
    missing = 0
    for figure in parsed:
        start, end = locate_caption(texts.get(figure.page, ""), figure.number, figure.caption)
        missing += start < 0
        figures.append(msgspec.structs.replace(figure, caption_start=start, caption_end=end))
    document["figures"] = msgspec.to_builtins(figures)
    for chunk in cast(list[dict[str, object]], document.get("chunks") or []):
        chunk.pop("figure", None)
        chunk.pop("mentions", None)
        chunk.update(
            chunk_figure_tags(
                str(chunk["text"]),
                int(cast(int, chunk["page_start"])),
                int(cast(int, chunk["char_start"])),
                int(cast(int, chunk["char_end"])),
                figures,
            )
        )
    return missing
