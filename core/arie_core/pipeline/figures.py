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
from arie_core.plugin_api.figures import caption_number, figure_mentions, picture_text_spans


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
    """``figure`` (the figure this chunk belongs to) and ``mentions`` (other figures it names).

    A chunk holding a caption (by the caption's offsets, else by its opening label) is that
    figure's caption chunk. A chunk that overlaps one of a figure's ``text_spans`` holds words
    drawn inside it: ``figure`` is set and ``figure_text`` is true (S23). Returns only the
    keys that are set, so a chunk without figures stays as it was.
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
    drawn = ""
    if not figure:
        drawn = next(
            (
                item.number
                for item in figures
                if item.page == page
                and any(start < char_end and char_start < end for start, end in item.text_spans)
            ),
            "",
        )
    mentions = [number for number in figure_mentions(text) if number not in {figure, drawn}]
    tags: dict[str, object] = {}
    if figure or drawn:
        tags["figure"] = figure or drawn
    if drawn:
        tags["figure_text"] = True
    if mentions:
        tags["mentions"] = mentions
    return tags


def chunk_figures(figure: object, mentions: object) -> list[str]:
    """A chunk's figure numbers for its citation: the one it captions, then the ones it names."""
    numbers = [figure] if isinstance(figure, str) and figure else []
    if isinstance(mentions, list | tuple):
        numbers += [str(n) for n in cast(Sequence[object], mentions) if str(n) not in numbers]
    return numbers


OCR_KEY = "ocr_text"


def ocr_targets(rows: Sequence[Mapping[str, object]]) -> list[int]:
    """Positions of the stored figures OCR may read (S25): those with no drawn words."""
    return [
        index
        for index, record in enumerate(figure_records_by_index(rows))
        if record is not None and not record.text_spans
    ]


def ocr_chunks(
    paper_id: str, rows: Sequence[Mapping[str, object]], first: int
) -> list[dict[str, object]]:
    """One chunk per stored figure with OCR text (``ocr_text``, S25), numbered from ``first``.

    The text is a transcription of the figure's image, not the paper's text layer, so its
    offsets address the chunk's own text (0 to its length) on the figure's page. The chunk is
    tagged ``figure``, ``figure_text`` (never a caption) and ``ocr``.
    """
    chunks: list[dict[str, object]] = []
    for row, record in zip(rows, figure_records_by_index(rows), strict=True):
        text = row.get(OCR_KEY)
        if record is None or not isinstance(text, str) or not text.strip():
            continue
        chunks.append(
            {
                "paper_id": paper_id,
                "chunk_id": f"{paper_id}:{first + len(chunks)}",
                "page_start": record.page,
                "char_start": 0,
                "char_end": len(text),
                "text": text,
                "figure": record.number,
                "figure_text": True,
                "ocr": True,
            }
        )
    return chunks


def figure_records_by_index(rows: Sequence[Mapping[str, object]]) -> list[Figure | None]:
    """Stored figure dicts as ``Figure``s in place; a malformed row is None."""
    records: list[Figure | None] = []
    for row in rows:
        try:
            records.append(msgspec.convert(row, Figure))
        except msgspec.ValidationError:
            records.append(None)
    return records


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
    (-1 when a caption is not there). A figure-text span is kept only when it is exactly a
    picture-text block of the stored page. Chunks gain ``figure``/``figure_text``/``mentions``
    tags; their text, ids and offsets stay, so vectors, guard scores and packs remain valid.
    OCR chunks are dropped, since they belong to the old records.
    Returns the number of captions not found in the stored text.
    """
    pages = cast(list[dict[str, object]], document.get("pages") or [])
    texts = {int(cast(int, page["number"])): str(page["text"]) for page in pages}
    figures: list[Figure] = []
    missing = 0
    for figure in parsed:
        page_text = texts.get(figure.page, "")
        start, end = locate_caption(page_text, figure.number, figure.caption)
        missing += start < 0
        stored = set(picture_text_spans(page_text))
        spans = tuple(span for span in figure.text_spans if span in stored)
        figures.append(
            msgspec.structs.replace(figure, caption_start=start, caption_end=end, text_spans=spans)
        )
    document["figures"] = msgspec.to_builtins(figures)
    # OCR chunks (S25) belong to the old figure records; the caller builds them again.
    if any(
        chunk.get("ocr") for chunk in cast(list[dict[str, object]], document.get("chunks") or [])
    ):
        document["chunks"] = [
            chunk
            for chunk in cast(list[dict[str, object]], document["chunks"])
            if not chunk.get("ocr")
        ]
    for chunk in cast(list[dict[str, object]], document.get("chunks") or []):
        chunk.pop("figure", None)
        chunk.pop("figure_text", None)
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
