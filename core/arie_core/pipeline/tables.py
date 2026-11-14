"""Table records: a Markdown table's caption, header and data rows as offsets (S27).

Everything here reads the stored chunk text. Offsets slice back to the exact text, so a row
can be quoted on its own and still pass the quote checks like any sentence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

import msgspec

from arie_core.contracts import Table
from arie_core.pipeline.chunker import is_table

# A "Table N" caption paragraph, optionally bold or italic as PDF-to-Markdown writes it.
_CAPTION = re.compile(r"^\s*[*_]*(?:Table|Tab\.)\s*([A-Z]?\d+)", re.IGNORECASE)
_SEPARATOR = re.compile(r"^[\s|:\-]*-[\s|:\-]*$")


class TableLayout(msgspec.Struct, frozen=True):
    """Offsets of one table's parts in the text ``table_layout`` read; spans are ``(-1, -1)``
    when absent. ``rows`` are the data rows, each without its newline."""

    number: str
    caption: tuple[int, int]
    header: tuple[int, int]
    rows: tuple[tuple[int, int], ...]


def _lines(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every line, without the newline."""
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        spans.append((start, start + len(body)))
        start += len(line)
    return spans


def _lstrip(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """``span`` without its leading whitespace."""
    start, end = span
    return start + len(text[start:end]) - len(text[start:end].lstrip()), end


def _is_row(text: str, span: tuple[int, int]) -> bool:
    return text[span[0] : span[1]].lstrip().startswith("|")


def table_layout(text: str) -> TableLayout | None:
    """The first Markdown table in ``text``: its caption, header lines and data rows.

    The header is every table line above the ``|---|`` separator (the first line when there
    is none). A data row needs a letter or digit, so empty filler rows are skipped. The
    caption is the first "Table N" paragraph of non-table lines in ``text``, above or below.
    Returns None when ``text`` holds no table.
    """
    if not is_table(text):
        return None
    lines = _lines(text)
    first = next(i for i, span in enumerate(lines) if _is_row(text, span))
    last = first
    while last + 1 < len(lines) and _is_row(text, lines[last + 1]):
        last += 1
    block = lines[first : last + 1]
    separator = next(
        (i for i, span in enumerate(block) if i and _SEPARATOR.match(text[span[0] : span[1]])),
        None,
    )
    head = block[:separator] if separator is not None else block[:1]
    body = block[separator + 1 :] if separator is not None else block[1:]
    rows = tuple(
        _lstrip(text, (start, end))
        for start, end in body
        if re.search(r"[^\W_]", text[start:end].replace("|", ""))
        and not _SEPARATOR.match(text[start:end])
    )
    header_start = _lstrip(text, head[0])[0]
    number, caption = "", (-1, -1)
    paragraph: list[tuple[int, int]] = []
    for index, span in enumerate([*lines, (len(text), len(text))]):
        inside = first <= index <= last
        if index < len(lines) and not inside and text[span[0] : span[1]].strip():
            paragraph.append(span)
            continue
        if paragraph:
            opening = text[paragraph[0][0] : paragraph[0][1]]
            if match := _CAPTION.match(opening):
                offset = len(opening) - len(opening.lstrip())
                number, caption = match.group(1), (paragraph[0][0] + offset, paragraph[-1][1])
                break
            paragraph = []
    return TableLayout(number, caption, (header_start, head[-1][1]), rows)


def chunk_table_tags(text: str) -> dict[str, object]:
    """``table`` (the "Table N" this chunk holds) for a table chunk with a caption, else {}."""
    layout = table_layout(text)
    return {"table": layout.number} if layout is not None and layout.number else {}


def table_records(chunks: Sequence[Mapping[str, object]]) -> list[Table]:
    """One ``Table`` per stored table chunk, offsets in its page's text; OCR chunks skipped."""
    tables: list[Table] = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        layout = None if chunk.get("ocr") else table_layout(text)
        if layout is None:
            continue
        base = int(cast(int, chunk.get("char_start", 0)))
        start, end = layout.caption
        tables.append(
            Table(
                number=layout.number,
                page=int(cast(int, chunk.get("page_start", 0))),
                caption=text[start:end] if start >= 0 else "",
                chunk_id=str(chunk.get("chunk_id", "")),
                header_start=base + layout.header[0],
                header_end=base + layout.header[1],
                rows=tuple((base + a, base + b) for a, b in layout.rows),
                caption_start=base + start if start >= 0 else -1,
                caption_end=base + end if start >= 0 else -1,
            )
        )
    return tables


def table_context(text: str, layout: TableLayout) -> str:
    """The caption and header lines of a table, as the writer and Tier 3 see them with a row.

    Both are exact substrings of ``text``, joined by a newline; never another data row.
    """
    parts = [text[span[0] : span[1]] for span in (layout.caption, layout.header) if span[0] >= 0]
    return "\n".join(part.strip() for part in parts if part.strip())


# Table rows as sentences, for checking only (S28, ADR 0020). Never shown, never stored.

_TAG = re.compile(r"<br\s*/?>|<[^>]+>", re.IGNORECASE)
_CAPTION_PARTS = re.compile(r"^(Table|Tab\.)\s*([A-Z]?\d+)\s*[:.]?\s*(.*)$", re.IGNORECASE | re.S)


def table_cells(line: str) -> list[str]:
    """The cells of one ``| a | b |`` line, emphasis, HTML tags and extra spaces removed."""
    body = line.strip()
    body = body.removeprefix("|").removesuffix("|")
    cells = [_TAG.sub(" ", cell).replace("*", "").replace("`", "") for cell in body.split("|")]
    return [" ".join(cell.split()) for cell in cells]


def _caption_lead(caption: str) -> str:
    """``"Table 2 (Results on SQuAD; dev set)"`` from ``"**Table 2:** Results on SQuAD. Dev set."``.

    Inner sentence ends become ``;`` so the lead stays one sentence for the checks.
    """
    text = " ".join(caption.replace("*", "").split()).strip("_ ")
    if not text:
        return ""
    match = _CAPTION_PARTS.match(text)
    if match is None:
        return re.sub(r"[.!?]+\s+", "; ", text).rstrip(" .")
    label = f"Table {match.group(2)}"
    about = re.sub(r"[.!?]+\s+", "; ", match.group(3)).rstrip(" .")
    return f"{label} ({about})" if about else label


def linearize_row(caption: str, header: str, row: str) -> str:
    """One data row as a sentence: ``"Table 2 (Results): Model = BERT, F1 = 85.3."``.

    ``header`` may hold several lines; the last one names the columns. An empty cell is
    skipped, a cell under an empty or missing header is given alone, and without a caption
    the sentence starts at the cells. Every cell's text is kept as written.
    """
    lines = [line for line in header.splitlines() if line.strip()]
    names = table_cells(lines[-1]) if lines else []
    parts: list[str] = []
    for index, value in enumerate(table_cells(row)):
        if not value:
            continue
        name = names[index] if index < len(names) else ""
        parts.append(f"{name} = {value}" if name and name != value else value)
    lead = _caption_lead(caption)
    body = ", ".join(parts)
    return f"{lead}: {body}." if lead else f"{body}."


def table_premise(text: str) -> str | None:
    """``text`` with its first table written as sentences, for checking a claim against it.

    The header and table lines become one sentence per data row, each its own paragraph and
    led by "Table N" (nothing when uncaptioned). The caption stays once, as written, so its
    words (and any "only" or "not") are not repeated into every row. Every row is kept: a
    match on row labels missed rows named another way (S28 measurement, ADR 0020). The text
    around the table stays as it is. Returns None when ``text`` holds no table.
    """
    layout = table_layout(text)
    if layout is None:
        return None
    header = text[layout.header[0] : layout.header[1]]
    lead = f"Table {layout.number}" if layout.number else ""
    sentences = [linearize_row(lead, header, text[start:end]) for start, end in layout.rows]
    end = layout.rows[-1][1] if layout.rows else layout.header[1]
    return "\n\n".join(
        part.strip()
        for part in (text[: layout.header[0]], "\n\n".join(sentences), text[end:])
        if part.strip()
    )


def linearize_quote(context: str, quote: str) -> str:
    """A table-row quote with its S27 ``context`` (caption, then header lines) as a sentence."""
    lines = context.splitlines()
    header = "\n".join(line for line in lines if line.lstrip().startswith("|"))
    caption = " ".join(line for line in lines if not line.lstrip().startswith("|"))
    return linearize_row(caption, header, quote)
