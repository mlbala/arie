from __future__ import annotations

import re

# A "Table N" caption paragraph, optionally bold or italic as PDF-to-Markdown writes it.
_TABLE_CAPTION = re.compile(r"^\s*[*_]*Table\s+[A-Z]?\d", re.IGNORECASE)
_TABLE_LINE = re.compile(r"^\s*\|", re.MULTILINE)


def is_table(block: str) -> bool:
    return bool(_TABLE_LINE.search(block))


def chunk_spans(text: str, max_chars: int = 2200) -> list[tuple[int, int]]:
    """Return lossless paragraph spans; preserve table and equation blocks and source offsets.

    A "Table N" caption paragraph directly before or after a table joins the table's
    span, so retrieval by caption words finds the table and a quote can carry both.
    """
    spans: list[tuple[int, int]] = []
    tables: set[int] = set()
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        start, end = paragraph.span()
        block = paragraph.group()
        if not block.strip():
            continue
        if "|" in block or "$$" in block:
            if (
                is_table(block)
                and spans
                and _is_caption(text, spans[-1])
                and _adjacent(text, spans[-1], start)
            ):
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
            if is_table(block):
                tables.add(len(spans) - 1)
            continue
        # A caption below its table joins it, unless the table already has one above.
        if (
            _TABLE_CAPTION.match(block)
            and len(block) <= max_chars
            and len(spans) - 1 in tables
            and _adjacent(text, spans[-1], start)
            and not _has_caption(text, spans[-1])
        ):
            spans[-1] = (spans[-1][0], end)
            continue
        while end - start > max_chars:
            split = text.rfind(" ", start, start + max_chars)
            if split <= start:
                split = start + max_chars
            spans.append((start, split))
            start = split
        if end > start:
            spans.append((start, end))
    return spans


def _is_caption(text: str, span: tuple[int, int]) -> bool:
    block = text[span[0] : span[1]]
    return bool(_TABLE_CAPTION.match(block)) and not is_table(block)


def _has_caption(text: str, span: tuple[int, int]) -> bool:
    return bool(_TABLE_CAPTION.match(text[span[0] : span[1]]))


def _adjacent(text: str, span: tuple[int, int], start: int) -> bool:
    """Only blank lines separate the span from ``start``."""
    return not text[span[1] : start].strip()
