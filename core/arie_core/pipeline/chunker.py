from __future__ import annotations

import re


def chunk_spans(text: str, max_chars: int = 2200) -> list[tuple[int, int]]:
    """Return lossless paragraph spans; preserve table and equation blocks and source offsets."""
    spans: list[tuple[int, int]] = []
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        start, end = paragraph.span()
        block = paragraph.group()
        if not block.strip():
            continue
        if "|" in block or "$$" in block:
            spans.append((start, end))
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
