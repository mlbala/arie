"""Figure labels, captions and picture-caption pairing for parser plugins (ADR 0015).

Pure text and geometry: a figure is known only by its number, its box and its caption.
Nothing here describes an image.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

Box = tuple[float, float, float, float]

# At most this many figures are kept per paper.
MAX_FIGURES = 60
# A caption below its picture may sit under further sub-panels; one above must be close.
MAX_GAP_BELOW = 250.0
MAX_GAP_ABOVE = 60.0

# One figure label: "3", "A1"; a sub-panel suffix ("3b", "3(b)", "3 (a-c)") belongs to it.
_LABEL = r"[A-Z]?\d+(?:[a-z]\b|\s*\([a-z0-9][a-z0-9,\s\-–]*\))?"
_MENTION = re.compile(
    rf"\b(?:Figures?|FIGURES?|figures?|Figs?\.?|FIGS?\.?)\s*~?\s*"
    rf"({_LABEL}(?:\s*(?:,|and|&|to|–|-)\s*(?:and\s+)?{_LABEL})*)"
)
_NUMBER = re.compile(r"[A-Z]?\d+")
# A caption opens its paragraph with the label and then punctuation or the end of bold text.
_CAPTION = re.compile(
    r"^\s*[*_]*(?:Figure|FIGURE|Fig\.?|FIG\.?)\s*([A-Z]?\d+)\s*(?:[*_]+|[:.|–—-])"
)


def figure_mentions(text: str) -> list[str]:
    """Figure numbers that ``text`` refers to, in order, once each ("Figs. 2 and 3" -> 2, 3).

    A sub-panel names its figure ("Figure 3(b)" -> "3"); a numeric range ("Figures 2-4") is
    expanded when it spans at most ten figures.
    """
    found: list[str] = []
    for match in _MENTION.finditer(text):
        group = match.group(1)
        numbers = [m.group() for m in _NUMBER.finditer(re.sub(r"\([^)]*\)", "", group))]
        ranged = re.fullmatch(r"\s*(\d+)\s*(?:–|-|to)\s*(\d+)\s*", group)
        if ranged and 0 < int(ranged.group(2)) - int(ranged.group(1)) <= 10:
            first, last = int(ranged.group(1)), int(ranged.group(2))
            numbers = [str(n) for n in range(first, last + 1)]
        for number in numbers:
            if number not in found:
                found.append(number)
    return found


def caption_number(text: str) -> str:
    """The figure number when ``text`` starts like a caption ("Figure 3:", "**Fig. 3.**")."""
    match = _CAPTION.match(text)
    return match.group(1) if match else ""


def _overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0])


def match_captions(pictures: Sequence[Box], captions: Sequence[Box]) -> tuple[dict[int, Box], int]:
    """Pair each picture with the nearest caption box below it, else just above it.

    A caption must overlap the picture horizontally. Pictures that share a caption (sub-panels)
    merge into one box. Returns ``{caption index: figure box}`` and the number of pictures
    without a caption, which are skipped.
    """
    boxes: dict[int, Box] = {}
    uncaptioned = 0
    for picture in pictures:
        below = [
            (caption[1] - picture[3], index)
            for index, caption in enumerate(captions)
            if _overlaps(picture, caption) and -2.0 <= caption[1] - picture[3] <= MAX_GAP_BELOW
        ]
        above = [
            (picture[1] - caption[3], index)
            for index, caption in enumerate(captions)
            if _overlaps(picture, caption) and -2.0 <= picture[1] - caption[3] <= MAX_GAP_ABOVE
        ]
        choice = min(below) if below else min(above) if above else None
        if choice is None:
            uncaptioned += 1
            continue
        index = choice[1]
        old = boxes.get(index)
        boxes[index] = (
            picture
            if old is None
            else (
                min(old[0], picture[0]),
                min(old[1], picture[1]),
                max(old[2], picture[2]),
                max(old[3], picture[3]),
            )
        )
    return boxes, uncaptioned
