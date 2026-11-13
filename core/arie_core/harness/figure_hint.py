"""The "see Figure N" hint (S24): point an unanswered strict question at the paper's own figure.

When strict mode abstains or falls back to excerpts, the highest-ranked of the top 3 rows
read that is a figure's caption, the words drawn inside it, or a passage that mentions a
figure names that figure (a mention names the first figure it mentions). The page is the
figure's own, taken from its caption chunk; with no caption chunk the message names no page.
The hint names the figure; it never says what the figure shows (ADR 0015).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import msgspec

from arie_core.harness.injection import is_untrusted
from arie_core.harness.reader import EvidenceChunk

# Results that did not answer: the writer abstained, or the model path fell back to excerpts.
HINT_MODES = frozenset({"abstention", "source_excerpts"})
# How far down the rows read a figure passage may rank and still name its figure.
HINT_DEPTH = 3


class FigureHint(msgspec.Struct, frozen=True):
    number: str
    # The figure's page, or 0 when the paper has no caption chunk for it.
    page: int

    @property
    def message(self) -> str:
        where = f" (p. {self.page})" if self.page > 0 else ""
        return f"Not stated in the text; see Figure {self.number}{where}."


def figure_pages(rows: Sequence[EvidenceChunk]) -> dict[str, int]:
    """Each figure's page, from the first chunk that holds its caption."""
    pages: dict[str, int] = {}
    for row in rows:
        if row.figure and not row.figure_text:
            pages.setdefault(row.figure, row.page_start)
    return pages


def figure_hint(
    mode: str,
    paper_id: str,
    rows: Sequence[EvidenceChunk],
    pages: Mapping[str, int] | None = None,
) -> FigureHint | None:
    """The figure named by the best figure passage in this paper's top rows, for a non-answer.

    ``rows`` are the rows read, best first; ``pages`` maps figure numbers to their pages
    (``figure_pages`` over the whole paper).
    """
    if mode not in HINT_MODES:
        return None
    scoped = [row for row in rows if row.paper_id == paper_id and not is_untrusted(row)]
    for row in scoped[:HINT_DEPTH]:
        if row.figure:
            # A caption or figure-text chunk sits on its figure's page.
            return FigureHint(row.figure, (pages or {}).get(row.figure, row.page_start))
        if row.mentions:
            number = row.mentions[0]
            return FigureHint(number, (pages or {}).get(number, 0))
    return None


__all__ = ["HINT_DEPTH", "HINT_MODES", "FigureHint", "figure_hint", "figure_pages"]
