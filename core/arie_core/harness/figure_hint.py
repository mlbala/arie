""" "See Figure N" (S24): point an unanswered strict question at the paper's own figure.

When strict mode abstains or falls back to excerpts and the top-ranked row is a figure's
caption or the words drawn inside it, the reader is told which figure and page to look at.
The hint names the figure; it never says what the figure shows (ADR 0015).
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec

from arie_core.harness.injection import is_untrusted
from arie_core.harness.reader import EvidenceChunk

# Results that did not answer: the writer abstained, or the model path fell back to excerpts.
HINT_MODES = frozenset({"abstention", "source_excerpts"})


class FigureHint(msgspec.Struct, frozen=True):
    number: str
    page: int

    @property
    def message(self) -> str:
        return f"Not stated in the text; see Figure {self.number} (p. {self.page})."


def figure_hint(mode: str, paper_id: str, rows: Sequence[EvidenceChunk]) -> FigureHint | None:
    """The figure named by the top-ranked row of this paper, for a result that did not answer."""
    if mode not in HINT_MODES:
        return None
    top = next((row for row in rows if row.paper_id == paper_id and not is_untrusted(row)), None)
    if top is None or not top.figure:
        return None
    return FigureHint(top.figure, top.page_start)


__all__ = ["HINT_MODES", "FigureHint", "figure_hint"]
