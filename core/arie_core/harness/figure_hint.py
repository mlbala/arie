"""The figure hint (S24): tell the reader which figure an unanswered strict question points to.

When strict mode abstains or falls back to excerpts, the highest-ranked of the top 3 rows
read that is a figure's caption, the words drawn inside it, or a passage that mentions a
figure picks the figure (a mention picks the first figure it mentions). The hint fires only
when the question is about that figure: it names it ("Figure 2", "Fig. 2"), or it shares at
least two content words with the figure's caption (``query_terms`` words, stop words out,
long words matched by their root). The page is the figure's own, from its caption chunk;
with no caption chunk the message names no page. The message states where the closest
passage points; it never says what the figure shows (ADR 0015).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import msgspec

from arie_core.harness.injection import is_untrusted
from arie_core.harness.reader import EvidenceChunk, query_terms
from arie_core.plugin_api.figures import figure_mentions

# Results that did not answer: the writer abstained, or the model path fell back to excerpts.
HINT_MODES = frozenset({"abstention", "source_excerpts"})
# How far down the rows read a figure passage may rank and still pick its figure.
HINT_DEPTH = 3
# Content words a question must share with the figure's caption when it does not name it.
MIN_SHARED_WORDS = 2
# Words every caption has; naming the figure is checked on its own.
_CAPTION_WORDS = frozenset({"figure", "figures", "fig", "figs"})


class FigureHint(msgspec.Struct, frozen=True):
    number: str
    # The figure's page, or 0 when the paper has no caption chunk for it.
    page: int

    @property
    def message(self) -> str:
        where = f" (p. {self.page})" if self.page > 0 else ""
        return f"The closest passage refers to Figure {self.number}{where}."


def figure_pages(rows: Sequence[EvidenceChunk]) -> dict[str, int]:
    """Each figure's page, from the first chunk that holds its caption."""
    pages: dict[str, int] = {}
    for row in rows:
        if row.figure and not row.figure_text:
            pages.setdefault(row.figure, row.page_start)
    return pages


def figure_captions(rows: Sequence[EvidenceChunk]) -> dict[str, str]:
    """Each figure's caption: its caption chunk from the "Figure N" label to the paragraph end."""
    captions: dict[str, str] = {}
    for row in rows:
        if not row.figure or row.figure_text or row.figure in captions:
            continue
        label = re.search(rf"\b(?:Figure|Fig\.?)\s*{re.escape(row.figure)}\b", row.text)
        text = row.text[label.start() :] if label else row.text
        captions[row.figure] = text.split("\n\n", 1)[0]
    return captions


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\w+", text.lower())
        if word not in _CAPTION_WORDS and not word.isdigit()
    }


def shared_words(question: str, caption: str) -> int:
    """Question content words found in the caption, by the word or, if long, its 6-letter root."""
    keys = query_terms(caption)
    words = _content_words(question) & query_terms(question)
    return sum(bool({word, word[:6] if len(word) >= 7 else word} & keys) for word in words)


def is_about(question: str, number: str, caption: str) -> bool:
    """The question names the figure, or shares enough content words with its caption."""
    if number in figure_mentions(question):
        return True
    return bool(caption) and shared_words(question, caption) >= MIN_SHARED_WORDS


def figure_hint(
    mode: str,
    paper_id: str,
    question: str,
    rows: Sequence[EvidenceChunk],
    pages: Mapping[str, int] | None = None,
    captions: Mapping[str, str] | None = None,
) -> FigureHint | None:
    """The figure a non-answer points to, when the question is about it.

    ``rows`` are the rows read, best first; ``pages`` and ``captions`` map figure numbers to
    their pages and captions (``figure_pages``/``figure_captions`` over the whole paper).
    """
    if mode not in HINT_MODES:
        return None
    scoped = [row for row in rows if row.paper_id == paper_id and not is_untrusted(row)]
    picked = next((row for row in scoped[:HINT_DEPTH] if row.figure or row.mentions), None)
    if picked is None:
        return None
    number = picked.figure or picked.mentions[0]
    if not is_about(question, number, (captions or {}).get(number, "")):
        return None
    # A caption or figure-text chunk sits on its figure's page.
    page = (pages or {}).get(number, picked.page_start if picked.figure else 0)
    return FigureHint(number, page)


__all__ = [
    "HINT_DEPTH",
    "HINT_MODES",
    "MIN_SHARED_WORDS",
    "FigureHint",
    "figure_captions",
    "figure_hint",
    "figure_pages",
    "is_about",
    "shared_words",
]
