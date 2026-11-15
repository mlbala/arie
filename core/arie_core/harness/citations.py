"""Deterministic citation checks for an explain-mode answer (ADR 0010, SPEC 8.9).

Free and instant, no model: the answer is split into sentences and each one is ``cited_ok``,
``flagged`` or ``background``. A sentence with ``[cN]`` markers is checked against the raw text
of the chunks it names, never against pack text (which escapes literal markers). A sentence
with no marker and no number is background the model may write uncited. The checks are
evidence for the reader and the eval, not proof: a paraphrase can be flagged and a wrong
claim built from the right numbers can pass. The entailment judge comes later (S12).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

import msgspec

from arie_core.harness.verifier import canonical_numbers, content_tokens
from arie_core.pipeline.chunker import is_table
from arie_core.pipeline.pack import marker_numbers
from arie_core.pipeline.tables import table_premise

CITED_OK = "cited_ok"
FLAGGED = "flagged"
BACKGROUND = "background"

INVALID_MARKER = "invalid_marker"
NUMBER_NOT_IN_SOURCE = "number_not_in_source"
LOST_QUALIFIER = "lost_qualifier"
UNCITED_NUMBER = "uncited_number"

# A claim is aligned to the source sentence that holds at least this share of its content
# stems, and at least this many of them. Without an aligned sentence the qualifier check is
# skipped: a whole chunk holds a "not" somewhere, so comparing against it would flag everything.
_ALIGN_MIN_SHARE = 0.5
_ALIGN_MIN_STEMS = 3

# The explain check's qualifier groups (S30): negation and hedge as in strict
# ``lost_qualifier``, with past forms of the negation verbs. The restriction group ("only",
# "small", "limited", "restricted") is left out: on Qasper 30 all 21 of its flags were false,
# since dropping "a small sample" or "only one topic" doesn't make a claim say more.
_EXPLAIN_QUALIFIERS = (
    r"\b(?:not|no|never|without|cannot|neither|fail(?:s|ed)?|unable|lack(?:s|ed)?"
    r"|remov(?:e|es|ed)|dispensing)\b|n't",
    r"\b(?:may|might|could|suggests?|appears?|potentially|preliminary)\b",
)
# "not only ... but also" adds and "whether ... or not" covers both cases: neither negates.
_NOT_NEGATION = re.compile(r"\bnot\s+only\b|\bor\s+not\b", re.I)
# Clauses of a source sentence: a qualifier binds only within its own clause.
_CLAUSE_BREAK = re.compile(r";|(?<!\d):(?!\d)|,\s+so\s")

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_TABLE_SEPARATOR = re.compile(r"^[\s|:-]*-[\s|:-]*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
# Terminal punctuation, closing quotes or emphasis, and the markers that follow the full stop.
_BOUNDARY = re.compile(r"""[.!?]+["'”’)\]*_]*(?:\s*\[c\d+\])*(?=\s|$)""")
_ANY_MARKER = re.compile(r"[ \t]*\[\\?c\d+\]")
_TRAILING_WORD = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$")
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "fig", "figs", "eq", "eqs", "sec", "tab", "al", "vs", "cf", "approx", "no"}
)


class SentenceCheck(msgspec.Struct, frozen=True):
    """One sentence of the answer: where it is, its markers and how the checks judged it.

    ``start`` and ``end`` are character offsets into the answer text, and ``text`` is
    ``text[start:end]`` with its markers. ``reasons`` is empty unless ``status`` is
    ``flagged``. ``table`` is true when a cited chunk holds a table (S28).
    """

    start: int
    end: int
    text: str
    status: str
    markers: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()
    table: bool = False


class CitationChecks(msgspec.Struct, frozen=True, kw_only=True):
    """The checks of a whole answer. ``cited`` counts the sentences that carry a marker.

    ``valid_rate`` is the share of those without a flag (``None`` if there are none) and
    ``background_share`` the share of all sentences that are background. A flagged sentence
    with no marker (``uncited_number``) is in ``flagged`` but not in ``cited``.
    """

    sentences: tuple[SentenceCheck, ...] = ()
    cited_ok: int = 0
    flagged: int = 0
    background: int = 0
    cited: int = 0
    valid_rate: float | None = None
    background_share: float | None = None

    def flagged_markers(self) -> frozenset[int]:
        """Markers that appear in at least one flagged sentence."""
        return frozenset(
            marker
            for sentence in self.sentences
            if sentence.status == FLAGGED
            for marker in sentence.markers
        )


def _close(spans: list[tuple[int, int]], current: tuple[int, int] | None) -> None:
    if current is not None and current[1] > current[0]:
        spans.append(current)


def _blocks(text: str) -> list[tuple[int, int]]:
    """Spans of the prose in Markdown ``text``: paragraphs, list items and table rows.

    Code blocks, headings, rules and table separators are skipped. Every list item and table
    row is a block of its own; a paragraph is its consecutive lines.
    """
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    fence = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        body = line.rstrip()
        end = line_start + len(body)
        indent = len(body) - len(body.lstrip())
        if fence:
            stripped = body.strip()
            if len(stripped) >= len(fence) and set(stripped) == {fence[0]}:
                fence = ""
            continue
        if opened := _FENCE.match(body):
            _close(spans, current)
            current, fence = None, opened[1]
        elif not body.strip() or _HEADING.match(body) or _RULE.match(body):
            _close(spans, current)
            current = None
        elif body.lstrip().startswith("|"):
            _close(spans, current)
            current = None
            if not _TABLE_SEPARATOR.match(body):
                spans.append((line_start + indent, end))
        elif item := _LIST_ITEM.match(body):
            _close(spans, current)
            current = (line_start + item.end(), end)
        elif current is None:
            current = (line_start + indent, end)
        else:
            current = (current[0], end)
    _close(spans, current)
    return spans


def _has_content(value: str) -> bool:
    return any(char.isalnum() for char in _ANY_MARKER.sub("", value))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in Markdown ``text``, in order.

    A sentence ends at ``.``, ``!`` or ``?`` followed by space, unless the word before is an
    abbreviation (``e.g.``, ``Fig.``, ``et al.``) or the next word is lowercase. Markers after
    the full stop (``... works. [c3]``) stay with the sentence they follow. A span with no
    letter or digit besides markers is dropped.
    """
    spans: list[tuple[int, int]] = []
    for block_start, block_end in _blocks(text):
        block = text[block_start:block_end]
        begin = 0
        for match in _BOUNDARY.finditer(block):
            word = _TRAILING_WORD.search(block[: match.start()])
            if word and word[1].lower() in _ABBREVIATIONS:
                continue
            following = block[match.end() :].lstrip()
            if following[:1].islower():
                continue
            spans.append((block_start + begin, block_start + match.end()))
            begin = match.end()
        spans.append((block_start + begin, block_end))
    trimmed: list[tuple[int, int]] = []
    for a, b in spans:
        piece = text[a:b]
        a += len(piece) - len(piece.lstrip())
        if _has_content(text[a:b]):
            trimmed.append((a, b))
    return trimmed


def _stems(value: str) -> frozenset[str]:
    """Content words without a final ``s``, cut to five letters: ``improves`` meets ``improve``."""
    tokens = content_tokens(unicodedata.normalize("NFKC", value))
    return frozenset(token.removesuffix("s")[:5] for token in tokens)


class _Aligner:
    """Finds the sentence of a cited chunk that a claim restates, to compare qualifiers."""

    def __init__(self) -> None:
        self._cache: dict[str, list[tuple[str, frozenset[str]]]] = {}

    def _sentences(self, raw: str) -> list[tuple[str, frozenset[str]]]:
        if raw not in self._cache:
            self._cache[raw] = [(raw[a:b], _stems(raw[a:b])) for a, b in sentence_spans(raw)]
        return self._cache[raw]

    def best(self, claim: str, texts: Sequence[str]) -> str | None:
        wanted = _stems(claim)
        if len(wanted) < _ALIGN_MIN_STEMS:
            return None
        best_share, best_text = 0.0, None
        for raw in texts:
            for sentence, stems in self._sentences(raw):
                shared = len(wanted & stems)
                share = shared / len(wanted)
                if shared >= _ALIGN_MIN_STEMS and share > best_share:
                    best_share, best_text = share, sentence
        return best_text if best_share >= _ALIGN_MIN_SHARE else None


def lost_qualifier_in_clause(claim: str, sentence: str) -> bool:
    """True if a clause of ``sentence`` that the claim restates has a negation or hedge the
    claim lacks (S30).

    A clause is restated when it shares a content stem with the claim; clauses end at ``;``,
    ``:`` and ``, so``, so "this does not mean X: Y" restated as Y keeps no "not". "Not only"
    and "or not" are not negations. Restriction words are not checked (strict still checks
    them with ``lost_qualifier``).
    """
    wanted = _stems(claim)
    clauses = [
        clause
        for clause in _CLAUSE_BREAK.split(_NOT_NEGATION.sub(" ", sentence))
        if wanted & _stems(clause)
    ]
    return any(
        any(re.search(group, clause, re.I) for clause in clauses)
        and not re.search(group, claim, re.I)
        for group in _EXPLAIN_QUALIFIERS
    )


def check_answer(text: str, sources: Mapping[int, str], *, tables: bool = True) -> CitationChecks:
    """Check every sentence of an explain answer; ``sources`` maps a marker to its raw chunk text.

    ``sources`` holds the chunks of the bound paper that the model saw, so a marker outside it
    is ``invalid_marker``. For a sentence with valid markers: ``number_not_in_source`` (a
    number that is in none of the cited chunks, ``92.10`` and ``92.1`` being equal),
    ``lost_qualifier`` (a clause of the cited sentence it restates has a negation or hedge
    that the claim lacks, ``lost_qualifier_in_clause``). A sentence with no marker is
    ``uncited_number`` if it states a number and background otherwise.

    A cited chunk that holds a table is read with that table as one sentence per row, with its
    column names, and the caption once (``table_premise``, S28). Both checks compare against
    those sentences. ``tables=False`` reads the raw pipes instead, as S10-S27 did; the eval
    uses it to count the difference.
    """
    aligner = _Aligner()
    checked: list[SentenceCheck] = []
    for start, end in sentence_spans(text):
        raw = text[start:end]
        markers = tuple(dict.fromkeys(marker_numbers(raw)))
        claim = _ANY_MARKER.sub("", raw).strip()
        numbers = canonical_numbers(claim)
        table = False
        reasons: list[str] = []
        if not markers:
            status = FLAGGED if numbers else BACKGROUND
            if numbers:
                reasons.append(UNCITED_NUMBER)
        else:
            found = [number for number in markers if number in sources]
            if len(found) < len(markers):
                reasons.append(INVALID_MARKER)
            if found:
                table = any(is_table(sources[number]) for number in found)
                texts = [
                    (table_premise(sources[number]) if tables else None) or sources[number]
                    for number in found
                ]
                if numbers - canonical_numbers(" ".join(texts)):
                    reasons.append(NUMBER_NOT_IN_SOURCE)
                restated = aligner.best(claim, texts)
                if restated is not None and lost_qualifier_in_clause(claim, restated):
                    reasons.append(LOST_QUALIFIER)
            status = FLAGGED if reasons else CITED_OK
        checked.append(SentenceCheck(start, end, raw, status, markers, tuple(reasons), table))

    count = {name: sum(item.status == name for item in checked) for name in (CITED_OK, BACKGROUND)}
    cited = sum(bool(item.markers) for item in checked)
    return CitationChecks(
        sentences=tuple(checked),
        cited_ok=count[CITED_OK],
        flagged=sum(item.status == FLAGGED for item in checked),
        background=count[BACKGROUND],
        cited=cited,
        valid_rate=round(count[CITED_OK] / cited, 3) if cited else None,
        background_share=round(count[BACKGROUND] / len(checked), 3) if checked else None,
    )
