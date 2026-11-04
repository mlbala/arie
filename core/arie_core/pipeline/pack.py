"""The paper pack: one paper as a single, byte-stable text with ``[cN]`` chunk markers.

Explain mode puts the pack at the front of the prompt so the writer, the judge and the guide
can reuse one cached prefix (ADR 0010). ``N`` is the number in the chunk id
``<paper_id>:<N>``. The pack is pure: the same inputs always give the same bytes, and it
reads nothing but its arguments.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

import msgspec

from arie_core.harness.reader import EvidenceChunk, annotate_sections, eligible
from arie_core.harness.session import estimate_tokens

# Bump when the layout changes, so stored packs are rebuilt.
PACK_FORMAT = "p1"

_MARKER = re.compile(r"\[c(\d+)\]")


def _has_text(row: EvidenceChunk) -> bool:
    """False for a chunk that is only a heading or whitespace; its section is kept by name."""
    return any(
        line.strip() and not line.lstrip().startswith("#") for line in row.text.splitlines()
    )


class PaperPack(msgspec.Struct, frozen=True):
    text: str
    # Chunk numbers in the order their markers appear in ``text``.
    chunk_numbers: tuple[int, ...]
    token_estimate: int
    revision: str


def pack_revision(paper_revision: str) -> str:
    """The revision a pack built from this paper revision carries."""
    return f"{paper_revision}:{PACK_FORMAT}"


def escape_markers(text: str) -> str:
    """Turn a literal ``[c12]`` from the paper into ``[\\c12]``, which no citation parser reads."""
    return _MARKER.sub(r"[\\c\1]", text)


def marker_numbers(text: str) -> tuple[int, ...]:
    """Chunk numbers of the real markers in a pack. Paper text is escaped, so only ours match."""
    return tuple(int(match[1]) for match in _MARKER.finditer(text))


def chunk_number(paper_id: str, chunk_id: str) -> int | None:
    """``N`` from ``<paper_id>:<N>``, or ``None`` for an id that is not this paper's."""
    prefix, _, number = chunk_id.rpartition(":")
    return int(number) if prefix == paper_id and number.isascii() and number.isdigit() else None


def build_pack(
    paper_id: str,
    title: str,
    abstract: str,
    revision: str,
    chunks: Sequence[EvidenceChunk],
) -> PaperPack:
    """Title, abstract, section outline, then every kept chunk as ``[cN]`` plus its text.

    Chunks of other papers are ignored (I1). Contents listings, credits, references and
    acknowledgements are left out by ``eligible``, the rule the strict answer path uses, and so
    are chunks that are only a heading. Everything else stays, equations and captions
    included: the pack is the whole paper, not a selection of prose. ``revision`` identifies
    the paper version (the vector index's ``<dedup_hash>:<chunk_count>``).
    """
    numbered: list[tuple[int, EvidenceChunk]] = []
    for row in chunks:
        number = chunk_number(paper_id, row.chunk_id)
        if row.paper_id == paper_id and number is not None:
            numbered.append((number, row))
    # Page order, not input order, so the bytes don't depend on how the rows were loaded.
    numbered.sort(key=lambda item: (item[1].page_start, item[1].char_start, item[0]))
    numbers = {row.chunk_id: number for number, row in numbered}
    kept = [
        row
        for row in eligible("", annotate_sections([row for _, row in numbered]))
        if _has_text(row)
    ]

    outline = list(dict.fromkeys(row.section for row in kept if row.section))
    parts = [f"Title: {escape_markers(title.strip())}"]
    if abstract.strip():
        parts.append(f"Abstract:\n{escape_markers(abstract.strip())}")
    if outline:
        parts.append("Outline:\n" + "\n".join(f"- {escape_markers(name)}" for name in outline))
    section = ""
    for row in kept:
        block = f"[c{numbers[row.chunk_id]}] {escape_markers(row.text.strip())}"
        if row.section and row.section != section:
            block = f"Section: {escape_markers(row.section)}\n{block}"
        section = row.section or section
        parts.append(block)

    text = "\n\n".join(parts) + "\n"
    return PaperPack(
        text=text,
        chunk_numbers=tuple(numbers[row.chunk_id] for row in kept),
        token_estimate=estimate_tokens(len(text.encode())),
        revision=pack_revision(revision),
    )


def pack_fits(pack: PaperPack, context_tokens: int, share: float) -> bool:
    """Whether the whole pack stays within ``share`` of the provider's context window."""
    return pack.token_estimate <= Decimal(str(share)) * context_tokens
