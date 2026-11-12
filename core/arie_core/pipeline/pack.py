"""The paper pack: one paper as a single, byte-stable text with ``[cN]`` chunk markers.

Explain mode puts the pack at the front of the prompt so the writer, the judge and the guide
can reuse one cached prefix (ADR 0010). ``N`` is the number in the chunk id
``<paper_id>:<N>``. The pack is pure: the same inputs always give the same bytes, and it
reads nothing but its arguments.

Paper text is untrusted (ADR 0013). Chunks that read as instructions to the assistant are left
out, as on the strict path, and every role sends the pack inside ``<paper>`` fences
(``fence_pack``) that paper text cannot open or close.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal

import msgspec

from arie_core.harness.reader import (
    EvidenceChunk,
    annotate_sections,
    eligible,
    is_untrusted_instruction,
)
from arie_core.harness.session import estimate_tokens

# Bump when the layout changes, so stored packs are rebuilt.
PACK_FORMAT = "p2"

_MARKER = re.compile(r"\[c(\d+)\]")
# A fence tag, or one spelled to look like it, inside paper text.
_FENCE = re.compile(r"<(?=\s*/?\s*paper\b)", re.IGNORECASE)
FENCE_OPEN = "<paper>"
FENCE_CLOSE = "</paper>"


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


def escape_paper(text: str) -> str:
    """Escape paper text for the pack: literal markers, and ``<paper>`` tags as ``&lt;paper>``.

    So text inside a PDF can neither cite as ARIE nor close the fence around the paper.
    """
    return _FENCE.sub("&lt;", escape_markers(text))


def fence_pack(pack_text: str) -> str:
    """The pack as every explain-mode role sends it: between ``<paper>`` and ``</paper>``.

    Pure and the same for every role, so the writer, judge and guide share one cached prefix.
    """
    body = pack_text.strip("\n")
    return f"{FENCE_OPEN}\n{body}\n{FENCE_CLOSE}"


def marker_numbers(text: str) -> tuple[int, ...]:
    """Chunk numbers of the real markers in a pack. Paper text is escaped, so only ours match."""
    return tuple(int(match[1]) for match in _MARKER.finditer(text))


def chunk_number(paper_id: str, chunk_id: str) -> int | None:
    """``N`` from ``<paper_id>:<N>``, or ``None`` for an id that is not this paper's."""
    prefix, _, number = chunk_id.rpartition(":")
    return int(number) if prefix == paper_id and number.isascii() and number.isdigit() else None


def pack_chunks(paper_id: str, chunks: Sequence[EvidenceChunk]) -> list[tuple[int, EvidenceChunk]]:
    """The chunks a pack keeps, in page order, each with its number and annotated section.

    Chunks of other papers are ignored (I1). Contents listings, credits, references and
    acknowledgements are left out by ``eligible``, the rule the strict answer path uses, and so
    are chunks that are only a heading. Chunks that read as instructions to the assistant
    (``is_untrusted_instruction``) are left out too, as on the strict path. Everything else
    stays, equations and captions included: the pack is the whole paper, not a selection of
    prose.
    """
    numbered: list[tuple[int, EvidenceChunk]] = []
    for row in chunks:
        number = chunk_number(paper_id, row.chunk_id)
        if (
            row.paper_id == paper_id
            and number is not None
            and not is_untrusted_instruction(row.text)
        ):
            numbered.append((number, row))
    # Page order, not input order, so the bytes don't depend on how the rows were loaded.
    numbered.sort(key=lambda item: (item[1].page_start, item[1].char_start, item[0]))
    numbers = {row.chunk_id: number for number, row in numbered}
    return [
        (numbers[row.chunk_id], row)
        for row in eligible("", annotate_sections([row for _, row in numbered]))
        if _has_text(row)
    ]


def render_blocks(numbered: Sequence[tuple[int, EvidenceChunk]]) -> list[str]:
    """One ``[cN] text`` block per chunk; a block that starts a section opens with its name."""
    blocks: list[str] = []
    section = ""
    for number, row in numbered:
        block = f"[c{number}] {escape_paper(row.text.strip())}"
        if row.section and row.section != section:
            block = f"Section: {escape_paper(row.section)}\n{block}"
        section = row.section or section
        blocks.append(block)
    return blocks


def build_pack(
    paper_id: str,
    title: str,
    abstract: str,
    revision: str,
    chunks: Sequence[EvidenceChunk],
) -> PaperPack:
    """Title, abstract, section outline, then every kept chunk (``pack_chunks``) as ``[cN]``.

    ``revision`` identifies the paper version (the vector index's
    ``<dedup_hash>:<chunk_count>``). An abstract that reads as instructions to the assistant is
    left out, like such a chunk.
    """
    kept = pack_chunks(paper_id, chunks)
    outline = list(dict.fromkeys(row.section for _, row in kept if row.section))
    parts = [f"Title: {escape_paper(title.strip())}"]
    if abstract.strip() and not is_untrusted_instruction(abstract):
        parts.append(f"Abstract:\n{escape_paper(abstract.strip())}")
    if outline:
        parts.append("Outline:\n" + "\n".join(f"- {escape_paper(name)}" for name in outline))
    parts.extend(render_blocks(kept))

    text = "\n\n".join(parts) + "\n"
    return PaperPack(
        text=text,
        chunk_numbers=tuple(number for number, _ in kept),
        token_estimate=estimate_tokens(len(text.encode())),
        revision=pack_revision(revision),
    )


def pack_header(pack: PaperPack) -> str:
    """Title, abstract and outline: the pack's text before its first chunk block."""
    first = _MARKER.search(pack.text)
    if first is None:
        return pack.text.removesuffix("\n")
    head = pack.text[: first.start()]
    if not head.endswith("\n\n"):
        # The first block opens with its "Section: <name>" line; section names hold no "\n\n".
        head = head[: head.rfind("\n\n") + 2]
    return head.removesuffix("\n\n")


def pack_limit(context_tokens: int, share: float) -> int:
    """The most estimated tokens a pack may have: ``share`` of the provider's context window."""
    return int(Decimal(str(share)) * context_tokens)


def pack_fits(pack: PaperPack, context_tokens: int, share: float) -> bool:
    """Whether the whole pack stays within ``share`` of the provider's context window."""
    return pack.token_estimate <= pack_limit(context_tokens, share)
