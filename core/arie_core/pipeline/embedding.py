"""What the embedder reads for each chunk: the chunk with its context (S27).

Only the embedding input changes. The stored chunk (text, offsets, quotes) stays as it is, and
every vector maps back to its chunk id, so retrieval still returns whole chunks (I2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import msgspec

from arie_core.harness.injection import evidence_chunks
from arie_core.harness.reader import annotate_sections
from arie_core.pipeline.chunker import is_table
from arie_core.pipeline.figures import figure_records
from arie_core.pipeline.tables import table_layout
from arie_core.plugin_api.figures import caption_number

# Bumped when the embedding input changes; it ends the vector revision, so every paper
# re-embeds once in the background (SPEC 7).
EMBED_FORMAT = "e2"
# A table whose embedding text is longer is embedded as row groups (caption + header + rows).
ROW_GROUP_CHARS = 1500

KINDS = ("prose", "table", "caption", "figure_text", "ocr")


class EmbeddingUnit(msgspec.Struct, frozen=True):
    """One vector to embed: ``id`` is the chunk id, or ``<chunk_id>#t<k>`` for a row group."""

    id: str
    chunk_id: str
    kind: str
    text: str


def chunk_kind(chunk: Mapping[str, object]) -> str:
    """``ocr``, ``figure_text``, ``table``, ``caption`` (a figure's caption chunk) or ``prose``."""
    if chunk.get("ocr"):
        return "ocr"
    if chunk.get("figure_text"):
        return "figure_text"
    if is_table(str(chunk.get("text", ""))):
        return "table"
    if chunk.get("figure"):
        return "caption"
    return "prose"


def _figure_caption(paper: Mapping[str, object], number: str) -> str:
    """The figure's caption as "Figure N: <caption>" from its record, or "Figure N"."""
    rows = cast(Sequence[Mapping[str, object]], paper.get("figures") or [])
    caption = next((f.caption for f in figure_records(rows) if f.number == number), "").strip()
    if caption and caption_number(caption) == number:
        return caption
    return f"Figure {number}: {caption}" if caption else f"Figure {number}"


def embedding_text(chunk: Mapping[str, object], paper: Mapping[str, object], section: str) -> str:
    """The chunk as the embedder reads it: its section heading first; figure text and OCR
    chunks then their figure's caption; a table chunk its "Table N" caption, moved first."""
    text = str(chunk.get("text", ""))
    kind = chunk_kind(chunk)
    prefix = [section.strip()] if section.strip() else []
    if kind in {"figure_text", "ocr"} and chunk.get("figure"):
        prefix.append(_figure_caption(paper, str(chunk["figure"])))
    if kind == "table" and (layout := table_layout(text)) is not None and layout.caption[0] >= 0:
        start, end = layout.caption
        prefix.append(text[start:end].strip())
        text = (text[:start].rstrip() + "\n" + text[end:].lstrip()).strip()
    return "\n".join([*prefix, text])


def _row_groups(chunk: Mapping[str, object], section: str, limit: int) -> list[str] | None:
    """A long table as groups of data rows, each with the section, caption and header."""
    text = str(chunk.get("text", ""))
    layout = table_layout(text)
    if layout is None or len(layout.rows) < 2:
        return None
    head = [section.strip()] if section.strip() else []
    if layout.caption[0] >= 0:
        head.append(text[layout.caption[0] : layout.caption[1]].strip())
    head.append(text[layout.header[0] : layout.header[1]])
    base = len("\n".join(head))
    groups: list[list[str]] = [[]]
    size = base
    for start, end in layout.rows:
        row = text[start:end]
        if groups[-1] and size + 1 + len(row) > limit:
            groups.append([])
            size = base
        groups[-1].append(row)
        size += 1 + len(row)
    return ["\n".join([*head, *rows]) for rows in groups]


def embedding_units(
    paper_id: str, paper: Mapping[str, object], limit: int = ROW_GROUP_CHARS
) -> list[EmbeddingUnit]:
    """Every vector of one paper's chunks, in chunk order; blank and foreign chunks skipped."""
    chunks = [
        chunk
        for chunk in cast(Sequence[Mapping[str, object]], paper.get("chunks") or [])
        if chunk.get("paper_id") == paper_id
        and chunk.get("chunk_id")
        and str(chunk.get("text", "")).strip()
    ]
    sections = {
        row.chunk_id: row.section
        for row in annotate_sections(evidence_chunks(paper_id, chunks, None))
    }
    units: list[EmbeddingUnit] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        section = sections.get(chunk_id, "")
        kind = chunk_kind(chunk)
        text = embedding_text(chunk, paper, section)
        groups = (
            _row_groups(chunk, section, limit) if kind == "table" and len(text) > limit else None
        )
        if groups and len(groups) > 1:
            units.extend(
                EmbeddingUnit(f"{chunk_id}#t{k}", chunk_id, kind, group)
                for k, group in enumerate(groups)
            )
        else:
            units.append(EmbeddingUnit(chunk_id, chunk_id, kind, text))
    return units


def vector_revision(dedup_hash: object, chunk_count: object) -> str:
    """A paper's vector revision: its content, chunk count and the embedding format."""
    return f"{dedup_hash}:{chunk_count}:{EMBED_FORMAT}"


def vector_chunk_id(vector_id: str, payload: Mapping[str, object]) -> str:
    """The chunk a vector belongs to: its ``chunk_id`` metadata, else the id before ``#t``."""
    chunk_id = payload.get("chunk_id")
    return chunk_id if isinstance(chunk_id, str) and chunk_id else vector_id.split("#t", 1)[0]
