"""Stored paper packs: built when a paper is indexed, and lazily for older papers."""

from __future__ import annotations

import asyncio
import json
from typing import cast

from arie_core.harness.reader import EvidenceChunk
from arie_core.pipeline.pack import PaperPack, build_pack, marker_numbers, pack_revision

from arie_api.runtime.base import Document, context
from arie_api.runtime.library import LibraryStore


def paper_revision(paper: Document) -> str:
    """The vector index's revision of a paper, ``<dedup_hash>:<chunk_count>``."""
    return f"{paper.get('dedup_hash', '')}:{len(paper.get('chunks', []))}"


class PaperPacks(LibraryStore):
    """One pack per indexed paper in ``arie_paper_packs``, keyed by paper revision."""

    async def ensure_pack(self, paper: Document) -> PaperPack:
        """Return the stored pack for this paper revision, building and storing it if missing."""
        paper_id = str(paper["id"])
        revision = paper_revision(paper)
        stored = await self._stored_pack(paper_id, pack_revision(revision))
        if stored is not None:
            return stored
        rows = [
            EvidenceChunk(**row)
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        ]
        pack = await asyncio.to_thread(
            build_pack,
            paper_id,
            str(paper.get("title", "")),
            str(paper.get("abstract", "")),
            revision,
            rows,
        )
        await self.db.execute(
            context(paper_id),
            """INSERT INTO arie_paper_packs (paper_id, revision, text, tokens)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET revision=excluded.revision,
            text=excluded.text, tokens=excluded.tokens""",
            (paper_id, pack.revision, pack.text, pack.token_estimate),
        )
        return pack

    async def get_pack(self, paper_id: str) -> PaperPack | None:
        """The pack of an indexed paper, built on first use for papers indexed before packs.

        A current pack is read without loading the paper's document.
        """
        revision = await self.current_pack_revision(paper_id)
        if revision is None:
            return None
        stored = await self._stored_pack(paper_id, revision)
        if stored is not None:
            return stored
        paper = await self.get_paper(paper_id)
        return await self.ensure_pack(paper) if paper is not None else None

    async def current_pack_revision(self, paper_id: str) -> str | None:
        """The pack revision an indexed paper's pack should have; ``None`` if not indexed.

        Read from the library summary, without loading the paper's document.
        """
        rows = await self.db.fetch(
            context(paper_id),
            "SELECT paper_id, status, summary_json FROM arie_library WHERE paper_id = ?",
            (paper_id,),
        )
        rows = [row for row in rows if row["paper_id"] == paper_id]
        if not rows or rows[0]["status"] != "INDEXED":
            return None
        summary = cast(Document, json.loads(str(rows[0]["summary_json"])))
        return pack_revision(f"{summary.get('revision', '')}:{summary.get('chunk_count', 0)}")

    async def _stored_pack(self, paper_id: str, revision: str) -> PaperPack | None:
        rows = await self.db.fetch(
            context(paper_id),
            "SELECT paper_id, revision, text, tokens FROM arie_paper_packs WHERE paper_id = ?",
            (paper_id,),
        )
        for row in rows:
            if row["paper_id"] == paper_id and row["revision"] == revision:
                text = str(row["text"])
                return PaperPack(text, marker_numbers(text), int(str(row["tokens"])), revision)
        return None
