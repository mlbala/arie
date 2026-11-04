"""Paper guides (S14): one cited reading guide per indexed paper, built in the background.

Sync queues a paper when it reaches INDEXED. ``run_guides`` builds the queued guides one at a
time, waits while any answer is in flight, and keeps a paper that failed queued, so it is
retried on the next wake-up (the next indexed paper, or five minutes after a failure). Papers
indexed before guides, or queued before a restart, get theirs from ``make guides``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import msgspec
from arie_core.harness.guide import COMPLETE, GuideResult, PaperGuide, load_guide_rules, write_guide
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import GUIDE_SECONDS
from arie_core.plugin_api import LLM

from arie_api.runtime.base import Document, context, explain_context
from arie_api.runtime.sync import SyncJobs

logger = logging.getLogger(__name__)

# Seconds before a pass with failures is retried when nothing else wakes the builder.
GUIDE_RETRY_SECONDS = 300.0
NO_MODEL = "model_unavailable"


class PaperGuides(SyncJobs):
    """One guide per paper in ``arie_paper_guides``, keyed by pack revision and prompt version."""

    def _keyed_llm(self, provider: str | None = None) -> LLM | None:
        """The LLM plugin for ``provider`` (default: the writer's), or ``None`` without a key."""
        llm = cast(LLM, self.instance("llm", provider))
        return llm if getattr(llm, "api_key", None) else None

    @contextmanager
    def answering(self) -> Generator[None]:
        """Mark one answer in flight; the guide builder waits until none is."""
        self.answers_active += 1
        self.answers_quiet.clear()
        try:
            yield
        finally:
            self.answers_active -= 1
            if not self.answers_active:
                self.answers_quiet.set()

    async def wait_until_quiet(self) -> None:
        while self.answers_active:
            await self.answers_quiet.wait()

    def queue_guide(self, paper_id: str) -> None:
        self.guide_pending[paper_id] = None
        self.guide_wakeup.set()

    async def get_guide(self, paper_id: str) -> PaperGuide | None:
        """The stored guide of an indexed paper if it matches its pack and the guide prompt."""
        revision = await self.current_pack_revision(paper_id)
        if revision is None:
            return None
        _, version = await asyncio.to_thread(load_guide_rules, self.root)
        rows = await self.db.fetch(
            context(paper_id),
            "SELECT paper_id, revision, prompt_version, json FROM arie_paper_guides "
            "WHERE paper_id = ?",
            (paper_id,),
        )
        for row in rows:
            if (
                row["paper_id"] == paper_id
                and row["revision"] == revision
                and row["prompt_version"] == version
            ):
                return msgspec.json.decode(str(row["json"]).encode(), type=PaperGuide)
        return None

    async def guide_view(self, paper: Document) -> Document:
        """The guide route's body: the guide and one citation per marker it uses.

        A citation spans the whole chunk, as in explain mode; every marker of a stored guide
        passed the citation checks, so each is ``checked``.
        """
        paper_id = str(paper["id"])
        guide = await self.get_guide(paper_id)
        if guide is None:
            queued = paper_id in self.guide_pending
            error = self.guide_errors.get(paper_id) or self.guide_errors.get("*", "")
            return {
                "paper_id": paper_id,
                "status": "pending" if queued else "missing",
                "error": error if queued else "",
            }
        chunks = {
            str(row["chunk_id"]): row
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        }
        citations: list[Document] = []
        for marker in guide.markers:
            row = chunks.get(f"{paper_id}:{marker}")
            if row is not None:
                citations.append(
                    {
                        "marker": marker,
                        "chunk_id": row["chunk_id"],
                        "page": row["page_start"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        "checked": True,
                    }
                )
        return {
            "paper_id": paper_id,
            "status": "ready",
            "guide": msgspec.to_builtins(guide),
            "citations": citations,
        }

    async def missing_guides(self) -> list[str]:
        """Indexed papers without a current guide, oldest first."""
        rows = await self.db.fetch(
            context("guides"),
            "SELECT paper_id FROM arie_library WHERE status = 'INDEXED' ORDER BY published_at",
        )
        return [
            paper_id
            for row in rows
            if await self.get_guide(paper_id := str(row["paper_id"])) is None
        ]

    async def build_guide(self, paper_id: str) -> GuideResult | None:
        """Write and store the guide of one indexed paper; ``None`` if it is not indexed.

        The call gets its own context and budget (I3). ``paper_id`` is bound here and every
        chunk is re-filtered by it (I1). A failed guide is not stored.
        """
        paper = await self.get_paper(paper_id)
        if paper is None or paper.get("status") != "INDEXED":
            return None
        pack = await self.ensure_pack(paper)
        rows = [
            EvidenceChunk(**row)
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        ]
        context_tokens = self.config.context_tokens_for(self.llm_provider())
        result = await write_guide(
            explain_context(f"guide:{paper_id}", context_tokens, GUIDE_SECONDS),
            paper_id,
            pack,
            rows,
            self._keyed_llm(),
            context_tokens=context_tokens,
            pack_share=self.config.explain.max_pack_share,
            project_root=self.root,
        )
        if result.status == COMPLETE and result.guide is not None:
            guide = result.guide
            stored = msgspec.json.encode(guide).decode()
            await self.db.execute(
                context(paper_id),
                """INSERT INTO arie_paper_guides (paper_id, revision, prompt_version, json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET revision=excluded.revision,
                prompt_version=excluded.prompt_version, json=excluded.json""",
                (paper_id, guide.revision, guide.prompt_version, stored),
            )
        return result

    async def guide_pass(self) -> None:
        """Build every queued guide once, one at a time, each after the answers in flight."""
        if self.guide_pending and self._keyed_llm() is None:
            # No key: nothing can be built, so try again later instead of failing each paper.
            self.guide_errors["*"] = NO_MODEL
            return
        self.guide_errors.pop("*", None)
        for paper_id in list(self.guide_pending):
            await self.wait_until_quiet()
            try:
                result = await self.build_guide(paper_id)
            except Exception as exc:
                logger.warning("Guide build failed for %s", paper_id, exc_info=True)
                self.guide_errors[paper_id] = str(exc) or type(exc).__name__
                continue
            if result is not None and result.status != COMPLETE:
                self.guide_errors[paper_id] = result.fallback_reason
                continue
            self.guide_pending.pop(paper_id, None)
            self.guide_errors.pop(paper_id, None)

    async def run_guides(self) -> None:
        """Wake when sync queues a paper; retry failures on the next wake-up."""
        while True:
            self.guide_wakeup.clear()
            await self.guide_pass()
            try:
                timeout = GUIDE_RETRY_SECONDS if self.guide_pending else None
                await asyncio.wait_for(self.guide_wakeup.wait(), timeout)
            except TimeoutError:
                pass
