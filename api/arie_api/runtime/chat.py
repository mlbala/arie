"""Chat history and the answer path shared by chat and per-paper evaluation."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

from arie_core.harness.intent import DialogueTurn
from arie_core.harness.orchestrator import ResearchAnswer, answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.plugin_api import (
    LLM,
)
from arie_core.registry.sync import months_ago as months_ago

from arie_api.runtime.base import (
    CHAT_PAPERS,
    CHAT_TURNS,
    CallContext,
    Document,
    context,
)
from arie_api.runtime.sync import SyncJobs


class ChatHistory(SyncJobs):
    """Chat history and the answer path shared by chat and per-paper evaluation."""

    async def save_chat_turn(
        self,
        paper_id: str,
        question: str,
        events: Sequence[Document],
        turn_id: str | None = None,
    ) -> None:
        """Store one answered turn as the exact events streamed, then enforce both limits.

        Keeps the last ``CHAT_TURNS`` turns of each paper and only the ``CHAT_PAPERS`` papers
        asked about most recently. Saved answers are display history; they never become
        evidence for later answers (I2).
        """
        ctx = context(f"chat:{paper_id}", 5.0)
        await self.db.execute(
            ctx,
            """INSERT INTO arie_chat_turns (turn_id, paper_id, question, events_json, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                turn_id or uuid.uuid4().hex,
                paper_id,
                question[:2000],
                json.dumps(list(events)),
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.db.execute(
            ctx,
            """DELETE FROM arie_chat_turns WHERE paper_id = ? AND turn_id NOT IN (
            SELECT turn_id FROM arie_chat_turns WHERE paper_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?)""",
            (paper_id, paper_id, CHAT_TURNS),
        )
        await self.db.execute(
            ctx,
            """DELETE FROM arie_chat_turns WHERE paper_id NOT IN (
            SELECT paper_id FROM arie_chat_turns GROUP BY paper_id
            ORDER BY MAX(created_at) DESC LIMIT ?)""",
            (CHAT_PAPERS,),
        )

    async def chat_history(self, paper_id: str) -> list[Document]:
        """Return a paper's saved turns, oldest first, with any feedback rating."""
        rows = await self.db.fetch(
            context(f"chat:{paper_id}", 5.0),
            """SELECT t.turn_id, t.question, t.events_json, f.rating FROM arie_chat_turns t
            LEFT JOIN arie_feedback f ON f.turn_id = t.turn_id AND f.paper_id = t.paper_id
            WHERE t.paper_id = ? ORDER BY t.created_at DESC, t.rowid DESC LIMIT ?""",
            (paper_id, CHAT_TURNS),
        )
        return [
            {
                "turn_id": str(row["turn_id"]),
                "question": str(row["question"]),
                "events": json.loads(str(row["events_json"])),
                "rating": row["rating"],
            }
            for row in reversed(rows)
        ]

    async def answer_paper_question(
        self,
        ctx: CallContext,
        paper: Document,
        question: str,
        history: Sequence[DialogueTurn] = (),
        *,
        rerank: bool = True,
    ) -> ResearchAnswer:
        """Answer one question about one indexed paper with every configured model.

        The chat route and the per-paper evaluation both call this, so an evaluation
        measures exactly what users get. ``rerank=False`` skips the reranker, for
        comparisons. ``paper["id"]`` is bound into every read and search (I1).
        """
        paper_id = str(paper["id"])
        rows = [
            EvidenceChunk(**row)
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        ]
        llm: LLM | None = cast(LLM, self.instance("llm"))
        if not getattr(llm, "api_key", None):
            llm = None

        async def semantic(queries: Sequence[str]) -> list[str]:
            return await self.semantic_search(paper_id, queries)

        return await answer_question(
            ctx,
            paper_id,
            question,
            rows,
            llm,
            history=history,
            semantic=semantic,
            entail=self.entailment_scores if self.models.get("entailment") else None,
            entail_floor=self.entail_floor,
            rerank=self.rerank_scores if rerank and self.models.get("reranker") else None,
        )

    async def entailment_scores(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Tier 3 scores for (quote, claim) pairs; raises while the model is unavailable."""
        model = self.models.get("entailment")
        if model is None:
            raise RuntimeError("Entailment is off")
        return await model.scores(list(pairs), count=len(pairs))

    async def rerank_scores(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Relevance of each passage to ``query``; raises while the model is unavailable."""
        model = self.models.get("reranker")
        if model is None:
            raise RuntimeError("Reranking is off")
        return await model.scores(query, list(passages), count=len(passages))
