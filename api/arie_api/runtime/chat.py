"""Chat history and the answer path shared by chat and per-paper evaluation."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime
from typing import cast

from arie_core.harness.explain import HISTORY_TURNS, ExplainResult, ExplainTurn, explain_answer
from arie_core.harness.feedback import implicit_negative
from arie_core.harness.intent import DialogueTurn
from arie_core.harness.judge import (
    PairwiseJudgement,
    SupportJudgement,
    judge_pairwise,
    judge_support,
)
from arie_core.harness.orchestrator import ResearchAnswer, answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import JUDGE_SECONDS
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
    explain_context,
)
from arie_api.runtime.guides import PaperGuides

logger = logging.getLogger(__name__)


def saved_answer_text(events: Sequence[Document]) -> str:
    """The prose of a saved turn: an explain turn's text, or a strict turn's claim texts."""
    deltas = [str(event["data"].get("text", "")) for event in events if event["event"] == "delta"]
    if deltas:
        return "".join(deltas)
    claims = [
        str(event["data"].get("claim", {}).get("text", ""))
        for event in events
        if event["event"] == "claim"
    ]
    return " ".join(text for text in claims if text)


class ChatHistory(PaperGuides):
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
        try:
            await self._note_follow_up(paper_id, question)
        except Exception:
            # A missed signal must not lose the turn.
            logger.warning("Could not read the follow-up signal for %s", paper_id, exc_info=True)
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

    async def _note_follow_up(self, paper_id: str, question: str) -> None:
        """Mark the paper's latest turn ``implicit negative`` if ``question`` reads as one.

        Called before the new turn is saved, so the latest saved turn is the one the reader is
        reacting to. The signal is kept even if the turn is rated later; readers of it leave out
        turns with an explicit rating (S15, SPEC 10).
        """
        ctx = context(f"chat:{paper_id}", 5.0)
        rows = await self.db.fetch(
            ctx,
            """SELECT turn_id, question FROM arie_chat_turns WHERE paper_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (paper_id,),
        )
        signal = implicit_negative(str(rows[0]["question"]), question) if rows else None
        if signal is None:
            return
        await self.db.execute(
            ctx,
            """INSERT INTO arie_implicit_feedback VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET signal = excluded.signal,
            follow_up = excluded.follow_up, created_at = excluded.created_at""",
            (
                str(rows[0]["turn_id"]),
                paper_id,
                str(rows[0]["question"])[:2000],
                signal,
                question[:500],
                datetime.now(UTC).isoformat(),
            ),
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
        guides: bool = True,
    ) -> ResearchAnswer:
        """Answer one question about one indexed paper with every configured model.

        The chat route and the per-paper evaluation both call this, so an evaluation
        measures exactly what users get. ``rerank=False`` skips the reranker and
        ``guides=False`` the guide's section summaries, for comparisons. ``paper["id"]`` is
        bound into every read and search (I1).
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

        guide = await self.get_guide(paper_id) if guides else None
        with self.answering():
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
                section_summaries=guide.summaries() if guide is not None else None,
            )

    async def explain_turns(self, paper_id: str) -> list[ExplainTurn]:
        """The last ``HISTORY_TURNS`` saved turns of this paper, as context for an explanation.

        Read from the server's own history, never from the request, so a client cannot write
        ARIE's side of the conversation. They are context for the question, not evidence.
        """
        saved = await self.chat_history(paper_id)
        return [
            ExplainTurn(str(turn["question"]), saved_answer_text(turn["events"]))
            for turn in saved[-HISTORY_TURNS:]
        ]

    async def explain_paper_question(
        self,
        trace_id: str,
        paper: Document,
        question: str,
        turns: Sequence[ExplainTurn] = (),
    ) -> AsyncGenerator[str | ExplainResult]:
        """Explain one indexed paper: text deltas as ``str``, then one ``ExplainResult``.

        Chat and the per-paper evaluation both call this. The request gets its own budget
        (``explain_context``): the model's whole context window and at least 90 seconds. With no
        LLM key the result is ``explain_failed`` with reason ``model_unavailable``.
        ``paper["id"]`` is bound here, never sent to the model (I1).
        """
        paper_id = str(paper["id"])
        context_tokens = self.config.context_tokens_for(self.llm_provider())
        ctx = explain_context(trace_id, context_tokens)
        rows = [
            EvidenceChunk(**row)
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        ]
        llm: LLM | None = cast(LLM, self.instance("llm"))
        if not getattr(llm, "api_key", None):
            llm = None
        pack = await self.ensure_pack(paper)
        stream = explain_answer(
            ctx,
            paper_id,
            question,
            pack,
            rows,
            llm,
            context_tokens=context_tokens,
            pack_share=self.config.explain.max_pack_share,
            turns=turns,
        )
        with self.answering():
            async with aclosing(stream) as items:
                async for item in items:
                    yield item

    async def judge_explanation(self, trace_id: str, result: ExplainResult) -> SupportJudgement:
        """Judge every cited sentence of an explain answer in one call on the writer's provider.

        The call sends the pack the writer saw, so the provider serves it from its cache. It
        gets its own context and budget (I3), separate from the writer's, and never raises on
        a model failure: the answer is judged ``failed`` instead. Chat and the per-paper
        evaluation both call this; ``judge.chat`` is the chat route's switch, not checked here.
        """
        context_tokens = self.config.context_tokens_for(self.llm_provider())
        ctx = explain_context(f"{trace_id}:judge", context_tokens, JUDGE_SECONDS)
        return await judge_support(
            ctx,
            self._keyed_llm(),
            result.pack_text,
            result.checks,
            context_tokens=context_tokens,
        )

    async def judge_explain_vs_strict(
        self,
        trace_id: str,
        question: str,
        key_facts: Sequence[str],
        explain_text: str,
        strict_text: str,
    ) -> PairwiseJudgement:
        """Judge an explanation against a strict answer, once each way round (evaluations).

        Runs on ``eval_judge_provider()``: another provider than the writer's when its key is
        set. The paper is not sent; the key facts are the reference.
        """
        provider = self.eval_judge_provider()
        ctx = explain_context(
            f"{trace_id}:pairwise", self.config.context_tokens_for(provider), JUDGE_SECONDS
        )
        return await judge_pairwise(
            ctx,
            self._keyed_llm(provider),
            question,
            key_facts,
            explain_text,
            strict_text,
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
