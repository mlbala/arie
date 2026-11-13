from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

import msgspec

from arie_core.harness.answer import PROJECT_ROOT, AnswerResult, compose_answer
from arie_core.harness.figure_hint import FigureHint, figure_hint, figure_pages
from arie_core.harness.guardrails import InputDecision, inspect_question
from arie_core.harness.injection import is_untrusted
from arie_core.harness.intent import AnswerPlan, DialogueTurn, plan_answer
from arie_core.harness.planner import plan_research
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.rerank import RerankCheck, rerank_reading
from arie_core.harness.research import ReadingResult, read_for_answer
from arie_core.harness.session import ModelSession
from arie_core.harness.verifier import MIN_ENTAILMENT, EntailmentCheck
from arie_core.plugin_api import LLM, Ctx

logger = logging.getLogger(__name__)

# Paper-bound vector search: queries in, chunk IDs of the bound paper out, best first.
SemanticSearch = Callable[[Sequence[str]], Awaitable[Sequence[str]]]


class ResearchAnswer(msgspec.Struct, frozen=True):
    plan: AnswerPlan
    reading: ReadingResult
    answer: AnswerResult
    guardrail: InputDecision
    planner_status: str = "deterministic"
    # "See Figure N" for a strict result that did not answer (S24); None otherwise.
    figure_hint: FigureHint | None = None


async def answer_question(
    ctx: Ctx,
    paper_id: str,
    question: str,
    rows: Sequence[EvidenceChunk],
    llm: LLM | None = None,
    *,
    history: Sequence[DialogueTurn] = (),
    project_root: Path = PROJECT_ROOT,
    semantic: SemanticSearch | None = None,
    entail: EntailmentCheck | None = None,
    entail_floor: float = MIN_ENTAILMENT,
    rerank: RerankCheck | None = None,
    rerank_busy: Callable[[], bool] | None = None,
    section_summaries: Mapping[str, str] | None = None,
) -> ResearchAnswer:
    """Coordinate guard → plan → bounded paper reads → rerank → synthesis → verification.

    Scope and budgets are enforced by code, not delegated to model judgment. The
    caller supplies a 25-second deadline; provider failure retains exact source excerpts.
    ``semantic`` must already be bound to ``paper_id`` (I1); its failure or timeout
    falls back to keyword retrieval. ``entail`` is the Tier 3 entailment model for
    model-written claims; without it, claims rest on Tiers 1-2 and the model reviewer.
    ``rerank`` is the cross-encoder that picks the final passages from the keyword and
    vector candidates; its failure or timeout keeps the keyword/vector choice. While
    ``rerank_busy()`` is true (the answer worker still runs a call its caller abandoned) it is
    skipped, since the call would queue behind that one and delay the claim check; the trace
    records ``rerank_skipped_busy``.
    ``section_summaries`` are the paper guide's (S14); the planner sees them with the outline.
    """
    started = time.monotonic()
    session = ModelSession(ctx, llm)
    guard = inspect_question(question)
    if not guard.allowed:
        return ResearchAnswer(
            plan_answer(question[:2000]),
            ReadingResult((), ()),
            AnswerResult(
                [],
                "abstention",
                fallback_reason=guard.code,
                trace=session.trace("abstention", guard.code, time.monotonic() - started),
            ),
            guard,
        )
    scoped = [row for row in rows if row.paper_id == paper_id and not is_untrusted(row)]
    scoped_ids = {row.chunk_id for row in scoped}
    safe_history = tuple(
        DialogueTurn(
            decision.question[:700], tuple(s for s in turn.source_ids[:8] if s in scoped_ids)
        )
        for turn in history[-4:]
        if (decision := inspect_question(turn.question)).allowed
    )
    with session.stage("plan"):
        plan, planner_status = await plan_research(
            guard.question, safe_history, scoped, session, project_root, section_summaries
        )
    with session.stage("vector"):
        ranked = await _semantic_ranking(ctx, semantic, (plan.question, *plan.queries))
    with session.stage("read"):
        reading = read_for_answer(ctx, paper_id, plan, scoped, ranked)
    if rerank is not None and rerank_busy is not None and rerank_busy():
        session.rerank_skipped_busy = True
        logger.info("Reranking skipped: the answer worker is busy (trace %s)", ctx.trace_id)
    elif rerank is not None:
        with session.stage("rerank"):
            reading = await rerank_reading(ctx, paper_id, plan, reading, rerank)
    answer = await compose_answer(
        ctx,
        paper_id,
        plan.question,
        reading.rows,
        project_root=project_root,
        plan=plan,
        session=session,
        entail=entail,
        entail_floor=entail_floor,
    )
    # compose_answer traced the stages it ran; this trace adds the earlier ones and the total.
    trace = session.trace(answer.mode, answer.fallback_reason, time.monotonic() - started)
    hint = figure_hint(answer.mode, paper_id, reading.rows, figure_pages(scoped))
    if hint is not None:
        trace = msgspec.structs.replace(trace, figure_hint=hint.number)
    return ResearchAnswer(
        plan,
        reading,
        msgspec.structs.replace(answer, trace=trace),
        guard,
        planner_status,
        hint,
    )


async def _semantic_ranking(
    ctx: Ctx, semantic: SemanticSearch | None, queries: Sequence[str]
) -> tuple[str, ...]:
    """Query the vector index within 6 s, leaving at least 10 s for synthesis and review."""
    budget = min(6.0, ctx.deadline - time.monotonic() - 10.0)
    if semantic is None or budget <= 0.5:
        return ()
    try:
        async with asyncio.timeout(budget):
            return tuple(await semantic(tuple(dict.fromkeys(q for q in queries if q))[:4]))
    except Exception:
        logger.warning("Vector search unavailable; using keyword retrieval", exc_info=True)
        return ()
