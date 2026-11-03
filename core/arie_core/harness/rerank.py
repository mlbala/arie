from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

import msgspec

from arie_core.harness.intent import AnswerPlan
from arie_core.harness.research import ReadingResult, select_rows
from arie_core.plugin_api import Ctx

logger = logging.getLogger(__name__)

# Cross-encoder: one question and passages in, one relevance score per passage out.
RerankCheck = Callable[[str, Sequence[str]], Awaitable[Sequence[float]]]
# Seconds that must remain after reranking for the writer, Tier 3 and the reviewer.
ANSWER_RESERVE = 14.0


def rerank_query(plan: AnswerPlan) -> str:
    """The resolved question plus the planner's search focus, as one reranker query."""
    parts = dict.fromkeys(part.strip() for part in (plan.question, *plan.queries) if part.strip())
    return "; ".join(parts)[:600]


async def rerank_reading(
    ctx: Ctx, paper_id: str, plan: AnswerPlan, reading: ReadingResult, rerank: RerankCheck
) -> ReadingResult:
    """Reorder the candidate pool by cross-encoder relevance, then reselect the rows.

    The reranker replaces keyword and section heuristics for the final choice; the
    per-section quota and byte cap still apply, so an overview stays spread across the
    paper. Follow-ups that ask for more put passages already cited after new ones.
    Every row is re-checked against ``paper_id`` (I1). Not a tool call: K is unchanged.
    Failure modes: a reranker error, malformed scores or a timeout (at most 3 s, and
    only while ``ANSWER_RESERVE`` seconds remain) keep the keyword/vector reading.
    """
    pool = [row for row in reading.pool if row.paper_id == paper_id]
    budget = min(3.0, ctx.deadline - time.monotonic() - ANSWER_RESERVE)
    if len(pool) < 2 or budget <= 0.3:
        return reading
    try:
        async with asyncio.timeout(budget):
            scores = [float(s) for s in await rerank(rerank_query(plan), [r.text for r in pool])]
        if len(scores) != len(pool):
            raise ValueError("Reranker returned the wrong number of scores")
    except Exception:
        logger.warning("Reranker unavailable; using keyword and vector order", exc_info=True)
        return reading
    prior: set[str] = set(plan.previous_source_ids) if plan.follow_up else set()
    order = sorted(range(len(pool)), key=lambda i: (pool[i].chunk_id in prior, -scores[i], i))
    rows = select_rows(plan, [pool[i] for i in order])
    if not rows:
        return reading
    return msgspec.structs.replace(reading, rows=tuple(rows), reranked=True)


__all__ = ["ANSWER_RESERVE", "RerankCheck", "rerank_query", "rerank_reading"]
