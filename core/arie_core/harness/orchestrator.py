from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import msgspec

from arie_core.harness.answer import PROJECT_ROOT, AnswerResult, compose_answer
from arie_core.harness.guardrails import InputDecision, inspect_question
from arie_core.harness.intent import AnswerPlan, DialogueTurn, plan_answer
from arie_core.harness.planner import plan_research
from arie_core.harness.reader import EvidenceChunk, is_untrusted_instruction
from arie_core.harness.research import ReadingResult, read_for_answer
from arie_core.harness.session import ModelSession
from arie_core.plugin_api import LLM, Ctx


class ResearchAnswer(msgspec.Struct, frozen=True):
    plan: AnswerPlan
    reading: ReadingResult
    answer: AnswerResult
    guardrail: InputDecision
    planner_status: str = "deterministic"


async def answer_question(
    ctx: Ctx,
    paper_id: str,
    question: str,
    rows: Sequence[EvidenceChunk],
    llm: LLM | None = None,
    *,
    history: Sequence[DialogueTurn] = (),
    project_root: Path = PROJECT_ROOT,
) -> ResearchAnswer:
    """Coordinate guard → plan → bounded paper reads → synthesis → independent review.

    Scope and budgets are enforced by code, not delegated to model judgment. The
    caller supplies a 25-second deadline; provider failure retains exact source excerpts.
    """
    guard = inspect_question(question)
    if not guard.allowed:
        return ResearchAnswer(
            plan_answer(question[:2000]),
            ReadingResult((), ()),
            AnswerResult([], "abstention", fallback_reason=guard.code),
            guard,
        )
    scoped = [
        row for row in rows if row.paper_id == paper_id and not is_untrusted_instruction(row.text)
    ]
    scoped_ids = {row.chunk_id for row in scoped}
    safe_history = tuple(
        DialogueTurn(
            decision.question[:700], tuple(s for s in turn.source_ids[:8] if s in scoped_ids)
        )
        for turn in history[-4:]
        if (decision := inspect_question(turn.question)).allowed
    )
    session = ModelSession(ctx, llm)
    plan, planner_status = await plan_research(
        guard.question, safe_history, scoped, session, project_root
    )
    reading = read_for_answer(ctx, paper_id, plan, scoped)
    answer = await compose_answer(
        ctx,
        paper_id,
        plan.question,
        reading.rows,
        project_root=project_root,
        plan=plan,
        session=session,
    )
    return ResearchAnswer(plan, reading, answer, guard, planner_status)
