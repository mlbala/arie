from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import msgspec

from arie_core.harness.answer import load_playbook
from arie_core.harness.guardrails import inspect_question
from arie_core.harness.intent import AnswerPlan, DialogueTurn, plan_answer
from arie_core.harness.reader import EvidenceChunk, annotate_sections
from arie_core.harness.session import ModelSession, failure_reason


class PlannerDraft(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """An allowlisted reading plan, never an answer or executable tool arguments."""

    task: Literal[
        "overview",
        "definition",
        "mechanism",
        "comparison",
        "results",
        "methodology",
        "limitations",
        "critique",
        "lookup",
        "synthesis",
        "answer",
    ]
    depth: Literal["brief", "standard", "detailed"]
    queries: tuple[str, ...]
    aspects: tuple[str, ...]
    sections: tuple[str, ...] = ()
    resolved_question: str = ""
    follow_up: bool = False


def _task(question: str, overview: bool) -> str:
    if overview:
        return "overview"
    routes = (
        ("comparison", r"compar|differ|versus|\bvs\b|trade.off|relative to"),
        ("limitations", r"limitation|weakness|caveat|failure|shortcoming"),
        ("results", r"result|accuracy|performance|finding|benchmark|how (?:much|well)"),
        ("methodology", r"dataset|training|experiment|protocol|setup|reproduc"),
        ("critique", r"evidence|support|justify|convinc|strength|assumption"),
        ("mechanism", r"\bhow\b|\bwhy\b|mechanism|step.by.step|algorithm"),
        ("definition", r"\bwhat (?:is|are)\b|defin|meaning|means"),
        ("synthesis", r"connect|relat|synthesi|combine"),
    )
    return next((name for name, pattern in routes if re.search(pattern, question, re.I)), "answer")


async def plan_research(
    question: str,
    history: Sequence[DialogueTurn],
    rows: Sequence[EvidenceChunk],
    session: ModelSession,
    project_root: Path,
) -> tuple[AnswerPlan, str]:
    """Plan arbitrary paper tasks; malformed plans fall back without widening paper scope.

    The model sees questions and a bounded outline, never previous answer prose,
    paper IDs or other-paper rows. It cannot change tools, limits or citation policy.
    """
    basic = plan_answer(question, history)
    basic = msgspec.structs.replace(
        basic,
        task=_task(basic.question, basic.overview),
        queries=(basic.question[:300],),
        user_question=question,
    )
    if session.llm is None or not rows:
        return basic, "deterministic"
    outline = tuple(dict.fromkeys(row.section for row in annotate_sections(rows) if row.section))
    # Spread a long outline across the paper rather than spending the budget on its introduction.
    step = max(1, len(outline) // 18)
    outline = tuple(section[:100] for section in outline[::step][:18])
    try:
        playbook = await asyncio.to_thread(load_playbook, project_root)
        raw = await session.call(
            playbook["plan_system"],
            {
                "question": question,
                "prior_questions": [turn.question[:700] for turn in history[-3:]],
                "outline": outline,
            },
            128,
            4.0,
        )
        draft = msgspec.json.decode(msgspec.json.encode(raw), type=PlannerDraft)
        if not 1 <= len(draft.queries) <= 3 or not 1 <= len(draft.aspects) <= 3:
            raise ValueError("Invalid plan size")
        if any(not value.strip() or len(value) > 180 for value in (*draft.queries, *draft.aspects)):
            raise ValueError("Invalid plan focus")
        if len(draft.sections) > 3 or any(value not in outline for value in draft.sections):
            raise ValueError("Unknown section")
        resolved = draft.resolved_question.strip() or basic.question
        if len(resolved) > 700 or not inspect_question(resolved).allowed:
            raise ValueError("Invalid resolved question")
        explicit_depth = bool(
            re.search(r"brief|concise|short|detail|expand|elaborate", question, re.I)
        )
        follow_up = bool(history) and (basic.follow_up or draft.follow_up)
        sources = tuple(
            dict.fromkeys(source for turn in history[-4:] for source in turn.source_ids)
        )
        return msgspec.structs.replace(
            basic,
            question=resolved,
            depth=basic.depth if explicit_depth else draft.depth,
            overview=draft.task == "overview",
            task=draft.task,
            queries=draft.queries,
            aspects=draft.aspects,
            sections=draft.sections,
            follow_up=follow_up,
            previous_source_ids=sources[:24] if follow_up else (),
            topic_question=resolved[:700],
        ), "model"
    except Exception as exc:
        reason = failure_reason(exc)
        # A transport/auth failure should not trigger further failing calls in this request.
        if reason.startswith("provider_http_") or reason in {"TimeoutError", "ConnectError"}:
            session.llm = None
            session.unavailable_reason = reason
        return basic, reason
