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


# Tasks answered with a multi-paragraph explanation unless the user asks for brevity.
EXPLANATORY = frozenset(
    {"mechanism", "methodology", "results", "limitations", "critique", "comparison", "synthesis"}
)


# Model labels too broad to overrule a recognized overview wording.
BROAD = frozenset({"answer", "overview", "synthesis"})


def _focus(values: Sequence[str]) -> tuple[str, ...]:
    """Keep up to three distinct, non-empty plan entries of at most 180 characters."""
    kept = (value.strip() for value in values)
    return tuple(dict.fromkeys(value for value in kept if value and len(value) <= 180))[:3]


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
    sections = tuple(dict.fromkeys(row.section for row in annotate_sections(rows) if row.section))
    # Every top-level section first (a conclusion often holds the limitations), then
    # subsections spread across the paper, in reading order, within 18 entries.
    top = {section for section in sections if " / " not in section}
    rest = [section for section in sections if section not in top]
    step = max(1, len(rest) // max(1, 18 - len(top)))
    keep = top | set(rest[::step][: max(0, 18 - len(top))])
    outline = tuple(section[:100] for section in sections if section in keep)[:18]
    try:
        playbook = await asyncio.to_thread(load_playbook, project_root)
        raw = await session.call(
            playbook["plan_system"],
            {
                "question": question,
                "prior_questions": [turn.question[:700] for turn in history[-3:]],
                "outline": outline,
            },
            160,
            4.0,
        )
        draft = msgspec.json.decode(msgspec.json.encode(raw), type=PlannerDraft)
        # Over-long or partly unusable plans are trimmed, not discarded: one extra section
        # name must not drop the whole plan back to keyword rules. Unknown section names
        # are removed, so a plan can only narrow reading within the bound paper.
        queries = _focus(draft.queries)
        aspects = _focus(draft.aspects)
        if not queries or not aspects:
            raise ValueError("Invalid plan focus")
        planned_sections = tuple(dict.fromkeys(s for s in draft.sections if s in outline))[:3]
        resolved = draft.resolved_question.strip() or basic.question
        if len(resolved) > 700 or not inspect_question(resolved).allowed:
            raise ValueError("Invalid resolved question")
        explicit_depth = bool(
            re.search(r"brief|concise|short|detail|expand|elaborate", question, re.I)
        )
        # The model decides what kind of question this is. Keyword rules only fill in when
        # the model's label is broad: a recognized overview wording stays an overview unless
        # the model names a specific task, and the model's "overview" always wins.
        overview = draft.task == "overview" or (basic.overview and draft.task in BROAD)
        depth = basic.depth if explicit_depth else draft.depth
        generic = draft.task in {"answer", "overview"}
        task = "overview" if overview else basic.task if generic else draft.task
        if not explicit_depth and (overview or task in EXPLANATORY):
            depth = "detailed"  # Explanations cover substance; lookups and definitions stay short.
        elif depth == "brief" and len(aspects) > 2 and not explicit_depth:
            depth = "standard"  # Two brief claims cannot cover three required aspects.
        # An explicitly brief answer keeps its length and drops the least essential aspect.
        aspects = aspects[:2] if depth == "brief" else aspects
        follow_up = bool(history) and (basic.follow_up or draft.follow_up)
        sources = tuple(
            dict.fromkeys(source for turn in history[-4:] for source in turn.source_ids)
        )
        return msgspec.structs.replace(
            basic,
            question=resolved,
            depth=depth,
            overview=overview,
            task=task,
            queries=queries,
            aspects=aspects,
            sections=planned_sections,
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
