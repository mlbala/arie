from __future__ import annotations

import re
from collections.abc import Sequence

import msgspec

from arie_core.harness.intent import AnswerPlan
from arie_core.harness.reader import (
    EvidenceChunk,
    annotate_sections,
    is_untrusted_instruction,
    prose_sentences,
    query_terms,
    retrieve,
)
from arie_core.plugin_api import Ctx


class ReadingResult(msgspec.Struct, frozen=True):
    rows: tuple[EvidenceChunk, ...]
    actions: tuple[str, ...]


def read_for_answer(
    ctx: Ctx, paper_id: str, plan: AnswerPlan, rows: Sequence[EvidenceChunk]
) -> ReadingResult:
    """Search, expand source sections and inspect gaps under three paper-bound tool calls.

    History IDs are untrusted locators. All reads are post-filtered to the bound paper;
    no previous answer prose or other paper can enter the evidence set.
    """
    scoped = annotate_sections(
        [row for row in rows if row.paper_id == paper_id and not is_untrusted_instruction(row.text)]
    )
    actions: list[str] = []

    def record(name: str) -> None:
        ctx.budget.record_tool_call(name)
        actions.append(name)

    record("search_paper")
    selected: list[EvidenceChunk] = []
    for query in (plan.question, *plan.queries):
        for row in retrieve(paper_id, query, scoped):
            if row.paper_id == paper_id and row not in selected:
                selected.append(row)
    complex_task = plan.task in {"comparison", "synthesis", "critique", "methodology"}
    if plan.depth != "detailed" and not complex_task and not plan.sections:
        return ReadingResult(
            tuple(row for row in selected[:10] if row.paper_id == paper_id), tuple(actions)
        )

    record("read_section")
    prior_ids = set(plan.previous_source_ids)
    anchors = [row for row in scoped if row.chunk_id in prior_ids] or selected
    terms = query_terms(" ".join((plan.question, *plan.queries)))
    sections = {row.section for row in anchors if row.section}
    candidates = list(selected)
    for row in scoped:
        if row.paper_id != paper_id or not row.text.strip():
            continue
        if row.section in plan.sections:
            relevant = True
        elif plan.overview:
            relevant = bool(
                re.search(
                    r"method|model|architecture|attention|result|experiment|conclu|discuss|limitation",
                    row.section,
                    re.I,
                )
            )
        else:
            relevant = row.section in sections or len(terms & query_terms(row.text)) >= 2
        if relevant and row not in candidates:
            candidates.append(row)

    if plan.overview:
        candidates = [
            row
            for row in candidates
            if prose_sentences(row.text)
            and not re.match(r"^\s*[#*]*(?:Table|Figure)\s+\d", row.text, re.I)
        ]
    new_rows = [row for row in candidates if row.chunk_id not in prior_ids]
    if len(new_rows) < 4 or len({row.section.split(" / ")[0] for row in candidates}) < 2:
        record("get_page")
        pages = {row.page_start for row in anchors}
        for row in scoped:
            if row.paper_id == paper_id and row.page_start in pages and row not in candidates:
                candidates.append(row)

    def rank(row: EvidenceChunk) -> int:
        score = 5 * len(terms & query_terms(row.text))
        if plan.follow_up and row.chunk_id not in prior_ids:
            score += 30
        if plan.follow_up and row.chunk_id in prior_ids:
            score -= 30
        if row.section in plan.sections:
            score += 20
        if plan.overview and re.search(
            r"method|architecture|model|attention", row.section.split(" / ")[0], re.I
        ):
            score += 18
        if plan.overview and re.search(
            r"result|experiment|conclu|discuss", row.section.split(" / ")[0], re.I
        ):
            score += 16
        return score

    result: list[EvidenceChunk] = []
    used_bytes = 0
    per_section: dict[str, int] = {}
    for row in sorted(candidates, key=rank, reverse=True):
        if row.paper_id != paper_id or row in result or len(result) >= 10:
            continue
        # Exclude contributor notes and bibliography from ordinary research explanations.
        if re.search(r"references|acknowledg|author contributions", row.section, re.I):
            continue
        if re.search(
            r"equal contribution|listing order is random|work performed while", row.text, re.I
        ):
            continue
        group = (
            row.section.split(" / ")[0]
            if plan.overview and row.section
            else row.section or f"page-{row.page_start}"
        )
        size = len(row.text.encode("utf-8"))
        if per_section.get(group, 0) >= (4 if plan.overview else 3) or used_bytes + size > 8500:
            continue
        result.append(row)
        used_bytes += size
        per_section[group] = per_section.get(group, 0) + 1
    return ReadingResult(tuple(result), tuple(actions))
