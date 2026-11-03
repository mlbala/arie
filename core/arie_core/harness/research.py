from __future__ import annotations

import re
from collections.abc import Sequence

import msgspec

from arie_core.harness.intent import AnswerPlan
from arie_core.harness.prefetch import reciprocal_rank_fusion
from arie_core.harness.reader import (
    EvidenceChunk,
    annotate_sections,
    eligible,
    is_untrusted_instruction,
    prose_sentences,
    query_terms,
    retrieve,
)
from arie_core.plugin_api import Ctx


class ReadingResult(msgspec.Struct, frozen=True):
    rows: tuple[EvidenceChunk, ...]
    actions: tuple[str, ...]
    # A wider candidate set from the same reads, in keyword/vector order, for the reranker.
    pool: tuple[EvidenceChunk, ...] = ()
    reranked: bool = False


# Vector hits merged per answer; keyword hits keep their own per-query cap of five.
SEMANTIC_DEPTH = 6
# Candidates offered to the reranker, and the vector hits that may join them.
POOL_SIZE = 24
POOL_SEMANTIC_DEPTH = 12
_EXCLUDED_SECTIONS = re.compile(r"references|acknowledg|author contributions", re.I)
_CREDITS = re.compile(r"equal contribution|listing order is random|work performed while", re.I)


def _usable(row: EvidenceChunk) -> bool:
    """Evidence a reader would quote: prose or a table, never credits or bibliography."""
    return (
        not _EXCLUDED_SECTIONS.search(row.section)
        and not _CREDITS.search(row.text)
        and bool(prose_sentences(row.text) or "|" in row.text)
    )


def _pool(
    plan: AnswerPlan,
    scoped: Sequence[EvidenceChunk],
    ordered: Sequence[EvidenceChunk],
    vector_ids: Sequence[str],
) -> tuple[EvidenceChunk, ...]:
    """Collect up to ``POOL_SIZE`` candidates: keyword/vector order, more vector hits, and
    for whole-paper questions the opening passage of every top-level section, so a
    reranker can choose across the paper rather than only among keyword matches."""
    by_id = {row.chunk_id: row for row in scoped}
    found = [*ordered, *(by_id[i] for i in vector_ids if i in by_id)]
    if plan.overview:
        openings: dict[str, EvidenceChunk] = {}
        for row in scoped:
            top = row.section.split(" / ")[0]
            if top and top not in openings and "appendix" not in top.lower() and _usable(row):
                openings[top] = row
        found.extend(openings.values())
    return tuple(dict.fromkeys(row for row in found if _usable(row)))[:POOL_SIZE]


def read_for_answer(
    ctx: Ctx,
    paper_id: str,
    plan: AnswerPlan,
    rows: Sequence[EvidenceChunk],
    semantic: Sequence[str] = (),
) -> ReadingResult:
    """Search, expand source sections and inspect gaps under three paper-bound tool calls.

    ``semantic`` ranks chunk IDs from the paper's vector index, best first. It is fused
    with keyword hits by RRF (k=60); IDs not in this paper's eligible rows are ignored.
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
    by_id = {row.chunk_id: row for row in eligible(plan.question, scoped)}
    vector_ids = [chunk_id for chunk_id in dict.fromkeys(semantic) if chunk_id in by_id]
    pool_vector_ids = vector_ids[:POOL_SEMANTIC_DEPTH]
    vector_ids = vector_ids[:SEMANTIC_DEPTH]
    if vector_ids:
        fused = reciprocal_rank_fusion([row.chunk_id for row in selected], vector_ids)
        found = {row.chunk_id: row for row in selected} | {i: by_id[i] for i in vector_ids}
        selected = [found[i] for i in sorted(fused, key=fused.__getitem__, reverse=True)]
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids)}
    complex_task = plan.task in {"comparison", "synthesis", "critique", "methodology"}
    if plan.depth != "detailed" and not complex_task and not plan.sections:
        return ReadingResult(
            tuple(row for row in selected[:10] if row.paper_id == paper_id),
            tuple(actions),
            _pool(plan, scoped, selected, pool_vector_ids),
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
                    r"method|model|architecture|approach|result|experiment|conclu|discuss|limitation",
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
        if row.chunk_id in vector_rank:
            score += 12 - 2 * vector_rank[row.chunk_id]
        if plan.follow_up and row.chunk_id not in prior_ids:
            score += 30
        if plan.follow_up and row.chunk_id in prior_ids:
            score -= 30
        if row.section in plan.sections:
            score += 20
        if plan.overview and re.search(
            r"method|architecture|model|approach", row.section.split(" / ")[0], re.I
        ):
            score += 18
        if plan.overview and re.search(
            r"result|experiment|conclu|discuss", row.section.split(" / ")[0], re.I
        ):
            score += 16
        return score

    ordered = [
        row for row in sorted(candidates, key=rank, reverse=True) if row.paper_id == paper_id
    ]
    return ReadingResult(
        tuple(select_rows(plan, ordered)),
        tuple(actions),
        _pool(plan, scoped, ordered, pool_vector_ids),
    )


def select_rows(plan: AnswerPlan, ordered: Sequence[EvidenceChunk]) -> list[EvidenceChunk]:
    """Take the best-first rows within 10 rows, 8,500 bytes and a per-section quota.

    The quota spreads evidence across the paper: an overview groups by top-level section
    (4 each), other answers by section (3 each), and a section the planner chose gets 5.
    Credits, bibliography, bare headings and layout fragments are skipped.
    """
    result: list[EvidenceChunk] = []
    used_bytes = 0
    per_section: dict[str, int] = {}
    for row in ordered:
        if row in result or len(result) >= 10 or not _usable(row):
            continue
        group = (
            row.section.split(" / ")[0]
            if plan.overview and row.section
            else row.section or f"page-{row.page_start}"
        )
        size = len(row.text.encode("utf-8"))
        quota = 5 if row.section in plan.sections else 4 if plan.overview else 3
        if per_section.get(group, 0) >= quota or used_bytes + size > 8500:
            continue
        result.append(row)
        used_bytes += size
        per_section[group] = per_section.get(group, 0) + 1
    return result
