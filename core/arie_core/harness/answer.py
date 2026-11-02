from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import msgspec
import yaml

from arie_core.harness.intent import AnswerPlan, plan_answer
from arie_core.harness.reader import (
    EvidenceChunk,
    Excerpt,
    is_untrusted_instruction,
    locate_quote,
    prose_sentences,
    query_terms,
    select_excerpts,
)
from arie_core.harness.session import ModelSession, failure_reason
from arie_core.plugin_api import LLM, Ctx

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AnswerResult(msgspec.Struct, frozen=True):
    excerpts: list[Excerpt]
    mode: str
    model_calls: int = 0
    reviewed_claims: int = 0
    rejected_claims: int = 0
    input_bytes: int = 0
    output_tokens: int = 0
    fallback_reason: str = ""
    answer_relevant: bool = False
    covered_aspects: int = 0
    required_aspects: int = 0


def load_playbook(project_root: Path) -> dict[str, str]:
    """Load the editable role prompts; fail closed on missing or invalid prompts."""
    loaded: object = yaml.safe_load((project_root / "prompts" / "qa.v1.yaml").read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("The QA playbook must be a mapping")
    data = cast(Mapping[str, object], loaded)
    result: dict[str, str] = {}
    for name in ("plan_system", "answer_system", "review_system"):
        prompt = data.get(name)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"The QA playbook requires {name}")
        result[name] = prompt
    return result


def answer_prompts(project_root: Path) -> tuple[str, str]:
    data = load_playbook(project_root)
    return data["answer_system"], data["review_system"]


def _source_windows(
    rows: Sequence[EvidenceChunk], plan: AnswerPlan, *, max_bytes: int = 3500
) -> list[EvidenceChunk]:
    """Catalog exact, bounded evidence units; source IDs replace copied quotes in model output."""
    groups: list[list[EvidenceChunk]] = []
    terms = query_terms(plan.question)
    for row in rows[:10]:
        sentences = prose_sentences(row.text)
        if (
            not sentences
            and 15 <= len(row.text.strip().encode("utf-8")) <= 650
            and not row.text.lstrip().startswith("#")
            and ("|" in row.text or "=" in row.text)
        ):
            sentences = [row.text.strip()]

        def score(text: str) -> int:
            purpose = bool(
                re.search(r"\b(?:we propose|we introduce|this paper|this work)\b", text, re.I)
            )
            return len(terms & query_terms(text)) * 5 + (20 if plan.overview and purpose else 0)

        units: list[EvidenceChunk] = []
        for sentence in sorted(sentences, key=score, reverse=True)[:4]:
            start = row.text.find(sentence)
            if start < 0:
                continue
            units.append(
                msgspec.structs.replace(
                    row,
                    text=sentence,
                    char_start=row.char_start + start,
                    char_end=row.char_start + start + len(sentence),
                )
            )
        groups.append(units)
    windows: list[EvidenceChunk] = []
    used = 0
    limit = max(1000, min(max_bytes, 6500 - 2 * len(plan.question.encode("utf-8"))))
    for offset in range(4):
        for units in groups:
            if offset >= len(units):
                continue
            unit = units[offset]
            size = len(unit.text.encode("utf-8"))
            if used + size > limit or any(other.text == unit.text for other in windows):
                continue
            windows.append(unit)
            used += size
            if len(windows) == (24 if plan.depth == "detailed" else 16):
                return windows
    return windows


def preserves_anchors(text: str, quote: str) -> bool:
    """Reject changed numbers, lost qualifications and new absolutes before model review."""

    def numbers(value: str) -> set[str]:
        return set(re.findall(r"\d+(?:\.\d+)?", value.replace(",", "")))

    if not numbers(text) <= numbers(quote):
        return False
    groups = (
        r"\b(?:not|no|never|without|cannot|neither|fails?|unable|lacks?|removes?|dispensing)\b|n't",
        r"\b(?:may|might|could|suggests?|appears?|potentially|preliminary)\b",
        r"\b(?:only|limited|small|restricted)\b",
    )
    if any(re.search(group, quote, re.I) and not re.search(group, text, re.I) for group in groups):
        return False
    absolutes = r"\b(?:always|never|every|guarantees?|proves?|perfect|all datasets)\b"
    return not (re.search(absolutes, text, re.I) and not re.search(absolutes, quote, re.I))


async def compose_answer(
    ctx: Ctx,
    paper_id: str,
    question: str,
    rows: Sequence[EvidenceChunk],
    llm: LLM | None = None,
    *,
    project_root: Path | None = None,
    plan: AnswerPlan | None = None,
    session: ModelSession | None = None,
) -> AnswerResult:
    """Draft then review against exact, paper-scoped evidence; fall back if either stage fails.

    Writer and reviewer share the planner's input/output limits when orchestrated.
    The caller owns the 25-second request timeout. Neither model receives tools.
    """
    scoped = [
        row for row in rows if row.paper_id == paper_id and not is_untrusted_instruction(row.text)
    ]
    plan = plan or plan_answer(question)
    question = plan.question
    overview = plan.overview
    session = session or ModelSession(ctx, llm)
    reviewed = rejected = 0

    async def fallback(reason: str = "") -> AnswerResult:
        excerpts = await select_excerpts(
            ctx,
            question,
            scoped,
            max_quotes=plan.max_claims,
            max_bytes=plan.extractive_bytes,
            previous_source_ids=plan.previous_source_ids if plan.follow_up else (),
            paragraph_size=3 if plan.depth == "detailed" else 0,
        )
        return AnswerResult(
            excerpts,
            "extractive_summary" if overview else "source_excerpts",
            session.calls,
            reviewed,
            rejected,
            session.input_bytes,
            session.output_tokens,
            reason or session.unavailable_reason,
            required_aspects=len(plan.aspects),
        )

    if session.llm is None or not scoped:
        return await fallback()
    windows = _source_windows(scoped, plan, max_bytes=2500 if session.calls else 3500)
    if not windows:
        return await fallback()

    try:
        writer, reviewer = await asyncio.to_thread(answer_prompts, project_root or PROJECT_ROOT)
        writer_limit = 472 if session.calls else 600
        draft = await session.call(
            writer,
            {
                "task": plan.task,
                "user_question": plan.user_question or question,
                "question": question,
                "required_aspects": list(plan.aspects),
                "depth": plan.depth,
                "target_words": plan.target_words,
                "max_claims": plan.max_claims,
                "max_paragraphs": 3 if plan.depth == "detailed" else 1,
                "output_limit": writer_limit,
                "source_passages": [
                    {"source": i, "section": row.section, "page": row.page_start, "text": row.text}
                    for i, row in enumerate(windows)
                ],
            },
            writer_limit,
            13.0,
        )
        if not isinstance(draft, dict):
            return await fallback("invalid_draft")
        claims = cast(dict[str, object], draft).get("claims")
        if not isinstance(claims, list):
            return await fallback("invalid_draft")
        candidates: list[Excerpt] = []
        allowed_labels = {
            "Contribution",
            "Method",
            "Evaluation",
            "Findings",
            "Limitations",
            "Answer",
        }
        for raw in cast(list[object], claims)[: plan.max_claims]:
            rejected += 1
            if not isinstance(raw, dict):
                continue
            item = cast(dict[str, object], raw)
            text, quote, index = item.get("text"), item.get("quote"), item.get("source")
            if not isinstance(text, str) or type(index) is not int:
                continue
            if not 0 <= index < len(windows) or not 15 <= len(text) <= 900:
                continue
            row = windows[index]
            if quote is None:
                quote = row.text
            if not isinstance(quote, str) or len(quote) < 15:
                continue
            span = locate_quote(row.text, quote)
            if span is None:
                continue
            start, end = span
            authoritative = row.text[start:end]
            if not preserves_anchors(text, authoritative):
                continue
            if any(
                candidate.text == text or candidate.quote == authoritative
                for candidate in candidates
            ):
                continue
            label = str(item.get("aspect", "Answer"))
            paragraph = item.get("paragraph", 0)
            if type(paragraph) is not int or not 0 <= paragraph <= (
                2 if plan.depth == "detailed" else 0
            ):
                continue
            candidates.append(
                Excerpt(
                    text.strip(),
                    authoritative,
                    row.page_start,
                    row.char_start + start,
                    row.char_start + end,
                    row.chunk_id,
                    label if label in allowed_labels else "Answer",
                    ("tier_1", "llm_review"),
                    paragraph,
                )
            )
            rejected -= 1
        if not candidates:
            return await fallback("no_supported_draft")
        verdict = await session.call(
            reviewer,
            {
                "question": question,
                "user_question": plan.user_question or question,
                "task": plan.task,
                "required_aspects": list(plan.aspects),
                "depth": plan.depth,
                "claims": [
                    {"id": i, "summary": item.text, "evidence": item.quote}
                    for i, item in enumerate(candidates)
                ],
            },
            100,
            7.0,
        )
        reviewed = len(candidates)
        if not isinstance(verdict, dict):
            return await fallback("invalid_review")
        review = cast(dict[str, object], verdict)
        if review.get("answers_question") is not True:
            rejected += len(candidates)
            return await fallback("question_not_answered")
        if plan.depth == "detailed" and review.get("adequate_detail") is not True:
            rejected += len(candidates)
            return await fallback("insufficient_detail")
        supported = review.get("supported")
        if not isinstance(supported, list):
            return await fallback("invalid_review")
        accepted = {
            value
            for value in cast(list[object], supported)
            if type(value) is int and 0 <= value < len(candidates)
        }
        excerpts = [candidate for i, candidate in enumerate(candidates) if i in accepted]
        rejected += len(candidates) - len(excerpts)
        if not excerpts:
            return await fallback("review_rejected")
        covered = review.get("covered", [])
        if not isinstance(covered, list):
            return await fallback("invalid_coverage")
        covered_ids = {
            value
            for value in cast(list[object], covered)
            if type(value) is int and 0 <= value < len(plan.aspects)
        }
        if plan.aspects and len(covered_ids) != len(plan.aspects):
            return await fallback("incomplete_coverage")
        return AnswerResult(
            excerpts,
            "grounded_summary" if overview else "reviewed_answer",
            session.calls,
            reviewed,
            rejected,
            session.input_bytes,
            session.output_tokens,
            answer_relevant=True,
            covered_aspects=len(covered_ids),
            required_aspects=len(plan.aspects),
        )
    except Exception as exc:
        return await fallback(failure_reason(exc))
