from __future__ import annotations

import asyncio
import re
import time
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
from arie_core.harness.verifier import (
    MIN_ENTAILMENT,
    REVIEW_OPTIONAL_ENTAILMENT,
    EntailmentCheck,
    EvidenceVerifier,
    preserves_anchors,
)
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
    input_tokens: int = 0


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


TABLE_EXCERPT_BYTES = 1200
_TABLE_CAPTION = re.compile(r"\s*[*_]*Table\s+[A-Z]?\d", re.IGNORECASE)


def table_excerpt(text: str, limit: int = TABLE_EXCERPT_BYTES) -> str:
    """The quotable start of a Markdown table: its caption, header and leading rows.

    Returns an exact substring of ``text`` that ends on a whole row, so it passes the
    quote checks like any sentence, or ``""`` when ``text`` has no table or not even
    the header and one row fit in ``limit`` bytes. A "Table N" caption paragraph just
    above the table is included.
    """
    lines = text.splitlines(keepends=True)
    first = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None)
    if first is None:
        return ""
    start = first
    above = first - 1
    while above >= 0 and not lines[above].strip():
        above -= 1
    if above >= 0:
        paragraph = above
        while paragraph > 0 and lines[paragraph - 1].strip():
            paragraph -= 1
        if _TABLE_CAPTION.match(lines[paragraph]):
            start = paragraph
    kept: list[str] = []
    used = rows = 0
    for index in range(start, len(lines)):
        line = lines[index]
        if index >= first and not line.lstrip().startswith("|"):
            break
        size = len(line.encode("utf-8"))
        if used + size > limit:
            break
        kept.append(line)
        used += size
        rows += index >= first
    # Header and separator alone say nothing; require at least one data row.
    return "".join(kept).strip() if rows >= 3 else ""


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
        elif table := table_excerpt(row.text):
            sentences.append(table)

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


# Single-point questions; every other task needs the reviewer's whole-answer judgment.
DIRECT_TASKS = frozenset({"lookup", "definition", "answer"})


def review_optional(plan: AnswerPlan, scores: Sequence[float]) -> bool:
    """Decide whether Tier 3 alone suffices, so the LLM reviewer can be skipped (ADR 0008).

    Tier 3 checks each sentence against its own quote; it cannot judge whether the
    sentences together answer the question or cover every requested point. So only a
    direct, single-aspect, non-detailed answer whose every claim scores at least
    ``REVIEW_OPTIONAL_ENTAILMENT`` skips review. Overviews, explanations, comparisons,
    critiques and multi-part questions are always reviewed.
    """
    return (
        bool(scores)
        and min(scores) >= REVIEW_OPTIONAL_ENTAILMENT
        and plan.task in DIRECT_TASKS
        and not plan.overview
        and plan.depth != "detailed"
        and not plan.detail_requested
        and len(plan.aspects) <= 1
    )


def fit_evidence(claims: Sequence[Excerpt], allowance: int) -> list[str]:
    """Fit review evidence into ``allowance`` bytes, keeping each claim's most relevant sentences.

    Kept sentences are verbatim from the cited passage, so support the reviewer finds in
    them is support in the full citation shown to the user; trimming can only add
    rejections. Evidence that still cannot fit is empty, so its claim is rejected.
    """
    quotes = [claim.quote for claim in claims]
    if sum(len(quote.encode("utf-8")) for quote in quotes) <= allowance:
        return quotes
    share = max(0, allowance) // max(1, len(claims))
    fitted: list[str] = []
    for claim in claims:
        # Every sentence or line is a candidate, including tables, never a filtered subset.
        sentences = [part for part in re.split(r"(?<=[.!?])\s+|\n+", claim.quote) if part.strip()]
        terms = query_terms(claim.text)
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: len(terms & query_terms(sentences[i])),
            reverse=True,
        )
        kept: list[int] = []
        used = 0
        for index in ranked:
            # JSON-escaped bytes plus a joining space; the request already holds the quotes.
            size = len(msgspec.json.encode(sentences[index])) - 1
            if used + size <= share:
                kept.append(index)
                used += size
        fitted.append(" ".join(sentences[index] for index in sorted(kept)))
    return fitted


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
    entail: EntailmentCheck | None = None,
    entail_floor: float = MIN_ENTAILMENT,
) -> AnswerResult:
    """Draft, entailment-check, then review against exact, paper-scoped evidence.

    Writer and reviewer share the planner's input/output limits when orchestrated.
    The caller owns the 25-second request timeout. Neither model receives tools.
    With ``entail``, each drafted claim must pass Tier 3 against its own quote before
    review; a model failure or timeout abstains to source excerpts (I2), never skips it.
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
            input_tokens=session.input_tokens,
        )

    if session.llm is None or not scoped:
        return await fallback()
    windows = _source_windows(scoped, plan, max_bytes=8000 if session.calls else 8500)
    if not windows:
        return await fallback()

    try:
        writer, reviewer = await asyncio.to_thread(answer_prompts, project_root or PROJECT_ROOT)
        # Planner 160 + writer 440 + reviewer 100 = the 700-token output cap (I3).
        writer_limit = 440 if session.calls else 600
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
        if not claims:
            # The writer judged that this paper's evidence cannot answer: abstain rather
            # than show excerpts that do not answer the question.
            return AnswerResult(
                [],
                "abstention",
                session.calls,
                fallback_reason="not_in_paper",
                input_bytes=session.input_bytes,
                output_tokens=session.output_tokens,
                required_aspects=len(plan.aspects),
                input_tokens=session.input_tokens,
            )
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
            # One passage may support several separately stated points (the writer is told
            # to split sentences), but never the same point twice.
            if any(candidate.text == text.strip() for candidate in candidates) or (
                sum(candidate.quote == authoritative for candidate in candidates) >= 3
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
        if entail is not None:
            # Tier 3 runs before review so coverage is judged only on entailed claims.
            budget = min(4.0, ctx.deadline - time.monotonic() - 3.0)
            if budget <= 0.2:
                return await fallback("entailment_timeout")
            try:
                async with asyncio.timeout(budget):
                    scores = await EvidenceVerifier.tier_3_scores(
                        entail, [(item.quote, item.text) for item in candidates]
                    )
            except TimeoutError:
                return await fallback("entailment_timeout")
            except Exception:
                return await fallback("entailment_unavailable")
            kept = [
                (msgspec.structs.replace(item, checks=(*item.checks, "tier_3_entailment")), score)
                for item, score in zip(candidates, scores, strict=True)
                if score >= entail_floor
            ]
            rejected += len(candidates) - len(kept)
            candidates = [item for item, _ in kept]
            if not candidates:
                return await fallback("entailment_rejected")
            if rejected == 0 and review_optional(plan, [score for _, score in kept]):
                return AnswerResult(
                    [
                        msgspec.structs.replace(item, checks=("tier_1", "tier_3_entailment"))
                        for item in candidates
                    ],
                    "entailed_answer",
                    session.calls,
                    0,
                    0,
                    session.input_bytes,
                    session.output_tokens,
                    required_aspects=len(plan.aspects),
                    input_tokens=session.input_tokens,
                )
        review_request: dict[str, object] = {
            "question": question,
            "user_question": plan.user_question or question,
            "task": plan.task,
            "required_aspects": list(plan.aspects),
            "depth": plan.depth,
            "claims": [
                {"id": i, "summary": item.text, "evidence": ""} for i, item in enumerate(candidates)
            ],
        }
        evidence = fit_evidence(candidates, session.remaining_input(reviewer, review_request))
        review_request["claims"] = [
            {"id": i, "summary": item.text, "evidence": quote}
            for i, (item, quote) in enumerate(zip(candidates, evidence, strict=True))
        ]
        verdict = await session.call(reviewer, review_request, 100, 7.0)
        reviewed = len(candidates)
        if not isinstance(verdict, dict):
            return await fallback("invalid_review")
        review = cast(dict[str, object], verdict)
        if review.get("answers_question") is not True:
            rejected += len(candidates)
            return await fallback("question_not_answered")
        # Shallow answers fail only when the user asked for depth, not when ARIE chose it.
        if plan.detail_requested and review.get("adequate_detail") is not True:
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
            input_tokens=session.input_tokens,
        )
    except Exception as exc:
        return await fallback(failure_reason(exc))
