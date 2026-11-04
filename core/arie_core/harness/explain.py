"""Explain mode's writer: the cached prompt, the paged pack and one streamed, cited answer.

ADR 0010. The prompt is the shared header as system prompt, then the paper pack, the writer
rules, the recent turns and the question, so every role and turn of a conversation reuses one
cached prefix. The answer's ``[cN]`` markers are mapped to chunks and every sentence is
checked against the chunks it cites (``citations.py``, S10).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import cast

import msgspec
import yaml

from arie_core.contracts import LLMRequest, TokenUsage
from arie_core.harness.answer import PROJECT_ROOT
from arie_core.harness.citations import CitationChecks, check_answer
from arie_core.harness.guardrails import inspect_question
from arie_core.harness.intent import AnswerPlan, DialogueTurn, plan_answer
from arie_core.harness.reader import EvidenceChunk, retrieve
from arie_core.harness.session import (
    BYTES_PER_TOKEN,
    EXPLAIN_BUDGET,
    ModelSession,
    StreamBudget,
    StreamStopped,
    estimate_tokens,
    failure_reason,
)
from arie_core.harness.trace import AnswerTrace
from arie_core.pipeline.pack import (
    PaperPack,
    marker_numbers,
    pack_chunks,
    pack_fits,
    pack_header,
    pack_limit,
    render_blocks,
)
from arie_core.plugin_api import LLM, Ctx

logger = logging.getLogger(__name__)

# Output tokens of the writer call per depth (ADR 0010).
OUTPUT_TOKENS = {"brief": 600, "standard": 1200, "detailed": 2500}
HISTORY_TURNS = 6
_QUESTION_CHARS = 700
_ANSWER_CHARS = 4000
# Our markers and escaped paper markers, with the spaces before them, stripped from history.
_ANY_MARKER = re.compile(r"[ \t]*\[\\?c\d+\]")
# A paged pack tries these top-level sections first, in this order.
_PRIORITY = ("abstract", "introduction", "conclu")


class ExplainPrompts(msgspec.Struct, frozen=True):
    shared_header: str
    writer_rules: str
    depth_words: dict[str, int]
    version: str


class ExplainTurn(msgspec.Struct, frozen=True):
    """One earlier turn: the reader's question and ARIE's answer. Never evidence."""

    question: str
    answer: str = ""


class ExplainCitation(msgspec.Struct, frozen=True):
    """A ``[cN]`` marker in the answer and the chunk of the bound paper it names."""

    marker: int
    chunk_id: str


class ExplainResult(msgspec.Struct, frozen=True, kw_only=True):
    """The last item ``explain_answer`` yields: the whole text and what it cites.

    ``mode`` is ``explain`` (complete), ``explain_partial`` (text arrived, then the call
    stopped), ``explain_failed`` (stopped before any text) or ``abstention`` (the guard
    blocked the question; ``message`` is for the reader). ``citations`` are unique, in order
    of first use; ``unknown_markers`` name no chunk in the pack the model saw. ``usage`` is
    the provider's report, ``None`` when it sent none. ``checks`` judges each sentence against
    the chunks it cites; it is empty when no text arrived.
    """

    text: str
    mode: str
    depth: str
    citations: tuple[ExplainCitation, ...] = ()
    unknown_markers: tuple[int, ...] = ()
    usage: TokenUsage | None = None
    paged: bool = False
    fallback_reason: str = ""
    message: str = ""
    trace: AnswerTrace = AnswerTrace()
    # Seconds from the start of the answer to the first text; ``None`` if no text arrived.
    first_token_seconds: float | None = None
    checks: CitationChecks = CitationChecks()
    # The pack text the writer was sent (the paged one if the paper did not fit), so the judge
    # can send the same bytes and reuse the provider's cached prefix. Empty if no call was made.
    pack_text: str = ""


def load_explain_prompts(project_root: Path) -> ExplainPrompts:
    """Load ``prompts/explain.v1.yaml``; fail closed on a missing or invalid part."""
    loaded: object = yaml.safe_load((project_root / "prompts" / "explain.v1.yaml").read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("The explain prompt must be a mapping")
    data = cast(Mapping[str, object], loaded)
    texts: dict[str, str] = {}
    for name in ("shared_header", "writer_rules"):
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The explain prompt requires {name}")
        texts[name] = value.strip()
    words = data.get("depth_words")
    if not isinstance(words, Mapping):
        raise ValueError("The explain prompt requires depth_words")
    depth_words: dict[str, int] = {}
    for depth in OUTPUT_TOKENS:
        value = cast(Mapping[str, object], words).get(depth)
        if type(value) is not int or value < 1:
            raise ValueError(f"The explain prompt requires depth_words.{depth}")
        depth_words[depth] = value
    title, version = data.get("name"), data.get("version")
    if not isinstance(title, str) or not title or type(version) is not int:
        raise ValueError("The explain prompt requires a name and an integer version")
    return ExplainPrompts(
        texts["shared_header"], texts["writer_rules"], depth_words, f"{title}.v{version}"
    )


def _turn_text(turn: ExplainTurn) -> str:
    question = " ".join(turn.question.split())[:_QUESTION_CHARS]
    answer = _ANY_MARKER.sub("", turn.answer).strip()[:_ANSWER_CHARS].strip()
    return f"Reader: {question}\nARIE: {answer}" if answer else f"Reader: {question}"


def build_prompt(
    header: str,
    pack_text: str,
    writer_rules: str,
    turns: Sequence[ExplainTurn],
    question: str,
    *,
    target_words: int,
    max_output_tokens: int,
) -> LLMRequest:
    """The writer request: header as system prompt; pack, rules, turns, question as prompt.

    Pure, so the same inputs give the same bytes, and nothing that changes per question comes
    before the turns. Only the last ``HISTORY_TURNS`` turns are kept, with markers stripped
    from their answers.
    """
    parts = [pack_text.rstrip("\n"), writer_rules.strip()]
    history = [_turn_text(turn) for turn in turns[-HISTORY_TURNS:]]
    if history:
        parts.append(
            "Earlier turns (context for the question, not evidence):\n\n" + "\n\n".join(history)
        )
    parts.append(f"Question: {question}\nLength: about {target_words} words.")
    return LLMRequest(
        prompt="\n\n".join(parts) + "\n",
        system_prompt=header,
        tools=[],
        max_output_tokens=max_output_tokens,
    )


def paged_pack(
    paper_id: str,
    pack: PaperPack,
    rows: Sequence[EvidenceChunk],
    plan: AnswerPlan,
    limit: int,
) -> PaperPack:
    """A pack of at most ``limit`` estimated tokens for a paper whose whole pack doesn't fit.

    Keeps the header (title, abstract, full outline) and adds whole sections: abstract,
    introduction and conclusion first, then the sections of strict mode's ``retrieve`` hits
    for the question in rank order, then the rest in page order. A section that does not fit
    whole is skipped. Only chunks of ``paper_id`` that the whole pack holds are used (I1).
    """
    in_pack = set(pack.chunk_numbers)
    kept = [(number, row) for number, row in pack_chunks(paper_id, rows) if number in in_pack]
    groups: dict[str, list[tuple[int, EvidenceChunk]]] = {}
    for item in kept:
        groups.setdefault(item[1].section, []).append(item)
    order = [
        name for word in _PRIORITY for name in groups if word in name.split(" / ")[0].lower()
    ]
    hits = retrieve(paper_id, plan.question, [row for _, row in kept])
    order += [row.section for row in hits if row.section in groups]
    order += list(groups)

    header = pack_header(pack)
    room = limit * BYTES_PER_TOKEN - len(header.encode()) - 1
    chosen: set[str] = set()
    for name in dict.fromkeys(order):
        # An upper bound: as if every block opened with its "Section:" line.
        cost = sum(len(render_blocks([item])[0].encode()) + 2 for item in groups[name])
        if cost <= room:
            chosen.add(name)
            room -= cost
    selected = [item for item in kept if item[1].section in chosen]
    text = "\n\n".join([header, *render_blocks(selected)]) + "\n"
    return PaperPack(
        text=text,
        chunk_numbers=tuple(number for number, _ in selected),
        token_estimate=estimate_tokens(len(text.encode())),
        revision=f"{pack.revision}:paged",
    )


def _check_sentences(
    paper_id: str, rows: Sequence[EvidenceChunk], in_pack: set[int], text: str
) -> CitationChecks:
    """Check ``text`` against the raw chunks of ``paper_id`` that the model saw (I1).

    The sources are the chunk texts, never the pack text, which escapes literal markers.
    """
    sources = {number: row.text for number, row in pack_chunks(paper_id, rows) if number in in_pack}
    return check_answer(text, sources)


async def explain_answer(
    ctx: Ctx,
    paper_id: str,
    question: str,
    pack: PaperPack,
    rows: Sequence[EvidenceChunk],
    llm: LLM | None,
    *,
    context_tokens: int,
    pack_share: float = 0.6,
    turns: Sequence[ExplainTurn] = (),
    budget: StreamBudget = EXPLAIN_BUDGET,
    project_root: Path = PROJECT_ROOT,
) -> AsyncGenerator[str | ExplainResult]:
    """Stream one cited explanation: text deltas as ``str``, then one ``ExplainResult``.

    Guard first (a blocked question abstains without a model call), depth from the keyword
    planner, the whole pack if it fits ``pack_share`` of ``context_tokens`` and a paged pack
    otherwise, then one streaming call within ``budget``. ``pack`` must be the stored pack of
    ``paper_id`` and ``rows`` are re-filtered by it (I1); ``paper_id`` never enters the
    prompt. A timeout, budget stop or provider error ends the stream with its reason in
    ``fallback_reason``; the text that arrived is kept.
    """
    started = time.monotonic()
    session = ModelSession(ctx, llm)
    guard = inspect_question(question)
    if not guard.allowed:
        yield ExplainResult(
            text="",
            mode="abstention",
            depth=plan_answer(question[:2000]).depth,
            fallback_reason=guard.code,
            message=guard.message,
            trace=session.trace("abstention", guard.code, time.monotonic() - started),
        )
        return
    history = tuple(
        ExplainTurn(decision.question, turn.answer)
        for turn in turns[-HISTORY_TURNS:]
        if (decision := inspect_question(turn.question)).allowed
    )
    with session.stage("plan"):
        plan = plan_answer(guard.question, [DialogueTurn(turn.question) for turn in history])
    prompts = await asyncio.to_thread(load_explain_prompts, project_root)
    session.prompt_version = prompts.version

    paged = not pack_fits(pack, context_tokens, pack_share)
    sent = pack
    if paged:
        scoped = [row for row in rows if row.paper_id == paper_id]
        with session.stage("read"):
            sent = paged_pack(paper_id, pack, scoped, plan, pack_limit(context_tokens, pack_share))
    limit = OUTPUT_TOKENS[plan.depth]
    request = build_prompt(
        prompts.shared_header,
        sent.text,
        prompts.writer_rules,
        history,
        guard.question,
        target_words=prompts.depth_words[plan.depth],
        max_output_tokens=limit,
    )

    texts: list[str] = []
    reason = ""
    first_token: float | None = None
    with session.stage("write"):
        try:
            stream = session.stream(request, input_limit=context_tokens - limit, budget=budget)
            async with aclosing(stream) as deltas:
                async for text in deltas:
                    if first_token is None:
                        first_token = round(time.monotonic() - started, 3)
                    texts.append(text)
                    yield text
        except StreamStopped as exc:
            reason = exc.reason
        except Exception as exc:
            reason = failure_reason(exc)
            logger.warning("Explain writer failed: %s trace=%s", reason, ctx.trace_id)

    text = "".join(texts)
    in_pack = set(sent.chunk_numbers)
    cited = tuple(dict.fromkeys(marker_numbers(text)))
    checks = (
        await asyncio.to_thread(_check_sentences, paper_id, rows, in_pack, text)
        if text
        else CitationChecks()
    )
    mode = "explain_partial" if reason and text else "explain_failed" if reason else "explain"
    yield ExplainResult(
        text=text,
        mode=mode,
        depth=plan.depth,
        citations=tuple(
            ExplainCitation(number, f"{paper_id}:{number}") for number in cited if number in in_pack
        ),
        unknown_markers=tuple(number for number in cited if number not in in_pack),
        usage=session.stream_usage,
        paged=paged,
        fallback_reason=reason,
        trace=session.trace(mode, reason, time.monotonic() - started),
        first_token_seconds=first_token,
        checks=checks,
        pack_text=sent.text,
    )
