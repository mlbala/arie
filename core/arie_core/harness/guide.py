"""The paper guide: one cited reading guide per paper, written once after indexing (S14).

One streamed call per paper, after the pack the writer and the judge send (same shared header,
same pack bytes, so the provider serves the pack from its cache), returns one JSON object. Each
factual item must carry ``[cN]`` markers and pass the deterministic citation checks of S10
(``citations.py``); an item that fails is dropped and named in ``dropped``, never repaired. A
reply that is not one JSON object, or keeps no cited item, fails the guide; a reply cut off by
the output cap keeps the items it finished.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import cast

import msgspec
import yaml

from arie_core.contracts import LLMRequest, TokenUsage
from arie_core.harness.answer import PROJECT_ROOT
from arie_core.harness.citations import check_answer
from arie_core.harness.explain import load_explain_prompts, paged_pack
from arie_core.harness.guardrails import inspect_question
from arie_core.harness.intent import plan_answer
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import (
    GUIDE_BUDGET,
    ModelSession,
    StreamBudget,
    StreamStopped,
    failure_reason,
)
from arie_core.pipeline.pack import (
    PaperPack,
    fence_pack,
    marker_numbers,
    pack_chunks,
    pack_fits,
    pack_limit,
)
from arie_core.plugin_api import LLM, Ctx

logger = logging.getLogger(__name__)

COMPLETE = "complete"
FAILED = "failed"
MALFORMED_OUTPUT = "malformed_output"
GUIDE_EMPTY = "guide_empty"
# In ``dropped`` when the reply hit the output cap and its unfinished item was lost.
TRUNCATED = "truncated"

GUIDE_OUTPUT_TOKENS = 2500
SUGGESTED_QUESTIONS = 5
# A paper too long for the context window gets a paged pack chosen for this question.
OVERVIEW_QUESTION = "Give an overview of the whole paper: its problem, method and results."

# Most items read from each list, and the longest text kept for each kind of item.
_LIST_LIMITS = {"method_steps": 8, "results": 6, "limitations": 5}
_EQUATIONS = 4
_SYMBOLS = 12
_GLOSSARY = 8
_SUMMARY_CHARS = 500
_ITEM_CHARS = 600
_TERM_CHARS = 120
_EQUATION_CHARS = 400
_QUESTION_CHARS = 200
_MARKER = re.compile(r"[ \t]*\[\\?c\d+\]")


class GuideTerm(msgspec.Struct, frozen=True):
    """A glossary term or an equation's symbol, and its cited meaning."""

    term: str
    meaning: str


class GuideEquation(msgspec.Struct, frozen=True):
    """An equation as the paper writes it, what it computes, and every symbol in it."""

    equation: str
    meaning: str
    symbols: tuple[GuideTerm, ...] = ()


class SectionSummary(msgspec.Struct, frozen=True):
    """One or two cited sentences on one outline heading of the pack."""

    section: str
    summary: str


class PaperGuide(msgspec.Struct, frozen=True, kw_only=True):
    """A validated guide. Every text but the questions, terms and equations carries markers.

    ``markers`` are the chunk numbers the kept items cite, in order of first use. ``dropped``
    names the items the checks removed (``results[2]``, ``key_equations``). ``revision`` is the
    revision of the pack the guide was written from.
    """

    tldr: str = ""
    problem: str = ""
    method_steps: tuple[str, ...] = ()
    key_equations: tuple[GuideEquation, ...] = ()
    results: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    glossary: tuple[GuideTerm, ...] = ()
    section_summaries: tuple[SectionSummary, ...] = ()
    suggested_questions: tuple[str, ...] = ()
    markers: tuple[int, ...] = ()
    dropped: tuple[str, ...] = ()
    prompt_version: str = ""
    revision: str = ""

    def summaries(self) -> dict[str, str]:
        """Section summaries without their markers, keyed by heading (for the strict planner)."""
        return {item.section: strip_markers(item.summary) for item in self.section_summaries}


class GuideResult(msgspec.Struct, frozen=True, kw_only=True):
    """What ``write_guide`` returns. A ``failed`` result has no guide and names its reason."""

    status: str
    guide: PaperGuide | None = None
    fallback_reason: str = ""
    prompt_version: str = ""
    usage: TokenUsage | None = None
    seconds: float = 0.0


def load_guide_rules(project_root: Path) -> tuple[str, str]:
    """``(rules, version)`` from ``prompts/guide.v1.yaml``; fail closed on a missing part."""
    loaded: object = yaml.safe_load((project_root / "prompts" / "guide.v1.yaml").read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("The guide prompt must be a mapping")
    data = cast(Mapping[str, object], loaded)
    rules, title, version = data.get("guide_rules"), data.get("name"), data.get("version")
    if not isinstance(rules, str) or not rules.strip():
        raise ValueError("The guide prompt requires guide_rules")
    if not isinstance(title, str) or not title or type(version) is not int:
        raise ValueError("The guide prompt requires a name and an integer version")
    return rules.strip(), f"{title}.v{version}"


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).strip()


def build_guide_prompt(header: str, pack_text: str, rules: str) -> LLMRequest:
    """Header as system prompt, then the fenced pack and the guide rules. Pure."""
    return LLMRequest(
        prompt=fence_pack(pack_text) + "\n\n" + rules.strip() + "\n",
        system_prompt=header,
        tools=[],
        max_output_tokens=GUIDE_OUTPUT_TOKENS,
    )


def _first_object(text: str) -> tuple[str, bool]:
    """The first JSON object in ``text`` and whether the reply stopped before it closed.

    Words or a code fence around the object are ignored. A reply cut off by the output cap is
    cut back to its last complete value (after a ``,`` or a closing bracket) and closed, so
    the items written in full are kept and the one being written is lost.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    closers: list[str] = []
    cut: tuple[int, tuple[str, ...]] | None = None
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]":
            if not closers or closers.pop() != char:
                raise ValueError("unbalanced JSON")
            if not closers:
                return text[start : index + 1], False
            cut = (index + 1, tuple(closers))
        elif char == ",":
            cut = (index, tuple(closers))
    if cut is None:
        raise ValueError("no complete JSON value")
    end, still_open = cut
    return text[start:end] + "".join(reversed(still_open)), True


def _json_object(text: str) -> tuple[dict[str, object], bool]:
    """The reply's JSON object and whether it had to be closed after a cut-off."""
    body, truncated = _first_object(text)
    loaded: object = json.loads(body)
    if not isinstance(loaded, dict):
        raise ValueError("not a JSON object")
    return cast(dict[str, object], loaded), truncated


class _Validator:
    """Keeps the items of a guide reply that pass the checks; records the ones it drops."""

    def __init__(self, sources: Mapping[int, str]) -> None:
        self.sources = sources
        self.dropped: list[str] = []

    def drop(self, name: str) -> None:
        self.dropped.append(name)

    def cited(self, value: object, name: str, limit: int = _ITEM_CHARS) -> str | None:
        """``value`` if it is text that cites the pack and no sentence of it is flagged."""
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            self.drop(name)
            return None
        text = " ".join(value.split())
        checks = check_answer(text, self.sources)
        if not checks.cited or checks.flagged:
            self.drop(name)
            return None
        return text

    def plain(self, value: object, name: str, limit: int) -> str | None:
        """``value`` as one line of text without markers (a term, an equation, a question)."""
        text = strip_markers(" ".join(value.split())) if isinstance(value, str) else ""
        if not text or len(text) > limit:
            self.drop(name)
            return None
        return text

    def items(self, value: object, name: str, most: int) -> list[object]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.drop(name)
            return []
        return cast(list[object], value)[:most]

    def term(self, value: object, name: str) -> GuideTerm | None:
        if not isinstance(value, dict):
            self.drop(name)
            return None
        data = cast(dict[str, object], value)
        term = self.plain(data.get("term"), f"{name}.term", _TERM_CHARS)
        meaning = self.cited(data.get("meaning"), f"{name}.meaning") if term else None
        return GuideTerm(term, meaning) if term and meaning else None

    def equation(self, value: object, name: str) -> GuideEquation | None:
        """An equation is kept only if its meaning and every symbol's meaning pass."""
        if not isinstance(value, dict):
            self.drop(name)
            return None
        data = cast(dict[str, object], value)
        equation = self.plain(data.get("equation"), f"{name}.equation", _EQUATION_CHARS)
        meaning = self.cited(data.get("meaning"), f"{name}.meaning") if equation else None
        raw = self.items(data.get("symbols"), f"{name}.symbols", _SYMBOLS) if meaning else []
        symbols = [self.term(item, f"{name}.symbols[{i}]") for i, item in enumerate(raw)]
        if not equation or not meaning or None in symbols:
            if equation and meaning:
                self.drop(name)
            return None
        return GuideEquation(equation, meaning, tuple(item for item in symbols if item))


def _kept[T](values: Sequence[T | None]) -> tuple[T, ...]:
    return tuple(value for value in values if value is not None)


def validate_guide(
    data: Mapping[str, object], sources: Mapping[int, str], outline: Sequence[str]
) -> PaperGuide:
    """Keep the items of a parsed guide reply that pass the S10 checks against ``sources``.

    ``sources`` maps a marker to the raw text of a chunk the model saw, so a marker outside it
    fails its item. Section summaries keep only headings in ``outline``, in outline order.
    Suggested questions pass the question guard and are kept up to five, without markers.
    Raises ``ValueError`` if no cited item survives.
    """
    check = _Validator(sources)
    lists: dict[str, tuple[str, ...]] = {}
    for name, most in _LIST_LIMITS.items():
        raw = check.items(data.get(name), name, most)
        lists[name] = _kept([check.cited(item, f"{name}[{i}]") for i, item in enumerate(raw)])
    equations = check.items(data.get("key_equations"), "key_equations", _EQUATIONS)
    glossary = check.items(data.get("glossary"), "glossary", _GLOSSARY)

    summaries: dict[str, str] = {}
    raw_summaries = data.get("section_summaries")
    if isinstance(raw_summaries, dict):
        known = set(outline)
        for key, value in cast(dict[str, object], raw_summaries).items():
            heading = " ".join(key.split())
            if heading not in known:
                check.drop(f"section_summaries[{heading[:60]}]")
            elif text := check.cited(value, f"section_summaries[{heading[:60]}]", _SUMMARY_CHARS):
                summaries[heading] = text
    elif raw_summaries is not None:
        check.drop("section_summaries")

    questions: list[str] = []
    asked = check.items(data.get("suggested_questions"), "suggested_questions", 8)
    for i, item in enumerate(asked):
        text = check.plain(item, f"suggested_questions[{i}]", _QUESTION_CHARS)
        if text and inspect_question(text).allowed and text not in questions:
            questions.append(text)

    def single(name: str) -> str:
        value = data.get(name)
        return (check.cited(value, name) or "") if value is not None else ""

    guide = PaperGuide(
        tldr=single("tldr"),
        problem=single("problem"),
        method_steps=lists["method_steps"],
        key_equations=_kept(
            [check.equation(item, f"key_equations[{i}]") for i, item in enumerate(equations)]
        ),
        results=lists["results"],
        limitations=lists["limitations"],
        glossary=_kept([check.term(item, f"glossary[{i}]") for i, item in enumerate(glossary)]),
        section_summaries=tuple(
            SectionSummary(name, summaries[name]) for name in outline if name in summaries
        ),
        suggested_questions=tuple(questions[:SUGGESTED_QUESTIONS]),
        dropped=tuple(check.dropped),
    )
    markers = tuple(dict.fromkeys(marker_numbers(" ".join(_cited_texts(guide)))))
    if not markers:
        raise ValueError(GUIDE_EMPTY)
    return msgspec.structs.replace(guide, markers=markers)


def _cited_texts(guide: PaperGuide) -> list[str]:
    texts = [guide.tldr, guide.problem, *guide.method_steps, *guide.results, *guide.limitations]
    for equation in guide.key_equations:
        texts += [equation.meaning, *(symbol.meaning for symbol in equation.symbols)]
    texts += [term.meaning for term in guide.glossary]
    texts += [item.summary for item in guide.section_summaries]
    return [text for text in texts if text]


async def write_guide(
    ctx: Ctx,
    paper_id: str,
    pack: PaperPack,
    rows: Sequence[EvidenceChunk],
    llm: LLM | None,
    *,
    context_tokens: int,
    pack_share: float = 0.6,
    budget: StreamBudget = GUIDE_BUDGET,
    project_root: Path = PROJECT_ROOT,
) -> GuideResult:
    """Write and validate the guide of one paper in one streamed call; never raises on a model
    failure.

    ``pack`` must be the stored pack of ``paper_id`` and ``rows`` are re-filtered by it (I1);
    ``paper_id`` never enters the prompt. A paper whose pack does not fit ``pack_share`` of
    ``context_tokens`` gets a paged pack chosen for an overview question. ``ctx`` should be the
    guide's own context, with its own budget (I3).
    """
    started = time.monotonic()
    rules, version = await asyncio.to_thread(load_guide_rules, project_root)
    header = (await asyncio.to_thread(load_explain_prompts, project_root)).shared_header
    session = ModelSession(ctx, llm)
    session.prompt_version = version

    def usage() -> TokenUsage:
        return TokenUsage(
            session.input_tokens, session.output_tokens, session.cache_hit, session.cache_miss
        )

    def failed(reason: str) -> GuideResult:
        logger.warning("Guide failed: %s trace=%s", reason, ctx.trace_id)
        return GuideResult(
            status=FAILED,
            fallback_reason=reason,
            prompt_version=version,
            usage=usage(),
            seconds=round(time.monotonic() - started, 3),
        )

    scoped = [row for row in rows if row.paper_id == paper_id]
    sent = pack
    if not pack_fits(pack, context_tokens, pack_share):
        plan = plan_answer(OVERVIEW_QUESTION)
        sent = paged_pack(paper_id, pack, scoped, plan, pack_limit(context_tokens, pack_share))
    request = build_guide_prompt(header, sent.text, rules)
    texts: list[str] = []
    try:
        stream = session.stream(
            request, input_limit=context_tokens - GUIDE_OUTPUT_TOKENS, budget=budget
        )
        async with aclosing(stream) as deltas:
            async for text in deltas:
                texts.append(text)
    except StreamStopped as exc:
        return failed(exc.reason)
    except Exception as exc:
        return failed(failure_reason(exc))

    in_pack = set(sent.chunk_numbers)
    numbered = pack_chunks(paper_id, scoped)
    sources = {number: row.text for number, row in numbered if number in in_pack}
    outline = list(
        dict.fromkeys(row.section for number, row in numbered if row.section and number in in_pack)
    )
    try:
        data, truncated = _json_object("".join(texts))
        guide = validate_guide(data, sources, outline)
    except ValueError as exc:
        logger.warning("Guide output rejected (%s) trace=%s", exc, ctx.trace_id)
        return failed(GUIDE_EMPTY if str(exc) == GUIDE_EMPTY else MALFORMED_OUTPUT)
    return GuideResult(
        status=COMPLETE,
        guide=msgspec.structs.replace(
            guide,
            dropped=(*guide.dropped, TRUNCATED) if truncated else guide.dropped,
            prompt_version=version,
            revision=pack.revision,
        ),
        prompt_version=version,
        usage=usage(),
        seconds=round(time.monotonic() - started, 3),
    )


__all__ = [
    "COMPLETE",
    "FAILED",
    "GUIDE_EMPTY",
    "GUIDE_OUTPUT_TOKENS",
    "MALFORMED_OUTPUT",
    "GuideEquation",
    "GuideResult",
    "GuideTerm",
    "PaperGuide",
    "SectionSummary",
    "TRUNCATED",
    "build_guide_prompt",
    "load_guide_rules",
    "validate_guide",
    "write_guide",
]
