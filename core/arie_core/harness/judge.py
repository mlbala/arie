"""Explain mode's judge: does the paper support each cited sentence, and which answer is better.

ADR 0011 left the judge as the only model check. ``judge_support`` sends every cited sentence of
one answer in one streamed call, after the paper pack the writer saw (same shared header, same
pack bytes, so the provider serves the pack from its cache). ``judge_pairwise`` compares an
explanation with a strict answer for the same question, once each way round. Both fail closed:
a stopped call, a provider error or output that does not list each sentence exactly once gives
a ``failed`` judgement and no verdicts, never a guess.
"""

from __future__ import annotations

import asyncio
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
from arie_core.harness.citations import CITED_OK, CitationChecks
from arie_core.harness.explain import load_explain_prompts
from arie_core.harness.session import (
    JUDGE_BUDGET,
    ModelSession,
    StreamBudget,
    StreamStopped,
    failure_reason,
)
from arie_core.plugin_api import LLM, Ctx

logger = logging.getLogger(__name__)

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
VERDICTS = (SUPPORTED, PARTIAL, UNSUPPORTED)

COMPLETE = "complete"
FAILED = "failed"
SKIPPED = "skipped"
MALFORMED_OUTPUT = "malformed_output"

EXPLAIN = "explain"
STRICT = "strict"
TIE = "tie"

# Output tokens a verdict line costs, with room to spare, and the most one call may ask for.
_TOKENS_PER_SENTENCE = 45
_BASE_TOKENS = 60
_MAX_SUPPORT_TOKENS = 3000
_PAIRWISE_TOKENS = 120
_SENTENCE_CHARS = 600
_REASON_CHARS = 240
_ANSWER_CHARS = 6000
_FACT_CHARS = 400
# Markdown a model puts around a field, stripped before the field is read.
_MARKUP = " \t*_`>#"
_NUMBER = re.compile(r"\d{1,4}", re.ASCII)
_BULLET = re.compile(r"^\s*[-•]\s+")
_MARKER = re.compile(r"[ \t]*\[\\?c\d+\]")
# Shown to the pairwise judge when the strict pipeline gave no answer.
NO_ANSWER = "(No answer was given.)"


class JudgePrompts(msgspec.Struct, frozen=True):
    support_rules: str
    pairwise_rules: str
    version: str


class SentenceVerdict(msgspec.Struct, frozen=True):
    """The judge's verdict on one cited sentence.

    ``number`` is the sentence's place in the prompt (1-based, among the cited sentences);
    ``start`` and ``end`` locate it in the answer text, as in ``CitationChecks``.
    """

    number: int
    start: int
    end: int
    verdict: str
    reason: str


class SupportJudgement(msgspec.Struct, frozen=True, kw_only=True):
    """What ``judge_support`` returns. ``status`` is ``complete``, ``failed`` or ``skipped``.

    ``skipped`` means the answer cited nothing, so no model was called. A ``failed`` judgement
    has no verdicts and names its reason in ``fallback_reason``.
    """

    status: str
    verdicts: tuple[SentenceVerdict, ...] = ()
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    fallback_reason: str = ""
    prompt_version: str = ""
    usage: TokenUsage | None = None
    seconds: float = 0.0


class PairwiseJudgement(msgspec.Struct, frozen=True, kw_only=True):
    """Explain against strict, judged twice. ``winner`` is ``explain``, ``strict`` or ``tie``.

    ``as_a`` and ``as_b`` are the two verdicts (each ``explain``, ``strict`` or ``tie``) when
    the explanation was shown first and second. They must agree on a winner; any disagreement
    is a tie. ``score`` is 1.0, 0.5 or 0.0 for the explanation, ``None`` when a call failed.
    """

    status: str
    winner: str = ""
    as_a: str = ""
    as_b: str = ""
    score: float | None = None
    reasons: tuple[str, ...] = ()
    fallback_reason: str = ""


def load_judge_prompts(project_root: Path) -> JudgePrompts:
    """Load ``prompts/judge.v1.yaml``; fail closed on a missing or invalid part."""
    loaded: object = yaml.safe_load((project_root / "prompts" / "judge.v1.yaml").read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("The judge prompt must be a mapping")
    data = cast(Mapping[str, object], loaded)
    texts: dict[str, str] = {}
    for name in ("support_rules", "pairwise_rules"):
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The judge prompt requires {name}")
        texts[name] = value.strip()
    title, version = data.get("name"), data.get("version")
    if not isinstance(title, str) or not title or type(version) is not int:
        raise ValueError("The judge prompt requires a name and an integer version")
    return JudgePrompts(texts["support_rules"], texts["pairwise_rules"], f"{title}.v{version}")


def strip_markers(text: str) -> str:
    """The answer without its ``[cN]`` markers, which would tell the pairwise judge its mode."""
    return _MARKER.sub("", text).strip()


def _flat(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def build_support_prompt(
    header: str, pack_text: str, rules: str, sentences: Sequence[str], *, max_output_tokens: int
) -> LLMRequest:
    """The support request: header, then the pack bytes the writer sent, then rules and sentences.

    Pure. The pack comes first and unchanged, so the prefix matches the writer's up to the end
    of the pack. The sentences keep their ``[cN]`` markers; the chunks are not sent again.
    """
    numbered = "\n".join(
        f"{number}. {_flat(text, _SENTENCE_CHARS)}" for number, text in enumerate(sentences, 1)
    )
    parts = [pack_text.rstrip("\n"), rules.strip(), "Sentences to judge:\n\n" + numbered]
    return LLMRequest(
        prompt="\n\n".join(parts) + "\n",
        system_prompt=header,
        tools=[],
        max_output_tokens=max_output_tokens,
    )


def parse_verdicts(text: str, count: int) -> list[tuple[str, str]]:
    """Read ``N | verdict | reason`` lines into ``(verdict, reason)`` for sentences 1 to ``count``.

    Lines that are not a numbered verdict (a preamble, a table rule) are ignored. Raises
    ``ValueError`` if a verdict word is unknown, or if any sentence number is missing, repeated
    or outside ``1..count``: a half-read answer would hide which sentences went unjudged.
    """
    found: dict[int, tuple[str, str]] = {}
    repeated: set[int] = set()
    for line in text.splitlines():
        bare = _BULLET.sub("", line).strip().strip("|")
        parts = [part.strip(_MARKUP) for part in bare.split("|", 2)]
        number = parts[0].rstrip(".)")
        if len(parts) < 2 or not _NUMBER.fullmatch(number):
            continue
        verdict = parts[1].lower().strip(".")
        if verdict not in VERDICTS:
            raise ValueError(f"sentence {number} has unknown verdict {parts[1][:20]!r}")
        index = int(number)
        if index in found:
            repeated.add(index)
        found[index] = (verdict, _flat(parts[2], _REASON_CHARS) if len(parts) > 2 else "")
    problems = [
        f"{name} {sorted(numbers)}"
        for name, numbers in (
            ("missing", {n for n in range(1, count + 1) if n not in found}),
            ("extra", {n for n in found if not 1 <= n <= count}),
            ("repeated", repeated),
        )
        if numbers
    ]
    if problems:
        raise ValueError("; ".join(problems))
    return [found[number] for number in range(1, count + 1)]


async def _ask(
    session: ModelSession, request: LLMRequest, *, input_limit: int, budget: StreamBudget
) -> str:
    texts: list[str] = []
    async with aclosing(
        session.stream(request, input_limit=input_limit, budget=budget)
    ) as deltas:
        async for text in deltas:
            texts.append(text)
    return "".join(texts)


def _usage(session: ModelSession) -> TokenUsage:
    return TokenUsage(
        session.input_tokens, session.output_tokens, session.cache_hit, session.cache_miss
    )


async def judge_support(
    ctx: Ctx,
    llm: LLM | None,
    pack_text: str,
    checks: CitationChecks,
    *,
    context_tokens: int,
    budget: StreamBudget = JUDGE_BUDGET,
    project_root: Path = PROJECT_ROOT,
) -> SupportJudgement:
    """Judge every cited sentence of one answer in one call; never raises on a model failure.

    ``pack_text`` is the pack the writer saw, and ``checks`` the answer's sentence checks. Each
    sentence with a marker is judged, flagged or not. An answer that cites nothing makes no
    call. ``ctx`` should be the judge's own context, with its own budget (I3).
    """
    started = time.monotonic()
    cited = [sentence for sentence in checks.sentences if sentence.markers]
    if not cited or not pack_text:
        return SupportJudgement(status=SKIPPED)
    prompts = await asyncio.to_thread(load_judge_prompts, project_root)
    header = (await asyncio.to_thread(load_explain_prompts, project_root)).shared_header
    session = ModelSession(ctx, llm)
    session.prompt_version = prompts.version

    def failed(reason: str) -> SupportJudgement:
        logger.warning("Judge failed: %s trace=%s", reason, ctx.trace_id)
        return SupportJudgement(
            status=FAILED,
            fallback_reason=reason,
            prompt_version=prompts.version,
            usage=_usage(session),
            seconds=round(time.monotonic() - started, 3),
        )

    limit = min(_MAX_SUPPORT_TOKENS, _BASE_TOKENS + _TOKENS_PER_SENTENCE * len(cited))
    request = build_support_prompt(
        header,
        pack_text,
        prompts.support_rules,
        [sentence.text for sentence in cited],
        max_output_tokens=limit,
    )
    try:
        text = await _ask(session, request, input_limit=context_tokens - limit, budget=budget)
    except StreamStopped as exc:
        return failed(exc.reason)
    except Exception as exc:
        return failed(failure_reason(exc))
    try:
        parsed = parse_verdicts(text, len(cited))
    except ValueError as exc:
        logger.warning("Judge output rejected (%s) trace=%s", exc, ctx.trace_id)
        return failed(MALFORMED_OUTPUT)
    verdicts = tuple(
        SentenceVerdict(number, sentence.start, sentence.end, verdict, reason)
        for number, (sentence, (verdict, reason)) in enumerate(zip(cited, parsed, strict=True), 1)
    )
    return SupportJudgement(
        status=COMPLETE,
        verdicts=verdicts,
        supported=sum(item.verdict == SUPPORTED for item in verdicts),
        partial=sum(item.verdict == PARTIAL for item in verdicts),
        unsupported=sum(item.verdict == UNSUPPORTED for item in verdicts),
        prompt_version=prompts.version,
        usage=_usage(session),
        seconds=round(time.monotonic() - started, 3),
    )


def citation_support(checks: CitationChecks, judgement: SupportJudgement) -> float | None:
    """The share of cited sentences that pass the deterministic checks and the judge.

    A cited sentence counts when its status is ``cited_ok`` and the judge did not call it
    ``unsupported`` (``partial`` counts). ``None`` unless the judge completed and the answer
    cites something.
    """
    cited = [sentence for sentence in checks.sentences if sentence.markers]
    if judgement.status != COMPLETE or not cited:
        return None
    unsupported = {item.start for item in judgement.verdicts if item.verdict == UNSUPPORTED}
    ok = sum(item.status == CITED_OK and item.start not in unsupported for item in cited)
    return round(ok / len(cited), 3)


def build_pairwise_prompt(
    header: str,
    rules: str,
    question: str,
    key_facts: Sequence[str],
    first: str,
    second: str,
) -> LLMRequest:
    """The pairwise request: no paper, since the key facts are the reference."""
    facts = "\n".join(f"- {_flat(fact, _FACT_CHARS)}" for fact in key_facts if fact.strip())
    parts = [
        rules.strip(),
        f"Question: {_flat(question, 700)}",
        "Key facts:\n" + (facts or "- (none given)"),
        f"Answer A:\n{first.strip()[:_ANSWER_CHARS]}",
        f"Answer B:\n{second.strip()[:_ANSWER_CHARS]}",
    ]
    return LLMRequest(
        prompt="\n\n".join(parts) + "\n",
        system_prompt=header,
        tools=[],
        max_output_tokens=_PAIRWISE_TOKENS,
    )


def parse_pairwise(text: str) -> tuple[str, str]:
    """Read ``verdict | reason`` from the first line; the verdict is ``A``, ``B`` or ``TIE``."""
    line = next((line for line in text.splitlines() if line.strip()), "")
    head, _, reason = line.partition("|")
    verdict = re.sub(r"^verdict\W*", "", head.strip(_MARKUP), flags=re.IGNORECASE)
    verdict = verdict.strip(_MARKUP + ".:").upper()
    if verdict not in {"A", "B", "TIE"}:
        raise ValueError(f"unknown verdict {head.strip()[:20]!r}")
    return verdict, _flat(reason.strip(_MARKUP), _REASON_CHARS)


async def judge_pairwise(
    ctx: Ctx,
    llm: LLM | None,
    question: str,
    key_facts: Sequence[str],
    explain_text: str,
    strict_text: str,
    *,
    budget: StreamBudget = JUDGE_BUDGET,
    project_root: Path = PROJECT_ROOT,
) -> PairwiseJudgement:
    """Judge an explanation against a strict answer, once each way round, and combine.

    Two calls run side by side, each on its own session. The explanation wins or loses only
    if both orders agree; otherwise it is a tie. If either call fails, the judgement fails
    and has no score.
    """
    prompts = await asyncio.to_thread(load_judge_prompts, project_root)
    header = (await asyncio.to_thread(load_explain_prompts, project_root)).shared_header

    async def once(first: str, second: str, first_is: str) -> tuple[str, str]:
        session = ModelSession(ctx, llm)
        request = build_pairwise_prompt(
            header, prompts.pairwise_rules, question, key_facts, first, second
        )
        text = await _ask(session, request, input_limit=12_000, budget=budget)
        verdict, reason = parse_pairwise(text)
        other = STRICT if first_is == EXPLAIN else EXPLAIN
        return {"A": first_is, "B": other, "TIE": TIE}[verdict], reason

    results = await asyncio.gather(
        once(explain_text, strict_text, EXPLAIN),
        once(strict_text, explain_text, STRICT),
        return_exceptions=True,
    )
    outcomes: list[tuple[str, str]] = []
    for item in results:
        if isinstance(item, tuple):
            outcomes.append(item)
            continue
        if not isinstance(item, Exception):
            raise item
        reason = (
            item.reason
            if isinstance(item, StreamStopped)
            else MALFORMED_OUTPUT
            if isinstance(item, ValueError)
            else failure_reason(item)
        )
        logger.warning("Pairwise judge failed: %s trace=%s", reason, ctx.trace_id)
        return PairwiseJudgement(status=FAILED, fallback_reason=reason)
    (as_a, reason_a), (as_b, reason_b) = outcomes
    winner = as_a if as_a == as_b else TIE
    return PairwiseJudgement(
        status=COMPLETE,
        winner=winner,
        as_a=as_a,
        as_b=as_b,
        score={EXPLAIN: 1.0, TIE: 0.5, STRICT: 0.0}[winner],
        reasons=(reason_a, reason_b),
    )


__all__ = [
    "COMPLETE",
    "NO_ANSWER",
    "FAILED",
    "MALFORMED_OUTPUT",
    "PARTIAL",
    "SKIPPED",
    "SUPPORTED",
    "UNSUPPORTED",
    "JudgePrompts",
    "PairwiseJudgement",
    "SentenceVerdict",
    "SupportJudgement",
    "build_pairwise_prompt",
    "build_support_prompt",
    "citation_support",
    "judge_pairwise",
    "judge_support",
    "load_judge_prompts",
    "parse_pairwise",
    "parse_verdicts",
    "strip_markers",
]
