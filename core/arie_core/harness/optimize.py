"""Pure parts of the offline prompt optimizer (S17): call budget, case split, metric, feedback.

No optimizer library, database or network here; ``arie_api.optimize_prompts`` wires GEPA to
these. The optimizer never changes a live prompt: it writes ``prompts/explain.v<N+1>.yaml``,
which nothing loads until a person switches the explain prompt to it.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import cast

import msgspec
import yaml

from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse
from arie_core.harness.citations import FLAGGED, CitationChecks
from arie_core.harness.explain import ExplainPrompts
from arie_core.harness.judge import FAILED, PARTIAL, UNSUPPORTED, SupportJudgement
from arie_core.harness.paper_eval import PaperEvalCase, PaperEvalResult
from arie_core.plugin_api import LLM, Ctx

# The share of cases kept out of the optimizer to score the old and new prompt on.
HELD_OUT_SHARE = 0.3
# The metric is fact recall (0..1) plus citation support (0..1) minus this for an answer the
# reader would have seen as excerpts, so it runs from -1 to 2.
EXCERPT_PENALTY = 1.0
PERFECT_SCORE = 2.0
MAX_RULES_CHARS = 4000
_MIN_GUARDED_CHARS = 12
_MARKER = re.compile(r"\[c\d+\]")
_FEEDBACK_CHARS = 1500
_SENTENCE_CHARS = 160
_LISTED = 3


class BudgetExhausted(RuntimeError):
    """The run asked for a model call beyond its cap."""


class CallBudget:
    """A hard cap on the model calls of one optimizer run (I3).

    ``take`` is called before every call; one beyond ``limit`` is refused, never made.
    ``refused`` stays true afterwards, so a caller whose own code swallowed the error can still
    tell that a score was cut short by the cap.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("The call cap must be at least 1")
        self.limit = limit
        self.used = 0
        self.refused = False

    @property
    def left(self) -> int:
        return self.limit - self.used

    def take(self) -> None:
        if self.used >= self.limit:
            self.refused = True
            raise BudgetExhausted(f"The cap of {self.limit} model calls is used up")
        self.used += 1


class CountingLLM:
    """An LLM that spends one unit of a ``CallBudget`` before each call it passes on."""

    def __init__(self, llm: LLM, budget: CallBudget) -> None:
        self.llm = llm
        self.budget = budget

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        self.budget.take()
        return await self.llm.complete(ctx, request)

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        self.budget.take()
        deltas = aiter(self.llm.stream(ctx, request))
        try:
            async for delta in deltas:
                yield delta
        finally:
            if isinstance(deltas, AsyncGenerator):
                await cast(AsyncGenerator[LLMDelta], deltas).aclose()


class OptCase(msgspec.Struct, frozen=True):
    """One scored case and where it came from; ``reasons`` are a reader's thumbs-down reasons."""

    paper_id: str
    case: PaperEvalCase
    reasons: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.paper_id, self.case.case_id


def _stratified(items: Sequence[OptCase], seed: int) -> list[list[OptCase]]:
    """The cases grouped by origin (sorted), each group in a seeded shuffle."""
    groups: dict[str, list[OptCase]] = defaultdict(list)
    for item in sorted(items, key=lambda item: item.key):
        groups[item.case.origin].append(item)
    ordered: list[list[OptCase]] = []
    for origin in sorted(groups):
        group = groups[origin]
        random.Random(f"{seed}:{origin}").shuffle(group)
        ordered.append(group)
    return ordered


def pick_cases(items: Sequence[OptCase], limit: int | None, seed: int) -> list[OptCase]:
    """At most ``limit`` cases, taken from every origin in turn, the same ones for one seed."""
    groups = _stratified(items, seed)
    picked: list[OptCase] = []
    depth = 0
    while any(depth < len(group) for group in groups) and (limit is None or len(picked) < limit):
        for group in groups:
            if depth < len(group) and (limit is None or len(picked) < limit):
                picked.append(group[depth])
        depth += 1
    return picked


def split_cases(
    items: Sequence[OptCase], *, seed: int, held_out_share: float = HELD_OUT_SHARE
) -> tuple[list[OptCase], list[OptCase]]:
    """Train and held-out cases, split inside each origin so both see every source.

    Fixed by ``seed``: the same cases and seed give the same split. An origin with one case
    goes to train. The two lists never share a case.
    """
    train: list[OptCase] = []
    held_out: list[OptCase] = []
    for group in _stratified(items, seed):
        count = min(max(1, round(len(group) * held_out_share)), len(group) - 1)
        held_out.extend(group[:count])
        train.extend(group[count:])
    return train, held_out


def _support(result: PaperEvalResult) -> float | None:
    """The judge's citation support, else the deterministic valid rate (judge failed)."""
    if result.citation_support is not None:
        return result.citation_support
    return result.citation_valid_rate


def metric(result: PaperEvalResult) -> float:
    """Fact recall plus citation support, minus a penalty for an answer that was not written.

    Citation support is the judge's figure; if the judge failed it is the deterministic valid
    rate, and an answer that cites nothing gets 0.
    """
    penalty = 0.0 if result.answer_written else EXCERPT_PENALTY
    return round(result.fact_recall_prose + (_support(result) or 0.0) - penalty, 3)


def _flat(text: str) -> str:
    return " ".join(text.lower().split())


def _clip(text: str, limit: int = _SENTENCE_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def feedback_text(
    item: OptCase,
    result: PaperEvalResult,
    text: str,
    checks: CitationChecks,
    judgement: SupportJudgement,
) -> str:
    """What went wrong with one answer, for GEPA's reflection: facts, flags, judge, reader.

    Names no score weights and no paper id. Each part is short; the whole is cut at
    ``_FEEDBACK_CHARS``.
    """
    lines: list[str] = []
    if not result.answer_written:
        reason = f" ({result.fallback_reason})" if result.fallback_reason else ""
        lines.append(f"No written answer{reason}: the reader would have seen only excerpts.")
    else:
        flat = _flat(text)
        facts = [fact for fact in item.case.key_facts if fact]
        missing = [fact for fact in facts if _flat(fact) not in flat]
        if missing:
            lines.append("Key facts the answer did not state: " + "; ".join(map(_clip, missing)))
        elif facts:
            lines.append("The answer stated every key fact.")
    flagged = [sentence for sentence in checks.sentences if sentence.status == FLAGGED]
    for sentence in flagged[:_LISTED]:
        why = ", ".join(sentence.reasons) or "flagged"
        lines.append(f'Citation check flagged ({why}): "{_clip(sentence.text)}"')
    by_start = {sentence.start: sentence for sentence in checks.sentences}
    doubted = [v for v in judgement.verdicts if v.verdict in {UNSUPPORTED, PARTIAL}]
    for verdict in doubted[:_LISTED]:
        sentence = by_start.get(verdict.start)
        quoted = f' "{_clip(sentence.text)}"' if sentence else ""
        lines.append(f"The judge found it {verdict.verdict}{quoted}: {verdict.reason}")
    if judgement.status == FAILED:
        lines.append(f"The support judge failed ({judgement.fallback_reason}).")
    if result.answer_written and not checks.cited:
        lines.append("No sentence cited the paper.")
    if item.reasons:
        reasons = ", ".join(item.reasons)
        lines.append(f"A reader rated an earlier answer to this question down: {reasons}")
    if item.case.note:
        lines.append(f"Reader's note: {_clip(item.case.note, 240)}")
    return "\n".join(lines)[:_FEEDBACK_CHARS]


class Rollout(msgspec.Struct, frozen=True):
    """One case answered with one candidate and scored: what GEPA's adapter returns."""

    item: OptCase
    text: str
    result: PaperEvalResult
    score: float
    feedback: str


def summarize_rollouts(rollouts: Sequence[Rollout]) -> dict[str, float | int]:
    """Means over a set of cases: the metric, prose fact recall, citation support, fallbacks."""
    if not rollouts:
        return {"cases": 0}
    support = [
        value
        for value in (_support(r.result) for r in rollouts)
        if value is not None
    ]
    count = len(rollouts)
    return {
        "cases": count,
        "score": round(sum(r.score for r in rollouts) / count, 3),
        "fact_recall_prose": round(sum(r.result.fact_recall_prose for r in rollouts) / count, 3),
        "citation_support": round(sum(support) / len(support), 3) if support else 0.0,
        "excerpt_fallbacks": sum(not r.result.answer_written for r in rollouts),
    }


def check_rules(rules: str, forbidden: Iterable[str] = ()) -> str:
    """Why a candidate's writer rules cannot be tried ("" if they can).

    The rules must keep a ``[cN]`` marker example (the citation checks read those markers),
    stay under ``MAX_RULES_CHARS``, and quote none of ``forbidden`` (train key facts and paper
    ids): a prompt that carries the answer would score well and mean nothing.
    """
    text = rules.strip()
    if not text:
        return "empty"
    if len(text) > MAX_RULES_CHARS:
        return "too_long"
    if not _MARKER.search(text):
        return "no_marker_example"
    flat = _flat(text)
    if any(len(item) >= _MIN_GUARDED_CHARS and _flat(item) in flat for item in forbidden):
        return "quotes_ground_truth"
    return ""


def next_version(prompts_dir: Path, name: str) -> int:
    """One more than the highest ``<name>.v<N>.yaml`` in ``prompts_dir`` (1 if there is none)."""
    pattern = re.compile(rf"{re.escape(name)}\.v(\d+)\.yaml")
    matches = (pattern.fullmatch(path.name) for path in prompts_dir.iterdir())
    return max((int(match.group(1)) for match in matches if match), default=0) + 1


def _block(text: str) -> str:
    return "\n".join(f"  {line}".rstrip() for line in text.strip().splitlines())


def render_candidate(seed: ExplainPrompts, rules: str, version: int, note: str) -> str:
    """The text of ``explain.v<version>.yaml``: the seed's header and depths, new writer rules.

    Raises ``ValueError`` if reading the text back does not give exactly the rules and the
    header, so a file that would load as something else is never written.
    """
    comment = " ".join(note.split())
    text = (
        f"version: {version}\n"
        "name: explain\n"
        "# A candidate from `make optimize-prompts` (S17). Nothing loads it: a person switches\n"
        f"# the explain prompt to it, or deletes it. {comment}\n"
        f"shared_header: |\n{_block(seed.shared_header)}\n"
        f"writer_rules: |\n{_block(rules)}\n"
        "depth_words:\n"
        + "".join(f"  {depth}: {words}\n" for depth, words in seed.depth_words.items())
    )
    loaded = yaml.safe_load(text)
    same = (
        loaded["version"] == version
        and loaded["shared_header"].strip() == seed.shared_header
        and loaded["writer_rules"].strip() == rules.strip()
        and loaded["depth_words"] == seed.depth_words
    )
    if not same:
        raise ValueError("The candidate does not read back the same; not written")
    return text


__all__ = [
    "EXCERPT_PENALTY",
    "HELD_OUT_SHARE",
    "MAX_RULES_CHARS",
    "PERFECT_SCORE",
    "BudgetExhausted",
    "CallBudget",
    "CountingLLM",
    "OptCase",
    "Rollout",
    "check_rules",
    "feedback_text",
    "metric",
    "next_version",
    "pick_cases",
    "render_candidate",
    "split_cases",
    "summarize_rollouts",
]
