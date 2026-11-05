"""The judge: one batched support call per answer, strict parsing and pairwise scoring."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from arie_api.runtime.base import CallContext
from arie_core.budget import BudgetHandle
from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.citations import CitationChecks, check_answer
from arie_core.harness.explain import build_prompt, load_explain_prompts
from arie_core.harness.judge import (
    COMPLETE,
    FAILED,
    MALFORMED_OUTPUT,
    SKIPPED,
    PairwiseJudgement,
    SentenceVerdict,
    SupportJudgement,
    citation_support,
    judge_pairwise,
    judge_support,
    load_judge_prompts,
    parse_pairwise,
    parse_verdicts,
)
from arie_core.harness.session import StreamBudget
from arie_core.plugin_api import LLM, Ctx
from conftest import ScriptedModel, StreamStep

ROOT = Path(__file__).resolve().parents[2]
PACK_TEXT = "Title: Cedar\n\n[c1] Cedar gates memory writes with a learned score.\n"
SOURCES = {
    1: "Cedar gates memory writes with a learned score.",
    2: "Accuracy increased from 62% to 68% on the Cedar benchmark.",
}
ANSWER = (
    "Memory is a general idea in retrieval. "
    "Cedar decides what to store with a learned score [c1]. "
    "Accuracy rose from 62% to 68% [c2]. "
    "It also reached 99% on every corpus [c2]."
)


def ctx(seconds: float = 60.0) -> CallContext:
    budget = BudgetHandle(max_tokens=10_000_000, max_wall_sec=seconds)
    return CallContext(time.monotonic() + seconds, "judge-test", budget)


def checks() -> CitationChecks:
    return check_answer(ANSWER, SOURCES)


async def support(model: object, found: CitationChecks | None = None) -> SupportJudgement:
    return await judge_support(
        ctx(),
        cast(LLM, model),
        PACK_TEXT,
        found if found is not None else checks(),
        context_tokens=1_000_000,
        project_root=ROOT,
    )


LINES = [
    "1 | supported | The chunk states the learned score.\n",
    "2 | partial | It gives the range but not the benchmark split.\n",
    "3 | unsupported | No chunk says 99%.\n",
]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("1 | supported | a\n2 | partial | b\n", "missing [3]"),
        (LINES[0] + LINES[2], "missing [2]"),
        (LINES[0] + LINES[1] + LINES[2] + "4 | supported | extra\n", "extra [4]"),
        ("".join(LINES) + "0 | supported | extra\n", "extra [0]"),
        ("".join(LINES) + "2 | supported | again\n", "repeated [2]"),
        ("".join(LINES).replace("partial", "mostly"), "unknown verdict"),
        ("", "missing [1, 2, 3]"),
        ("I could not tell.\n", "missing [1, 2, 3]"),
    ],
)
def test_output_that_misses_adds_or_repeats_a_sentence_number_is_rejected(
    text: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem.replace("[", r"\[").replace("]", r"\]")):
        parse_verdicts(text, 3)


def test_verdict_lines_are_read_through_the_markup_a_model_adds() -> None:
    text = (
        "Here are the verdicts:\n\n"
        "| N | verdict | reason |\n|---|---|---|\n"
        "| 1 | **Supported** | Stated in the chunk. |\n"
        "- 2. | `partial` | Only the range.\n"
        "3 | UNSUPPORTED | " + "long " * 100 + "\n"
    )
    parsed = parse_verdicts(text, 3)
    assert [verdict for verdict, _ in parsed] == ["supported", "partial", "unsupported"]
    assert parsed[0][1] == "Stated in the chunk."
    assert len(parsed[2][1]) <= 240


def test_verdicts_are_matched_to_numbers_not_to_line_order() -> None:
    parsed = parse_verdicts("3 | partial | c\n1 | supported | a\n2 | unsupported | b\n", 3)
    assert [verdict for verdict, _ in parsed] == ["supported", "unsupported", "partial"]


@pytest.mark.asyncio
async def test_every_cited_sentence_is_judged_in_one_call() -> None:
    model = ScriptedModel(streams=[["1 | supported | s\n", "2 | partial | p\n3 | unsupported | u"]])
    found = checks()
    judgement = await support(model, found)
    assert len(model.requests) == 1
    assert judgement.status == COMPLETE and judgement.fallback_reason == ""
    assert (judgement.supported, judgement.partial, judgement.unsupported) == (1, 1, 1)
    cited = [sentence for sentence in found.sentences if sentence.markers]
    assert len(cited) == 3  # the uncited first sentence is background, not judged
    assert [(v.number, v.start, v.end) for v in judgement.verdicts] == [
        (1, cited[0].start, cited[0].end),
        (2, cited[1].start, cited[1].end),
        (3, cited[2].start, cited[2].end),
    ]
    assert judgement.verdicts[2].reason == "u" and judgement.prompt_version == "judge.v1"
    assert judgement.usage == TokenUsage(1, 30)


@pytest.mark.asyncio
async def test_flagged_sentences_are_judged_too() -> None:
    found = check_answer("Cedar scored 91% [c1]. Cedar uses a score [c1].", SOURCES)
    assert [s.status for s in found.sentences] == ["flagged", "cited_ok"]
    model = ScriptedModel(streams=[["1 | unsupported | no 91%\n2 | supported | ok\n"]])
    judgement = await support(model, found)
    assert len(model.requests) == 1 and len(judgement.verdicts) == 2
    assert "1. Cedar scored 91% [c1]." in model.requests[0].prompt
    assert "2. Cedar uses a score [c1]." in model.requests[0].prompt


@pytest.mark.asyncio
async def test_the_prompt_keeps_the_writers_prefix_and_sends_markers_not_chunk_text() -> None:
    model = ScriptedModel(streams=[["".join(LINES)]])
    await support(model)
    request = model.requests[0]
    explain = load_explain_prompts(ROOT)
    judge = load_judge_prompts(ROOT)
    writer = build_prompt(
        explain.shared_header,
        PACK_TEXT,
        explain.writer_rules,
        [],
        "How does Cedar store?",
        target_words=350,
        max_output_tokens=1200,
    )
    prefix = PACK_TEXT.rstrip("\n") + "\n\n"
    assert request.system_prompt == writer.system_prompt == explain.shared_header
    assert request.prompt.startswith(prefix) and writer.prompt.startswith(prefix)
    assert judge.support_rules in request.prompt
    # The chunk text lives in the pack only; the sentences carry markers.
    assert request.prompt.count(SOURCES[1]) == 1
    assert "1. Cedar decides what to store with a learned score [c1]." in request.prompt
    assert "paper_id" not in request.prompt
    assert request.max_output_tokens == 60 + 45 * 3 and not request.tools


@pytest.mark.asyncio
async def test_an_answer_that_cites_nothing_makes_no_call() -> None:
    model = ScriptedModel(streams=[[]])
    found = check_answer("Memory helps retrieval. It is useful.", SOURCES)
    judgement = await support(model, found)
    assert judgement.status == SKIPPED and not judgement.verdicts and model.requests == []
    assert citation_support(found, judgement) is None
    empty = await support(model, CitationChecks())
    assert empty.status == SKIPPED and model.requests == []


@pytest.mark.asyncio
async def test_malformed_output_fails_closed_after_exactly_one_call() -> None:
    model = ScriptedModel(streams=[["1 | supported | s\n2 | partial | p\n"]])
    judgement = await support(model)
    assert len(model.requests) == 1
    assert judgement.status == FAILED and judgement.fallback_reason == MALFORMED_OUTPUT
    assert judgement.verdicts == () and citation_support(checks(), judgement) is None


@pytest.mark.asyncio
async def test_a_stopped_or_failing_call_gives_a_failed_judgement_not_an_exception() -> None:
    slow: list[StreamStep] = [0.3, "1 | supported | s\n"]
    model = ScriptedModel(streams=[slow])
    judgement = await judge_support(
        ctx(),
        cast(LLM, model),
        PACK_TEXT,
        checks(),
        context_tokens=1_000_000,
        budget=StreamBudget(first_token=0.05, stall=1.0, total=5.0),
        project_root=ROOT,
    )
    assert judgement.status == FAILED and judgement.fallback_reason == "first_token_timeout"

    class Broken:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            raise NotImplementedError

        async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
            raise ConnectionError("down")
            yield LLMDelta()

    broken = await support(Broken())
    assert broken.status == FAILED and broken.fallback_reason == "ConnectionError"
    no_model = await support(None)
    assert no_model.status == FAILED and no_model.fallback_reason == "model_unavailable"


@pytest.mark.asyncio
async def test_the_judge_cannot_exceed_the_models_window() -> None:
    model = ScriptedModel(streams=[["".join(LINES)]])
    judgement = await judge_support(
        ctx(), cast(LLM, model), PACK_TEXT, checks(), context_tokens=100, project_root=ROOT
    )
    assert judgement.status == FAILED and judgement.fallback_reason == "input_budget"
    assert model.requests == []


def test_citation_support_counts_sentences_that_pass_both_the_checks_and_the_judge() -> None:
    found = check_answer(
        "Cedar scored 91% [c1]. Cedar uses a score [c1]. Accuracy rose to 68% [c2]. "
        "Gates help [c2]. Plain background.",
        SOURCES,
    )
    cited = [s for s in found.sentences if s.markers]
    assert [s.status for s in cited] == ["flagged", "cited_ok", "cited_ok", "cited_ok"]

    def judged(*verdicts: str) -> SupportJudgement:
        return SupportJudgement(
            status=COMPLETE,
            verdicts=tuple(
                SentenceVerdict(i, s.start, s.end, v, "")
                for i, (s, v) in enumerate(zip(cited, verdicts, strict=True), 1)
            ),
        )

    # The flagged sentence never counts, even when the judge liked it; partial counts.
    both = judged("supported", "supported", "partial", "supported")
    assert citation_support(found, both) == 0.75
    one_out = judged("supported", "supported", "supported", "unsupported")
    assert citation_support(found, one_out) == 0.5
    none_left = judged("unsupported", "unsupported", "unsupported", "unsupported")
    assert citation_support(found, none_left) == 0.0
    assert citation_support(found, SupportJudgement(status=FAILED)) is None


class Judge:
    """A pairwise judge that decides from which answer sits in which slot."""

    def __init__(self, decide: str) -> None:
        self.decide = decide
        self.prompts: list[str] = []

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        self.prompts.append(request.prompt)
        answer_a = request.prompt.split("Answer A:\n")[1].split("\n\nAnswer B:")[0]
        if self.decide == "explain":
            yield LLMDelta("A | better.\n" if answer_a == "EXPLAINED" else "B | better.\n")
        elif self.decide == "strict":
            yield LLMDelta("A | better.\n" if answer_a == "QUOTED" else "B | better.\n")
        elif self.decide == "first":
            yield LLMDelta("**A** | I prefer the first.\n")
        elif self.decide == "garbage":
            yield LLMDelta("Both are fine.\n")
        else:
            yield LLMDelta(f"{self.decide}\n")
        yield LLMDelta(finish=True, usage=TokenUsage(10, 5))


async def pair(model: Judge) -> PairwiseJudgement:
    return await judge_pairwise(
        ctx(),
        cast(LLM, model),
        "How does Cedar decide?",
        ["a learned score", "threshold 0.4"],
        "EXPLAINED",
        "QUOTED",
        project_root=ROOT,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decide", "winner", "score"),
    [("explain", "explain", 1.0), ("strict", "strict", 0.0), ("TIE", "tie", 0.5)],
)
async def test_pairwise_judges_both_ways_round_and_agreeing_orders_decide(
    decide: str, winner: str, score: float
) -> None:
    model = Judge(decide)
    result = await pair(model)
    assert (result.status, result.winner, result.score) == (COMPLETE, winner, score)
    # One call per order, with each answer shown first once; no paper in the prompt.
    firsts = sorted(p.split("Answer A:\n")[1].split("\n")[0] for p in model.prompts)
    assert firsts == ["EXPLAINED", "QUOTED"]
    assert all("- a learned score" in p and "- threshold 0.4" in p for p in model.prompts)
    assert all("Role: pairwise judge" in p for p in model.prompts)


@pytest.mark.asyncio
async def test_a_judge_that_always_prefers_the_first_answer_ties() -> None:
    result = await pair(Judge("first"))
    assert result.winner == "tie" and result.score == 0.5
    assert result.as_a == "explain" and result.as_b == "strict"


@pytest.mark.asyncio
async def test_an_unreadable_pairwise_verdict_fails_instead_of_guessing() -> None:
    result = await pair(Judge("garbage"))
    assert result.status == FAILED and result.score is None
    assert result.fallback_reason == MALFORMED_OUTPUT


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("A | x", "A"),
        ("b|y", "B"),
        ("**tie** | z", "TIE"),
        ("Verdict: A | q", "A"),
        ("\n\nTie.", "TIE"),
    ],
)
def test_pairwise_verdicts_are_read_from_the_first_line(text: str, verdict: str) -> None:
    assert parse_pairwise(text)[0] == verdict


@pytest.mark.parametrize("text", ["", "Neither", "AB | x", "I think A is better"])
def test_other_pairwise_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        parse_pairwise(text)


def test_the_repo_judge_prompt_loads_and_a_missing_part_fails_closed(tmp_path: Path) -> None:
    prompts = load_judge_prompts(ROOT)
    assert prompts.version == "judge.v1"
    assert "N | verdict | reason" in prompts.support_rules
    assert "verdict | reason" in prompts.pairwise_rules
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "judge.v1.yaml").write_text("version: 1\nname: judge\n")
    with pytest.raises(ValueError, match="support_rules"):
        load_judge_prompts(tmp_path)
