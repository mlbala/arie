"""Reranked evidence selection, conditional review, and Tier 3 premise restating."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import msgspec
import pytest
from arie_api.runtime import context
from arie_core.contracts import LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.answer import compose_answer, review_optional
from arie_core.harness.intent import DialogueTurn, plan_answer
from arie_core.harness.orchestrator import answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.rerank import rerank_reading
from arie_core.harness.research import read_for_answer
from arie_core.harness.verifier import attributed_premise
from arie_core.plugin_api import LLM, Ctx

SECTIONS = [
    ("Abstract", "We propose Cedar, a retrieval model using a gated memory bank."),
    ("Introduction", "Retrieval models are widely used in question answering systems today."),
    ("Methods", "The gated memory bank stores passages and the gate selects which to read."),
    ("Methods", "Training uses four CPU workers and a local index of documents."),
    ("Results", "On 120 English documents, Cedar accuracy increased from 62% to 68%."),
    ("Results", "Ablating the gate reduced accuracy to 63% on the same documents."),
    ("Limitations", "We did not evaluate multilingual documents or speech data."),
    ("References", "Smith et al. showed retrieval models are widely used in 2020."),
]
FOREIGN = EvidenceChunk(
    "other", "foreign", "The foreign model was evaluated with 99% accuracy.", 1, 0, 50, "Results"
)


def rows() -> list[EvidenceChunk]:
    return [
        EvidenceChunk("paper", f"c{i}", text, i + 1, i * 100, i * 100 + len(text), section)
        for i, (section, text) in enumerate(SECTIONS)
    ]


def test_overview_pool_offers_every_section_but_no_bibliography_or_other_paper() -> None:
    plan = plan_answer("What is this paper about?")
    reading = read_for_answer(context("pool"), "paper", plan, [*rows(), FOREIGN])
    sections = {row.section for row in reading.pool}
    assert {"Abstract", "Introduction", "Methods", "Results", "Limitations"} <= sections
    assert "References" not in sections
    assert all(row.paper_id == "paper" for row in reading.pool)


@pytest.mark.asyncio
async def test_reranker_scores_choose_the_passages_within_the_bound_paper() -> None:
    plan = plan_answer("Does removing the gate hurt performance?")
    reading = read_for_answer(context("rerank"), "paper", plan, [*rows(), FOREIGN], ["foreign"])
    seen: dict[str, object] = {}

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        seen.update(query=query, passages=list(passages))
        return [0.99 if passage.startswith("Ablating") else 0.01 for passage in passages]

    result = await rerank_reading(context("rerank"), "paper", plan, reading, rerank)
    assert result.reranked and result.rows[0].chunk_id == "c5"
    assert "removing the gate" in str(seen["query"])
    assert all("foreign" not in passage for passage in cast(list[str], seen["passages"]))
    assert all(row.paper_id == "paper" for row in result.rows)
    assert result.actions == reading.actions, "reranking is not an LLM-visible tool call"


@pytest.mark.asyncio
async def test_vector_hits_without_keyword_overlap_reach_the_reranker() -> None:
    plan = plan_answer("How well does it perform?")
    reading = read_for_answer(context("vector"), "paper", plan, rows(), ["c4", "c5"])

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        del query
        return [0.9 if "62% to 68%" in passage else 0.1 for passage in passages]

    result = await rerank_reading(context("vector"), "paper", plan, reading, rerank)
    assert result.reranked and result.rows[0].chunk_id == "c4"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "wrong_count"])
async def test_reranker_failure_keeps_the_keyword_and_vector_reading(failure: str) -> None:
    plan = plan_answer("What is this paper about?")
    reading = read_for_answer(context("broken"), "paper", plan, rows())

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        del query
        if failure == "error":
            raise RuntimeError("Reranker is not ready")
        return [0.5] * (len(passages) - 1)

    result = await rerank_reading(context("broken"), "paper", plan, reading, rerank)
    assert result == reading and not result.reranked


@pytest.mark.asyncio
async def test_reranker_is_skipped_when_the_answer_needs_the_remaining_time() -> None:
    plan = plan_answer("What is this paper about?")
    reading = read_for_answer(context("late"), "paper", plan, rows())
    calls: list[str] = []

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        calls.append(query)
        return [0.5] * len(passages)

    result = await rerank_reading(context("late", 10.0), "paper", plan, reading, rerank)
    assert not calls and not result.reranked


@pytest.mark.asyncio
async def test_follow_up_prefers_passages_not_already_cited() -> None:
    plan = plan_answer("tell me more", [DialogueTurn("What were the results?", ("c4",))])
    assert plan.follow_up and plan.previous_source_ids == ("c4",)
    reading = read_for_answer(context("more"), "paper", plan, rows())

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        del query
        return [0.99 if "62% to 68%" in passage else 0.5 for passage in passages]

    result = await rerank_reading(context("more"), "paper", plan, reading, rerank)
    assert result.reranked and result.rows[0].chunk_id != "c4"


@pytest.mark.asyncio
async def test_orchestrator_reranks_before_writing() -> None:
    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        del query
        return [float(i) / len(passages) for i in range(len(passages))]

    run = await answer_question(
        context("orchestrated"), "paper", "What is this paper about?", rows(), rerank=rerank
    )
    assert run.reading.reranked and run.answer.excerpts
    assert all(item.chunk_id != "c7" for item in run.answer.excerpts)


class Writer:
    """Draft claims citing passages by substring, then approve everything as the reviewer."""

    def __init__(self, claims: list[tuple[str, str]]) -> None:
        self.claims = claims
        self.requests: list[LLMRequest] = []

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        del ctx
        self.requests.append(request)
        payload = cast(dict[str, object], json.loads(request.prompt))
        if "source_passages" in payload:
            passages = cast(list[dict[str, object]], payload["source_passages"])
            result: object = {
                "claims": [
                    {
                        "text": text,
                        "source": next(
                            i for i, row in enumerate(passages) if needle in str(row["text"])
                        ),
                        "paragraph": 0,
                    }
                    for text, needle in self.claims
                ]
            }
        else:
            claims = cast(list[object], payload["claims"])
            aspects = cast(list[object], payload["required_aspects"])
            result = {
                "supported": list(range(len(claims))),
                "answers_question": True,
                "adequate_detail": True,
                "covered": list(range(len(aspects))),
            }
        return LLMResponse(json.dumps(result), TokenUsage(50, 40))


WORKERS = ("Training uses four CPU workers.", "four CPU workers")


def scored(*scores: float):  # noqa: ANN201 - test helper returning an async scorer
    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return list(scores[: len(pairs)])

    return entail


@pytest.mark.asyncio
async def test_strongly_entailed_direct_answer_skips_the_reviewer() -> None:
    writer = Writer([WORKERS])
    plan = plan_answer("How many CPU workers are used?")
    result = await compose_answer(
        context("direct"), "paper", plan.question, rows(), cast(LLM, writer),
        plan=plan, entail=scored(0.97),
    )  # fmt: skip
    assert result.mode == "entailed_answer" and result.model_calls == 1
    assert result.excerpts[0].checks == ("tier_1", "tier_3_entailment")
    assert not result.answer_relevant, "relevance was not judged by the reviewer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "scores", "change"),
    [
        ("What is this paper about?", (0.97,), {}),  # overviews are always reviewed
        ("How many CPU workers are used?", (0.7,), {}),  # entailed, but not strongly
        ("How many CPU workers are used?", (0.97,), {"detail_requested": True}),
        ("How many CPU workers are used?", (0.97,), {"aspects": ("workers", "index")}),
        ("How many CPU workers are used?", (0.97,), {"task": "comparison"}),
    ],
)
async def test_other_answers_are_still_reviewed(
    question: str, scores: tuple[float, ...], change: dict[str, object]
) -> None:
    writer = Writer([WORKERS])
    plan = msgspec.structs.replace(plan_answer(question), **change)
    result = await compose_answer(
        context("reviewed"), "paper", plan.question, rows(), cast(LLM, writer),
        plan=plan, entail=scored(*scores),
    )  # fmt: skip
    assert result.mode in {"grounded_summary", "reviewed_answer"}
    assert result.model_calls == 2 and "llm_review" in result.excerpts[0].checks


@pytest.mark.asyncio
async def test_a_tier_3_rejection_sends_the_remaining_claims_to_review() -> None:
    writer = Writer([WORKERS, ("Training uses eight CPU workers.", "four CPU workers")])
    plan = plan_answer("How many CPU workers are used?")
    result = await compose_answer(
        context("partial"), "paper", plan.question, rows(), cast(LLM, writer),
        plan=plan, entail=scored(0.97, 0.01),
    )  # fmt: skip
    assert result.mode == "reviewed_answer" and result.model_calls == 2
    assert [item.text for item in result.excerpts] == [WORKERS[0]]


@pytest.mark.asyncio
async def test_one_passage_may_support_several_claims() -> None:
    writer = Writer(
        [
            ("Accuracy started at 62% on 120 English documents.", "62% to 68%"),
            ("Cedar accuracy reached 68% on 120 English documents.", "62% to 68%"),
        ]
    )
    plan = plan_answer("What accuracy was observed?")
    result = await compose_answer(
        context("split"), "paper", plan.question, rows(), cast(LLM, writer), plan=plan
    )
    assert len(result.excerpts) == 2
    assert result.excerpts[0].quote == result.excerpts[1].quote


@pytest.mark.asyncio
async def test_writer_finding_no_evidence_abstains_instead_of_showing_excerpts() -> None:
    plan = plan_answer("What is the training energy usage in kilowatt-hours?")
    result = await compose_answer(
        context("absent"), "paper", plan.question, rows(), cast(LLM, Writer([])), plan=plan
    )
    assert result.mode == "abstention" and not result.excerpts
    assert result.fallback_reason == "not_in_paper" and result.model_calls == 1


def test_premise_restates_author_voice_without_touching_facts() -> None:
    assert attributed_premise("We propose Cedar; our model beats us by 6%.") == (
        "The authors propose Cedar; the authors' model beats the authors by 6%."
    )
    assert attributed_premise("Western data did not improve.") == "Western data did not improve."


def test_review_is_optional_only_for_strong_direct_single_point_answers() -> None:
    plan = plan_answer("How many CPU workers are used?")
    assert review_optional(plan, [0.95, 0.9])
    assert not review_optional(plan, [0.95, 0.89])
    assert not review_optional(plan, [])
