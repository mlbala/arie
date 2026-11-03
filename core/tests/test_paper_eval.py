"""Per-paper evaluation: questions come from the paper's own sentences and scoring is exact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from arie_api.runtime import context
from arie_core.contracts import LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.answer import AnswerResult
from arie_core.harness.paper_eval import (
    PaperEvalCase,
    catalog,
    fixed_cases,
    generate_cases,
    score_case,
    summarize,
)
from arie_core.harness.reader import EvidenceChunk, Excerpt
from arie_core.plugin_api import LLM, Ctx

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = (
    ("Abstract", "We propose Cedar, a retrieval model that uses a gated memory bank for search."),
    (
        "Methods",
        "The method retrieves evidence from a local index using four CPU workers in total.",
    ),
    (
        "Results",
        "On 120 English documents, Cedar accuracy increased from 62% to 68% over baselines.",
    ),
    ("Limitations", "We did not evaluate multilingual documents or speech data in this paper."),
    ("References", "Smith et al. 2020. Proceedings of the Retrieval Conference, pages 1-10."),
)
ROWS = [
    EvidenceChunk("p1", f"p1:{i}", text, i + 1, 100 * i, 100 * i + len(text), section)
    for i, (section, text) in enumerate(SENTENCES)
]


class QuestionModel:
    def __init__(self, questions: list[dict[str, object]]) -> None:
        self.questions = questions
        self.payload: dict[str, object] = {}

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        del ctx
        self.payload = cast(dict[str, object], json.loads(request.prompt))
        return LLMResponse(json.dumps({"questions": self.questions}), TokenUsage(200, 120))


def test_catalog_offers_exact_sentences_and_skips_references() -> None:
    units = catalog(ROWS)
    assert [unit.section for unit in units] == ["Abstract", "Methods", "Results", "Limitations"]
    for unit in units:
        row = next(row for row in ROWS if row.chunk_id == unit.chunk_id)
        assert (
            row.text[unit.char_start - row.char_start : unit.char_end - row.char_start] == unit.text
        )


@pytest.mark.asyncio
async def test_only_questions_whose_key_facts_are_verbatim_in_the_source_survive() -> None:
    model = QuestionModel([])
    # Catalog IDs follow catalog order, which is deterministic for these rows.
    ids = {f"S{i + 1}": unit for i, unit in enumerate(catalog(ROWS))}
    results_id = next(key for key, unit in ids.items() if unit.section == "Results")
    methods_id = next(key for key, unit in ids.items() if unit.section == "Methods")
    abstract_id = next(key for key, unit in ids.items() if unit.section == "Abstract")
    questions: list[dict[str, object]] = [
        {"source": results_id, "question": "How much did accuracy improve?", "key_facts": ["68%"]},
        # Invented fact: "eight" is not in the sentence, so the case is dropped.
        {"source": methods_id, "question": "How many workers are used?", "key_facts": ["eight"]},
        # Unknown source ID.
        {"source": "S99", "question": "What is the dataset size?", "key_facts": ["120"]},
        # Restates the sentence instead of asking about it.
        {
            "source": abstract_id,
            "question": "We propose Cedar, a retrieval model that uses a gated memory bank",
            "key_facts": ["Cedar"],
        },
    ]
    model.questions = questions
    cases = await generate_cases(
        context("generate", 30.0), "p1", "Cedar", ROWS, cast(LLM, model), project_root=ROOT
    )
    assert [case.question for case in cases] == ["How much did accuracy improve?"]
    case = cases[0]
    assert case.kind == "fact" and case.answer_quote == SENTENCES[2][1] and case.page_start == 3
    assert "References" not in json.dumps(model.payload)


@pytest.mark.asyncio
async def test_generation_fails_closed_when_nothing_verifies() -> None:
    model = QuestionModel([{"source": "S1", "question": "What is it?", "key_facts": ["nope"]}])
    with pytest.raises(ValueError, match="verified"):
        await generate_cases(
            context("generate", 30.0), "p1", "Cedar", ROWS, cast(LLM, model), project_root=ROOT
        )


def excerpt(row: EvidenceChunk, start: int, end: int, text: str = "") -> Excerpt:
    quote = row.text[start:end]
    return Excerpt(
        text=text or quote,
        quote=quote,
        page_start=row.page_start,
        char_start=row.char_start + start,
        char_end=row.char_start + end,
        chunk_id=row.chunk_id,
        label="",
        paragraph=0,
        checks=("tier_1",),
    )


def fact_case() -> PaperEvalCase:
    row = ROWS[2]
    return PaperEvalCase(
        case_id="c1",
        question="How much did accuracy improve?",
        kind="fact",
        origin="generated",
        answer_quote=row.text,
        chunk_id=row.chunk_id,
        page_start=row.page_start,
        char_start=row.char_start,
        char_end=row.char_end,
        key_facts=["62%", "68%"],
    )


def test_scoring_checks_offsets_evidence_and_key_facts() -> None:
    row = ROWS[2]
    good = AnswerResult([excerpt(row, 0, len(row.text))], "reviewed_answer", model_calls=3)
    result = score_case(fact_case(), ROWS, good, seconds=4.2)
    assert result.passed and result.grounded and result.evidence_hit and result.fact_recall == 1.0
    assert result.cited_pages == [3]

    # A quote that does not match its offsets is never grounded, so the case fails.
    bad = AnswerResult(
        [
            Excerpt(
                text="Accuracy reached 99%.",
                quote="Accuracy reached 99%.",
                page_start=3,
                char_start=row.char_start,
                char_end=row.char_start + 21,
                chunk_id=row.chunk_id,
                label="",
                paragraph=0,
                checks=(),
            )
        ],
        "reviewed_answer",
    )
    assert score_case(fact_case(), ROWS, bad, seconds=1.0).grounded is False
    assert score_case(fact_case(), ROWS, bad, seconds=1.0).passed is False
    abstained = score_case(fact_case(), ROWS, AnswerResult([], "abstention"), seconds=1.0)
    assert abstained.abstained and abstained.passed is False


def test_abstain_and_feedback_cases_and_summary() -> None:
    fixed = fixed_cases("p1", ROWS)
    assert {case.kind for case in fixed} == {"abstain"} and len(fixed) == 3
    refused = score_case(fixed[0], ROWS, AnswerResult([], "abstention"), seconds=0.1)
    assert refused.passed is True
    review = PaperEvalCase(case_id="f", question="Why?", kind="feedback", origin="feedback")
    reviewed = score_case(review, ROWS, AnswerResult([], "abstention"), seconds=0.1)
    assert reviewed.passed is None
    row = ROWS[2]
    answered = score_case(
        fact_case(), ROWS, AnswerResult([excerpt(row, 0, len(row.text))], "x"), seconds=2.0
    )
    summary = summarize([answered, refused, reviewed])
    assert summary["scored"] == 2 and summary["passed"] == 2 and summary["review"] == 1
    assert summary["abstain_correct_rate"] == 1.0 and summary["mean_fact_recall"] == 1.0


def test_the_unanswerable_case_is_skipped_when_the_paper_reports_energy() -> None:
    energy = EvidenceChunk("p1", "p1:e", "Training used 42 kWh of energy.", 9, 0, 31, "Results")
    assert len(fixed_cases("p1", [*ROWS, energy])) == 2
