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
    check_feedback_facts,
    feedback_case,
    fixed_cases,
    generate_cases,
    score_case,
    summarize,
)
from arie_core.harness.reader import EvidenceChunk, Excerpt
from arie_core.harness.trace import AnswerTrace, StageSeconds
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


def test_a_scored_result_keeps_the_answer_trace() -> None:
    row = ROWS[2]
    trace = AnswerTrace(
        seconds=StageSeconds(write=1.5, total=2.0), mode="reviewed_answer", model="m"
    )
    answer = AnswerResult([excerpt(row, 0, len(row.text))], "reviewed_answer", trace=trace)
    assert score_case(fact_case(), ROWS, answer, seconds=2.0).trace == trace


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


def test_an_excerpt_fallback_that_quotes_the_source_passes_but_is_not_a_written_pass() -> None:
    row = ROWS[2]
    fallback = AnswerResult(
        [excerpt(row, 0, len(row.text))], "source_excerpts", fallback_reason="entailment_timeout"
    )
    result = score_case(fact_case(), ROWS, fallback, seconds=3.0)
    # The old scores still count the quote, so old runs stay comparable.
    assert result.passed and result.fact_recall == 1.0
    assert not result.answer_written and result.fact_recall_prose == 0.0
    assert result.passed_written is False
    assert result.citation_support is None and result.win_rate is None
    assert result.requested_mode == "strict"


def test_a_written_answer_is_scored_on_its_claim_text_and_not_its_quotes() -> None:
    row = ROWS[2]
    written = AnswerResult(
        [excerpt(row, 0, len(row.text), text="Accuracy rose from 62% to 68%.")], "reviewed_answer"
    )
    result = score_case(fact_case(), ROWS, written, seconds=2.0, requested_mode="explain")
    assert result.answer_written and result.fact_recall_prose == 1.0
    assert result.passed_written is True and result.requested_mode == "explain"

    # The quote holds both facts but the claim says nothing: retrieval worked, the answer did not.
    vague = AnswerResult(
        [excerpt(row, 0, len(row.text), text="Accuracy improved.")], "reviewed_answer"
    )
    scored = score_case(fact_case(), ROWS, vague, seconds=2.0)
    assert scored.passed and scored.fact_recall == 1.0
    assert scored.fact_recall_prose == 0.0 and scored.passed_written is False

    # Refusal cases expect no written answer, so they have no written score.
    refusal = score_case(
        fixed_cases("p1", ROWS)[0], ROWS, AnswerResult([], "abstention"), seconds=0
    )
    assert refusal.passed is True and refusal.passed_written is None


def test_the_summary_counts_written_passes_fallbacks_and_median_stage_times() -> None:
    row = ROWS[2]

    def traced(total: float, **stages: float) -> AnswerTrace:
        return AnswerTrace(seconds=StageSeconds(total=total, **stages))

    fallback = AnswerResult(
        [excerpt(row, 0, len(row.text))],
        "source_excerpts",
        fallback_reason="entailment_timeout",
        trace=traced(3.0, tier3=2.0),
    )
    full = AnswerResult(
        [excerpt(row, 0, len(row.text), text="Accuracy rose from 62% to 68%.")],
        "reviewed_answer",
        trace=traced(5.0, write=2.0),
    )
    vague = AnswerResult(
        [excerpt(row, 0, len(row.text), text="Accuracy improved.")],
        "reviewed_answer",
        trace=traced(4.0, write=1.0),
    )
    refusal = AnswerResult([], "abstention", fallback_reason="not_in_paper")
    results = [
        score_case(fact_case(), ROWS, answer, seconds=1.0) for answer in (fallback, full, vague)
    ]
    results.append(score_case(fixed_cases("p1", ROWS)[0], ROWS, refusal, seconds=0.1))
    summary = summarize(results)
    assert summary["passed"] == 4 and summary["passed_written"] == 1
    assert summary["pass_written_rate"] == 0.333 and summary["mean_fact_recall_prose"] == 0.333
    # A refusal's reason is shown but is not a fallback.
    assert summary["fallbacks"] == {"entailment_timeout": 1}
    assert summary["median_stage_seconds"] == {
        "plan": 0.0,
        "vector": 0.0,
        "read": 0.0,
        "rerank": 0.0,
        "write": 1.0,
        "tier3": 0.0,
        "review": 0.0,
        "total": 4.0,
    }
    assert summary["citation_support"] is None and summary["win_rate"] is None
    assert summarize([])["median_stage_seconds"] == {}

    # An excerpt answer is a fallback even when no reason was recorded.
    bare = AnswerResult([excerpt(row, 0, len(row.text))], "extractive_summary")
    assert summarize([score_case(fact_case(), ROWS, bare, seconds=1.0)])["fallbacks"] == {
        "unspecified": 1
    }


def test_feedback_facts_must_be_few_and_verbatim_in_the_source_sentence() -> None:
    sentence = "Cedar accuracy   increased from 62% to 68%"
    facts, unit = check_feedback_facts(ROWS, "p1", [" 62% ", "", "68%"], sentence)
    assert facts == ["62%", "68%"] and unit is not None
    # Offsets are exact even though the person's whitespace differed from the paper's.
    assert unit.text == "Cedar accuracy increased from 62% to 68%"
    assert unit.char_start == ROWS[2].char_start + ROWS[2].text.index("Cedar")
    assert check_feedback_facts(ROWS, "p1", ["62%"]) == (["62%"], None)

    for bad_facts in ([], ["", "  "], ["a", "b", "c", "d"]):
        with pytest.raises(ValueError, match="1 to 3 key facts"):
            check_feedback_facts(ROWS, "p1", bad_facts)
    with pytest.raises(ValueError, match="not in this paper"):
        check_feedback_facts(ROWS, "p1", ["62%"], "Accuracy increased from 62% to 68%.")
    with pytest.raises(ValueError, match="not in this paper"):
        check_feedback_facts(ROWS, "other", ["62%"], sentence)
    with pytest.raises(ValueError, match="appear in the source sentence"):
        check_feedback_facts(ROWS, "p1", ["99%"], sentence)


def test_a_feedback_case_with_key_facts_is_scored_like_a_fact_case() -> None:
    row = ROWS[2]
    plain = feedback_case("t1", "How much did accuracy improve?", "wrong")
    assert plain.kind == "feedback" and plain.case_id == "feedback-t1" and not plain.key_facts
    answer = AnswerResult(
        [excerpt(row, 0, len(row.text), text="Accuracy rose from 62% to 68%.")], "reviewed_answer"
    )
    assert score_case(plain, ROWS, answer, seconds=1.0).passed is None

    _, source = check_feedback_facts(ROWS, "p1", ["62%", "68%"], row.text)
    case = feedback_case("t1", plain.question, "wrong", ["62%", "68%"], source)
    assert case.kind == "fact" and case.origin == "feedback" and case.case_id == plain.case_id
    assert (case.chunk_id, case.char_start, case.char_end) == (
        row.chunk_id,
        row.char_start,
        row.char_end,
    )
    scored = score_case(case, ROWS, answer, seconds=1.0)
    assert scored.passed and scored.evidence_hit and scored.passed_written is True
    # Without a source sentence the case is scored on recall alone.
    unanchored = feedback_case("t2", plain.question, "", ["62%", "68%"])
    assert score_case(unanchored, ROWS, answer, seconds=1.0).evidence_hit is False
    assert score_case(unanchored, ROWS, answer, seconds=1.0).passed is True


def test_the_unanswerable_case_is_skipped_when_the_paper_reports_energy() -> None:
    energy = EvidenceChunk("p1", "p1:e", "Training used 42 kWh of energy.", 9, 0, 31, "Results")
    assert len(fixed_cases("p1", [*ROWS, energy])) == 2
