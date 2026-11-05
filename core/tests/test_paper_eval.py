"""Per-paper evaluation: questions come from the paper's own sentences and scoring is exact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from arie_api.runtime import context
from arie_core.contracts import LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.answer import AnswerResult
from arie_core.harness.citations import CitationChecks, check_answer
from arie_core.harness.explain import ExplainCitation, ExplainResult
from arie_core.harness.judge import (
    COMPLETE,
    FAILED,
    SKIPPED,
    PairwiseJudgement,
    SentenceVerdict,
    SupportJudgement,
)
from arie_core.harness.paper_eval import (
    PaperEvalCase,
    PaperEvalResult,
    answer_text,
    catalog,
    check_feedback_facts,
    feedback_case,
    fixed_cases,
    generate_cases,
    score_case,
    score_explained,
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


CACHED = TokenUsage(1000, 50, 800, 200)


def explained(
    text: str,
    *,
    mode: str = "explain",
    cited: tuple[int, ...] = (),
    unknown: tuple[int, ...] = (),
    usage: TokenUsage = CACHED,
    first_token: float | None = 1.5,
    reason: str = "",
    checks: CitationChecks | None = None,
) -> ExplainResult:
    return ExplainResult(
        text=text,
        mode=mode,
        depth="standard",
        citations=tuple(ExplainCitation(number, f"p1:{number}") for number in cited),
        unknown_markers=unknown,
        usage=usage,
        fallback_reason=reason,
        trace=AnswerTrace(tokens=usage, mode=mode, fallback_reason=reason),
        first_token_seconds=first_token,
        checks=checks or CitationChecks(),
    )


def test_an_explained_answer_is_scored_on_its_text_and_the_chunks_it_cites() -> None:
    answer = explained("Accuracy rose from 62% to 68% [c2].", cited=(2,))
    scored = score_explained(fact_case(), ROWS, answer, seconds=4.2)
    assert scored.requested_mode == "explain" and scored.mode == "explain"
    assert scored.passed is True and scored.passed_written is True
    assert scored.grounded and scored.evidence_hit and scored.answer_written
    assert scored.fact_recall == scored.fact_recall_prose == 1.0
    assert scored.cited_pages == [3] and scored.claims == 1 and scored.model_calls == 1
    assert scored.cache_hit_rate == 0.8 and scored.first_token_seconds == 1.5
    assert scored.answer.startswith("Accuracy rose") and scored.trace.tokens.cache_hit == 800

    # Right facts but no citation of the source chunk: the old pass needs the facts alone.
    uncited = score_explained(fact_case(), ROWS, explained("It went from 62% to 68%."), seconds=1)
    assert uncited.evidence_hit is False and uncited.passed is True
    # Neither the facts nor the source chunk: not passed.
    wrong = score_explained(
        fact_case(), ROWS, explained("It improved a lot [c0]", cited=(0,)), seconds=1
    )
    assert wrong.passed is False and wrong.passed_written is False and wrong.fact_recall == 0.0


def test_an_invented_marker_or_a_foreign_chunk_makes_an_explained_answer_ungrounded() -> None:
    invented = score_explained(
        fact_case(), ROWS, explained("It rose from 62% to 68% [c99].", unknown=(99,)), seconds=1
    )
    assert invented.grounded is False and invented.passed is False
    foreign = score_explained(
        fact_case(), ROWS, explained("It rose from 62% to 68% [c7].", cited=(7,)), seconds=1
    )
    assert foreign.grounded is False and foreign.passed_written is False


def test_failed_partial_and_refused_explanations_are_scored_by_what_arrived() -> None:
    failed = explained("", mode="explain_failed", reason="model_unavailable", first_token=None)
    result = score_explained(fact_case(), ROWS, failed, seconds=0.2)
    assert result.passed is False and result.abstained and not result.answer_written
    assert result.fallback_reason == "model_unavailable" and result.first_token_seconds is None
    abstain = fixed_cases("p1", ROWS)[0]
    # A call that failed is not an abstention; a refusal by the guard is.
    assert score_explained(abstain, ROWS, failed, seconds=0.2).passed is False
    refused = explained("", mode="abstention", reason="prompt_injection", first_token=None)
    assert score_explained(abstain, ROWS, refused, seconds=0.2).passed is True
    answered = explained("Cedar is a model [c0].", cited=(0,))
    assert score_explained(abstain, ROWS, answered, seconds=0.2).passed is False

    partial = explained("Accuracy rose", mode="explain_partial", reason="stall_timeout")
    kept = score_explained(fact_case(), ROWS, partial, seconds=3)
    assert kept.answer_written and kept.passed_written is False and kept.fallback_reason
    review = PaperEvalCase(case_id="f", question="Why?", kind="feedback", origin="feedback")
    assert score_explained(review, ROWS, answered, seconds=1).passed is None


def test_the_summary_reports_first_token_seconds_and_cache_hit_rate_for_explain_runs() -> None:
    results = [
        score_explained(fact_case(), ROWS, explained("62% and 68% [c2]", cited=(2,)), seconds=3),
        score_explained(
            fact_case(),
            ROWS,
            explained(
                "62% and 68% [c2]",
                cited=(2,),
                usage=TokenUsage(1000, 50, 400, 600),
                first_token=3.5,
            ),
            seconds=5,
        ),
        score_explained(
            fact_case(),
            ROWS,
            explained(
                "", mode="explain_failed", reason="x", first_token=None, usage=TokenUsage(0, 0)
            ),
            seconds=1,
        ),
    ]
    summary = summarize(results)
    assert summary["median_first_token_seconds"] == 3.5
    assert summary["mean_cache_hit_rate"] == 0.6 and summary["passed_written"] == 2
    assert summary["fallbacks"] == {"x": 1}
    strict = summarize([score_case(fact_case(), ROWS, AnswerResult([], "abstention"), seconds=1)])
    assert strict["median_first_token_seconds"] is None and strict["mean_cache_hit_rate"] is None


def test_explain_results_carry_the_citation_check_rates_and_the_summary_averages_them() -> None:
    def scored(text: str, sources: dict[int, str]) -> PaperEvalResult:
        answer = explained(text, cited=(2,), checks=check_answer(text, sources))
        return score_explained(fact_case(), ROWS, answer, seconds=1)

    # Marker 2 is not in this pack, so the one cited sentence is flagged; one number is uncited.
    first = scored("Accuracy rose [c2]. It has 12 layers. Attention weighs tokens.", {})
    assert first.citation_valid_rate == 0.0
    assert first.background_share == pytest.approx(0.333, abs=0.001)
    second = scored("Accuracy rose [c2]. Attention weighs tokens.", {2: "Accuracy rose."})
    assert second.citation_valid_rate == 1.0 and second.background_share == 0.5
    # No cited sentence and no sentence at all: nothing to measure, so unset.
    plain = scored("Attention weighs tokens.", {})
    assert plain.citation_valid_rate is None and plain.background_share == 1.0
    empty = scored("", {})
    assert empty.citation_valid_rate is None and empty.background_share is None

    summary = summarize([first, second, plain, empty])
    assert summary["mean_citation_valid_rate"] == 0.5
    assert summary["mean_background_share"] == pytest.approx(0.611, abs=0.001)
    strict = summarize([score_case(fact_case(), ROWS, AnswerResult([], "abstention"), seconds=1)])
    assert strict["mean_citation_valid_rate"] is None and strict["mean_background_share"] is None


def _judged(found: CitationChecks, *verdicts: str, status: str = COMPLETE) -> SupportJudgement:
    cited = [sentence for sentence in found.sentences if sentence.markers]
    return SupportJudgement(
        status=status,
        verdicts=tuple(
            SentenceVerdict(number, sentence.start, sentence.end, verdict, "")
            for number, (sentence, verdict) in enumerate(zip(cited, verdicts, strict=True), 1)
        ),
    )


def test_citation_support_and_win_rate_come_from_the_judge_and_never_change_passed() -> None:
    text = "Accuracy rose from 62% to 68% [c2]. It also reached 99% [c2]."
    found = check_answer(text, {2: "Accuracy rose from 62% to 68% on the benchmark."})
    answer = explained(text, cited=(2,), checks=found)
    plain = score_explained(fact_case(), ROWS, answer, seconds=1)
    assert plain.citation_support is None and plain.win_rate is None and plain.judge_status == ""

    supported = _judged(found, "supported", "supported")
    judged = score_explained(
        fact_case(),
        ROWS,
        answer,
        seconds=1,
        judgement=supported,
        pairwise=PairwiseJudgement(status=COMPLETE, winner="explain", score=1.0),
    )
    # The invented 99% is flagged by the deterministic checks, so the judge cannot rescue it.
    assert [s.status for s in found.sentences] == ["cited_ok", "flagged"]
    assert judged.citation_support == 0.5 and judged.win_rate == 1.0
    assert judged.judge_status == COMPLETE
    assert judged.passed == plain.passed and judged.passed_written == plain.passed_written

    harsh = _judged(found, "unsupported", "supported")
    unsupported = score_explained(fact_case(), ROWS, answer, seconds=1, judgement=harsh)
    assert unsupported.citation_support == 0

    failed = score_explained(
        fact_case(),
        ROWS,
        answer,
        seconds=1,
        judgement=SupportJudgement(status=FAILED, fallback_reason="malformed_output"),
        pairwise=PairwiseJudgement(status=FAILED, fallback_reason="first_token_timeout"),
    )
    assert failed.citation_support is None and failed.win_rate is None
    assert failed.judge_status == FAILED


def test_the_summary_averages_judged_cases_and_counts_failures_apart() -> None:
    text = "Accuracy rose from 62% to 68% [c2]."
    found = check_answer(text, {2: "Accuracy rose from 62% to 68%."})
    answer = explained(text, cited=(2,), checks=found)

    def result(
        judgement: SupportJudgement | None, pairwise: PairwiseJudgement | None
    ) -> PaperEvalResult:
        return score_explained(
            fact_case(), ROWS, answer, seconds=1, judgement=judgement, pairwise=pairwise
        )

    win = PairwiseJudgement(status=COMPLETE, winner="explain", score=1.0)
    tie = PairwiseJudgement(status=COMPLETE, winner="tie", score=0.5)
    summary = summarize(
        [
            result(_judged(found, "supported"), win),
            result(_judged(found, "unsupported"), tie),
            result(SupportJudgement(status=FAILED), PairwiseJudgement(status=FAILED)),
            result(SupportJudgement(status=SKIPPED), None),
            result(None, None),
        ]
    )
    assert summary["citation_support"] == 0.5  # (1.0 + 0.0) / 2; failed and skipped are left out
    assert summary["win_rate"] == 0.75 and summary["win_cases"] == 2
    assert summary["judge_failed"] == 1
    strict = summarize([score_case(fact_case(), ROWS, AnswerResult([], "abstention"), seconds=1)])
    assert strict["citation_support"] is None and strict["win_rate"] is None
    assert strict["win_cases"] == 0 and strict["judge_failed"] == 0


def test_a_strict_answer_is_judged_on_everything_it_wrote_not_the_saved_cut() -> None:
    long = Excerpt("w " * 900, "q" * 100, 3, 0, 100, "p1:2")
    assert answer_text(AnswerResult([long], "verified_answer")) == "w " * 900
    assert answer_text(AnswerResult([], "abstention")) == ""
