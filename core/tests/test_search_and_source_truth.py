from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from arie_api.runtime import ResearchRuntime, context
from arie_core.contracts import LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.answer import compose_answer, preserves_anchors
from arie_core.harness.reader import EvidenceChunk
from arie_core.plugin_api import LLM, Ctx
from arie_core.registry.manager import ArieConfig
from arie_core.search import score_paper

ROOT = Path(__file__).resolve().parents[2]


def runtime_at(tmp_path: Path) -> ResearchRuntime:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
        reranker={"use": "none"},
    )
    return ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))


def paper(identifier: str, title: str, published: str) -> dict[str, Any]:
    return dict(
        id=identifier,
        title=title,
        abstract="Research",
        published_at=published,
        source="arxiv",
        status="INDEXED",
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        pages=[],
        chunks=[],
    )


@pytest.mark.parametrize(
    "query",
    [
        "attention is all you need",
        "ATTENTION IS ALL YOU NEED",
        "attenti all you need",
        "Attentin All you Need",
        "attention need",
        "attenti all you need&#x20;",
    ],
)
def test_title_search_handles_case_missing_words_prefixes_and_typos(query: str) -> None:
    score, reason = score_paper(query, "Attention Is All You Need", "A Transformer model.")
    assert score >= 80 and reason in {"Title match", "Similar title"}
    assert score > score_paper(query, "Other research", "Attention is all you need.")[0]


def test_search_supports_unicode_and_rejects_unrelated_matches() -> None:
    assert score_paper("CAFÉ", "Café models", "")[0] > 0
    assert score_paper("quantum bananas", "Attention Is All You Need", "Language models.")[0] == 0
    assert score_paper("agents", "Planning research", "", "Agents")[1] == "Topic match"


@pytest.mark.asyncio
async def test_search_checks_all_pages_and_keeps_filters(tmp_path: Path) -> None:
    runtime: ResearchRuntime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        # The desired older paper appears after the 200-row metadata scan boundary.
        for i in range(205):
            await runtime.save_paper(
                paper(f"a-{i:03}", "Recent neural methods", "2026-10-01T00:00:00Z")
            )
        await runtime.save_paper(
            paper("z-attention", "Attention Is All You Need", "2017-06-12T00:00:00Z")
        )
        result = await runtime.list_papers(search="attenti all you need")
        assert result["items"][0]["id"] == "z-attention"
        assert not result["has_more"]
        assert not (await runtime.list_papers(search="ATTENTION", month="2026-10"))["items"]
    finally:
        runtime.close()


class ScriptedModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        del ctx
        self.requests.append(request)
        assert not request.tools
        return LLMResponse(json.dumps(self.responses.pop(0)), TokenUsage(1, 30))

    async def stream(self, ctx: Ctx, request: LLMRequest):
        del ctx, request
        yield


SOURCE = (
    "We propose Cedar, a retrieval model that uses a gated memory bank. "
    "On the Cedar benchmark, accuracy increased from 62% to 68%. "
    "We did not evaluate multilingual documents. "
    "The findings may not generalize beyond this small corpus."
)
ROW = EvidenceChunk("chosen", "source", SOURCE, 2, 120, 120 + len(SOURCE), "Methods and results")


@pytest.mark.asyncio
async def test_summary_is_rephrased_reviewed_and_cited_to_exact_saved_offsets() -> None:
    text = "Cedar retrieves evidence with a gated memory bank."
    model = ScriptedModel(
        [
            {
                "claims": [
                    {
                        "text": text,
                        "quote": SOURCE.split(". ")[0] + ".",
                        "source": 0,
                        "aspect": "Method",
                    }
                ]
            },
            {"supported": [0], "answers_question": True},
        ]
    )
    foreign = EvidenceChunk(
        "other", "foreign", "Secret foreign paper data must never leak.", 1, 0, 42
    )
    result = await compose_answer(
        context("summary"), "chosen", "What is the paper about?", [foreign, ROW], cast(LLM, model)
    )
    assert result.mode == "grounded_summary" and result.model_calls == 2
    assert result.reviewed_claims == 1 and result.excerpts[0].text == text
    assert result.excerpts[0].text != result.excerpts[0].quote
    assert result.excerpts[0].checks == ("tier_1", "llm_review")
    for item in result.excerpts:
        assert SOURCE[item.char_start - 120 : item.char_end - 120] == item.quote
    assert "Secret foreign" not in " ".join(request.prompt for request in model.requests)
    assert result.input_tokens <= 12000 and result.output_tokens <= 700
    assert sum(request.max_output_tokens for request in model.requests) == 700


def drafted(text: str) -> ScriptedModel:
    return ScriptedModel(
        [
            {"claims": [{"text": text, "quote": SOURCE.split(". ")[0] + ".", "source": 0}]},
            {"supported": [0], "answers_question": True},
        ]
    )


@pytest.mark.asyncio
async def test_tier_3_entailment_marks_claims_that_reach_the_reviewer() -> None:
    seen: list[tuple[str, str]] = []

    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        seen.extend(pairs)
        return [0.97 for _ in pairs]

    text = "Cedar retrieves evidence with a gated memory bank."
    result = await compose_answer(
        context("tier3"),
        "chosen",
        "What is the paper about?",
        [ROW],
        cast(LLM, drafted(text)),
        entail=entail,
    )
    assert result.mode == "grounded_summary"
    assert result.excerpts[0].checks == ("tier_1", "llm_review", "tier_3_entailment")
    # The premise is the quote in the third person: "We propose" -> "The authors propose".
    quote = SOURCE.split(". ")[0] + "."
    assert quote.startswith("We propose")
    assert seen == [(quote.replace("We propose", "The authors propose", 1), text)]


@pytest.mark.parametrize(
    ("scores", "reason"),
    [([0.02], "entailment_rejected"), ([0.4, 0.9], "entailment_unavailable")],
)
@pytest.mark.asyncio
async def test_tier_3_rejection_or_malformed_scores_skip_the_reviewer(
    scores: list[float], reason: str
) -> None:
    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        del pairs
        return scores

    model = drafted("Cedar retrieves evidence without any memory component at all.")
    result = await compose_answer(
        context("tier3-reject"),
        "chosen",
        "Summarize this paper",
        [ROW],
        cast(LLM, model),
        entail=entail,
    )
    assert result.mode == "extractive_summary" and result.fallback_reason == reason
    assert result.model_calls == 1, "a rejected draft must not be sent for review"
    assert all(item.checks == ("tier_1",) for item in result.excerpts)


@pytest.mark.asyncio
async def test_unavailable_entailment_model_abstains_to_exact_excerpts() -> None:
    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        raise RuntimeError("Entailment model is not ready")

    result = await compose_answer(
        context("tier3-down"),
        "chosen",
        "Summarize this paper",
        [ROW],
        cast(LLM, drafted("Cedar retrieves evidence with a gated memory bank.")),
        entail=entail,
    )
    assert result.fallback_reason == "entailment_unavailable"
    assert result.mode == "extractive_summary" and result.excerpts
    assert all("tier_3_entailment" not in item.checks for item in result.excerpts)


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        (
            "Cedar achieved 99% accuracy.",
            "On the Cedar benchmark, accuracy increased from 62% to 68%.",
        ),
        ("We evaluated multilingual documents.", "We did not evaluate multilingual documents."),
        (
            "The findings generalize beyond this corpus.",
            "The findings may not generalize beyond this small corpus.",
        ),
        (
            "Cedar always retrieves perfect results.",
            "Cedar retrieves evidence using a gated memory bank.",
        ),
    ],
)
def test_deterministic_guards_reject_changed_facts(text: str, quote: str) -> None:
    assert not preserves_anchors(text, quote)


@pytest.mark.asyncio
async def test_fabricated_quote_or_changed_number_cannot_reach_the_reviewer() -> None:
    model = ScriptedModel(
        [
            {
                "claims": [
                    {
                        "text": "Cedar achieved 99% accuracy.",
                        "quote": "Cedar achieved 99% accuracy.",
                        "source": 0,
                    },
                    {"text": "Cedar achieved 99% accuracy.", "quote": SOURCE, "source": 0},
                ]
            }
        ]
    )
    result = await compose_answer(
        context("reject"), "chosen", "What is this paper about?", [ROW], cast(LLM, model)
    )
    assert result.mode == "extractive_summary" and result.model_calls == 1
    assert result.rejected_claims == 2 and result.excerpts
    assert "99%" not in " ".join(item.text for item in result.excerpts)


@pytest.mark.parametrize("verdict", [{"supported": []}, {"supported": [True, "0", 9]}, {}])
@pytest.mark.asyncio
async def test_critic_rejection_or_malformed_verdict_falls_back_to_source(verdict: object) -> None:
    invented = "Cedar is designed to cure rare diseases."
    model = ScriptedModel(
        [
            {"claims": [{"text": invented, "quote": SOURCE.split(". ")[0] + ".", "source": 0}]},
            verdict,
        ]
    )
    result = await compose_answer(
        context("critic"), "chosen", "Summarize this paper", [ROW], cast(LLM, model)
    )
    assert result.mode == "extractive_summary"
    assert all(item.text != invented for item in result.excerpts)
    assert all(item.checks == ("tier_1",) for item in result.excerpts)


@pytest.mark.asyncio
async def test_no_evidence_abstains_without_a_model_call() -> None:
    model = ScriptedModel([])
    result = await compose_answer(
        context("empty"), "chosen", "What is the energy usage?", [], cast(LLM, model)
    )
    assert not result.excerpts and result.model_calls == 0


@pytest.mark.asyncio
async def test_instructions_in_source_cannot_become_fallback_claims() -> None:
    from arie_core.harness.evaluation import score_answer, source_truth_cases
    from arie_core.harness.reader import retrieve

    case = source_truth_cases()[-1]
    selected = retrieve("evaluation", case.question, case.rows)
    result = await compose_answer(context("injection"), "evaluation", case.question, selected)
    assert "100%" not in " ".join(item.text for item in result.excerpts)
    assert score_answer(case, result).grounding and score_answer(case, result).coverage


@pytest.mark.asyncio
async def test_provider_failure_is_reported_without_exposing_credentials() -> None:
    import httpx

    class UnavailableModel:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx, request
            response = httpx.Response(401, request=httpx.Request("POST", "https://provider.test"))
            response.raise_for_status()
            raise AssertionError("Unreachable")

    result = await compose_answer(
        context("authentication"),
        "chosen",
        "Summarize this paper",
        [ROW],
        cast(LLM, UnavailableModel()),
    )
    assert result.fallback_reason == "provider_http_401"
    assert result.mode == "extractive_summary" and result.output_tokens == 0


@pytest.mark.asyncio
async def test_bold_abstract_and_section_diversity_beat_intro_background() -> None:
    from arie_core.harness.reader import retrieve

    texts = [
        ("**Abstract**", 1),
        (
            "We evaluated Cedar, a retrieval model using a gated memory bank. "
            "We compared three memory settings on 120 English documents. "
            "The model improved accuracy from 62% to 68% on this benchmark.",
            1,
        ),
        ("## 1 Introduction", 1),
        ("Gains are driven by large architectures, curated datasets, and transfer learning.", 1),
        ("## 2 Results", 2),
        ("### 2.1 Findings", 2),
        ("The model improved accuracy from 62% to 68% on this benchmark.", 2),
        ("## 3 Methods", 3),
        ("The method uses a gated memory bank for retrieving evidence.", 3),
        ("## References", 4),
        ("We propose an unrelated method from another publication.", 4),
    ]
    rows = [
        EvidenceChunk("chosen", str(i), text, page, i * 1000, i * 1000 + len(text))
        for i, (text, page) in enumerate(texts)
    ]
    selected = retrieve("chosen", "What is this paper about?", rows)
    assert selected[0].section == "Abstract"
    assert any(row.section == "2 Results / 2.1 Findings" for row in selected)
    assert any(row.section == "3 Methods" for row in selected)
    assert all("References" not in row.section for row in selected)
    result = await compose_answer(
        context("abstract"), "chosen", "What is this paper about?", selected
    )
    assert "Cedar" in result.excerpts[0].text
    assert "68%" in " ".join(item.text for item in result.excerpts)
    assert "transfer learning" not in " ".join(item.text for item in result.excerpts)
    for item in result.excerpts:
        row = next(row for row in rows if row.chunk_id == item.chunk_id)
        assert (
            row.text[item.char_start - row.char_start : item.char_end - row.char_start]
            == item.quote
        )


@pytest.mark.parametrize("relevance", [False, None, "true", 1])
@pytest.mark.asyncio
async def test_reviewer_must_confirm_question_relevance(relevance: object) -> None:
    sentence = "Cedar uses a gated memory bank."
    model = ScriptedModel(
        [
            {"claims": [{"text": sentence, "quote": SOURCE.split(". ")[0] + ".", "source": 0}]},
            {"supported": [0], "answers_question": relevance},
        ]
    )
    result = await compose_answer(
        context("relevance"), "chosen", "What is this paper about?", [ROW], cast(LLM, model)
    )
    assert result.mode == "extractive_summary" and not result.answer_relevant
    assert result.fallback_reason == "question_not_answered"
    assert result.reviewed_claims == result.rejected_claims == 1


@pytest.mark.asyncio
async def test_generic_background_alone_does_not_become_a_paper_summary() -> None:
    row = EvidenceChunk(
        "chosen",
        "background",
        "Gains are driven by high-capacity architectures, curated datasets, and transfer learning.",
        1,
        0,
        90,
        "Introduction",
    )
    result = await compose_answer(
        context("background"), "chosen", "What is this paper about?", [row]
    )
    assert not result.excerpts


def test_evaluation_requires_relevant_opening_and_rejects_background_only_answers() -> None:
    from arie_core.harness.answer import AnswerResult
    from arie_core.harness.evaluation import score_answer, source_truth_cases
    from arie_core.harness.reader import Excerpt

    case = next(case for case in source_truth_cases() if case.name == "overview-with-background")
    excerpts = [
        Excerpt(row.text, row.text, row.page_start, 0, len(row.text), row.chunk_id)
        for row in case.rows[1:]
    ]
    answer = AnswerResult(
        excerpts,
        "grounded_summary",
        model_calls=2,
        reviewed_claims=len(excerpts),
        answer_relevant=True,
    )
    assert score_answer(case, answer).passed
    unreviewed = AnswerResult(excerpts, "grounded_summary")
    assert not score_answer(case, unreviewed).passed
    background = case.rows[0]
    irrelevant = AnswerResult(
        [
            Excerpt(
                background.text, background.text, 1, 0, len(background.text), background.chunk_id
            )
        ],
        "grounded_summary",
        model_calls=2,
        reviewed_claims=1,
        answer_relevant=True,
    )
    score = score_answer(case, irrelevant)
    assert score.grounding and not score.coverage and not score.passed
