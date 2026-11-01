import hashlib

import pytest
from arie_core.contracts import (
    Evidence,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from arie_core.plugin_api import ScrapedPaperInput, normalize_scraper_output
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError


@pytest.mark.parametrize(
    "paper_id",
    ["../payload", "..\\evil", "folder\\file", "a/../b"],
)
def test_scraped_paper_input_rejects_path_traversal(paper_id: str) -> None:
    with pytest.raises(ValueError):
        ScrapedPaperInput(
            paper_id=paper_id,
            title="Example",
            source="arxiv",
            pdf_url="https://example.com/paper.pdf",
        )


def test_scraped_paper_input_allows_arxiv_like_ids_and_hashes_safely() -> None:
    item = ScrapedPaperInput(
        paper_id="cs/0501001",
        title="  Safe Paper  ",
        source="arxiv",
        pdf_url=" https://example.com/paper.pdf ",
    )
    assert item.paper_id == "cs/0501001"
    assert item.storage_hash == hashlib.sha256(b"cs/0501001").hexdigest()[:16]
    assert item.dedup_hash == hashlib.sha256(b"safe paper||https://example.com/paper.pdf").hexdigest()


def test_scraped_paper_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ScrapedPaperInput.model_validate(
            {
                "paper_id": "cs/0501001",
                "title": "Example",
                "source": "arxiv",
                "pdf_url": "https://example.com/paper.pdf",
                "extra_field": "bad",
            }
        )


def test_normalize_scraper_output_accepts_dicts_and_models() -> None:
    raw = [
        {
            "paper_id": "cs/0501001",
            "title": "  Example Title  ",
            "source": "arxiv",
            "pdf_url": " https://example.com/example.pdf ",
            "authors": ["Ada Lovelace"],
        },
        ScrapedPaperInput(
            paper_id="cs/0501002",
            title="Another",
            source="arxiv",
            pdf_url="https://example.com/another.pdf",
        ),
    ]

    normalized = normalize_scraper_output(raw, "arxiv")
    assert len(normalized) == 2
    assert all(isinstance(item, ScrapedPaperInput) for item in normalized)
    assert normalized[0].title == "Example Title"
    assert normalized[0].pdf_url == "https://example.com/example.pdf"
    assert normalized[1].paper_id == "cs/0501002"


@given(st.text())
def test_hypothesis_rejects_path_traversal_only_when_unsafe(value: str) -> None:
    if ".." not in value and "\\" not in value:
        item = ScrapedPaperInput(
            paper_id=value,
            title="Example",
            source="arxiv",
            pdf_url="https://example.com/paper.pdf",
        )
        assert item.paper_id == value.strip()
    else:
        with pytest.raises(ValueError):
            ScrapedPaperInput(
                paper_id=value,
                title="Example",
                source="arxiv",
                pdf_url="https://example.com/paper.pdf",
            )


def test_contract_models_build_and_round_trip() -> None:
    Evidence(id="e1", chunk_id="c1", quote="The sky is blue.")
    tool = ToolCall(name="lookup", args={"query": "blue"}, call_id="t1")
    response = LLMResponse(
        text="A short answer",
        usage=TokenUsage(
            prompt=3,
            completion=2,
            cache_hit=1,
            cache_miss=0,
        ),
        tool_calls=[tool],
        finish=True,
    )
    delta = LLMDelta(text="A", finish=False)
    request = LLMRequest(model="gpt-4o-mini", prompt="Hello")

    assert response.tool_calls[0].name == "lookup"
    assert delta.text == "A"
    assert request.prompt == "Hello"
    assert response.usage.prompt == 3
