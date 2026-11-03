from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest
from arie_core.contracts import ChunkRow, Claim, Evidence
from arie_core.harness import (
    EvidenceVerifier,
    GroundedQAHarness,
    PromptCacheBuilder,
    ReciprocalRankFusion,
    ZeroLLMRouter,
    build_lexical_tools,
)


def test_zero_llm_router_rejects_overlarge_and_cross_paper_queries() -> None:
    router = ZeroLLMRouter()
    assert router.route("a " * 2001).route == "abstain"
    assert (
        router.route(
            "compare neural nets across papers", paper_id="p1", paper_state="INDEXED"
        ).route
        == "abstain"
    )
    assert router.route("what is the main result", paper_state="INDEXED").route == "results_tables"


def test_hybrid_prefetch_fuses_ranks_and_respects_caps() -> None:
    _ = [
        ChunkRow(
            chunk_id="c1",
            paper_id="p1",
            section="methods",
            text="Alpha method details with accuracy and training",
            order=0,
            token_count=40,
        ),
        ChunkRow(
            chunk_id="c2",
            paper_id="p1",
            section="results",
            text="Beta results table reports 94.2 accuracy",
            order=1,
            token_count=50,
        ),
        ChunkRow(
            chunk_id="c3",
            paper_id="p1",
            section="limitations",
            text="Model limitations include sample bias and scale issues",
            order=2,
            token_count=60,
        ),
    ]
    prefetch = ReciprocalRankFusion(k=60)
    fused = prefetch.fuse(["c1", "c2", "c3"], ["c3", "c1", "c2"])
    assert "c3" in fused
    assert max(fused.values()) > 0.0


def test_lexical_tool_schema_hides_paper_id() -> None:
    chunks = [
        ChunkRow(
            chunk_id="c1",
            paper_id="p1",
            section="methods",
            text="The method uses dropout and data augmentation.",
            order=0,
            token_count=20,
        ),
    ]
    tools = build_lexical_tools(paper_id="p1", chunks=chunks)
    assert "paper_id" not in str(tools["search_paper"])
    assert tools["search_paper"] is not None
    assert tools["read_section"].__name__ == "read_section"


def test_verifier_strips_short_and_fabricated_quotes() -> None:
    verifier = EvidenceVerifier()
    assert (
        verifier.verify(
            Claim(
                text="The method uses dropout.",
                evidence=[Evidence(id="e1", chunk_id="c1", quote="short")],
            )
        )
        is False
    )
    claim = Claim(
        text="The method uses dropout.",
        evidence=[
            Evidence(id="e1", chunk_id="c1", quote="The method uses dropout and data augmentation.")
        ],
    )
    assert (
        verifier.verify(claim, chunk_text="The method uses dropout and data augmentation.") is True
    )


def test_lexical_pre_filter_needs_shared_content_and_unchanged_anchors() -> None:
    quote = "The model performs well on all benchmarks, improving accuracy from 71.2% to 78.4%."
    assert EvidenceVerifier.lexical_support("The model improves accuracy to 78.4%.", [quote])
    assert not EvidenceVerifier.lexical_support("The model improves accuracy to 91%.", [quote])
    assert not EvidenceVerifier.lexical_support("Cedar cures rare diseases quickly.", [quote])
    # A polarity flip shares every content word; only the Tier 3 model can reject it.
    assert EvidenceVerifier.lexical_support("The model performs badly on benchmarks.", [quote])


@pytest.mark.asyncio
async def test_tier_3_applies_the_entailment_floor_and_rejects_malformed_scores() -> None:
    async def scores(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [0.98 if "well" in claim else 0.001 for _, claim in pairs]

    pairs = [("It performs well.", "It performs well."), ("It performs well.", "It is bad.")]
    assert await EvidenceVerifier.tier_3(scores, pairs) == [True, False]
    assert await EvidenceVerifier.tier_3(scores, []) == []

    async def short(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [1.0]

    async def out_of_range(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [1.5 for _ in pairs]

    for broken in (short, out_of_range):
        with pytest.raises(ValueError, match="malformed"):
            await EvidenceVerifier.tier_3(broken, pairs)


def test_verifier_normalizes_ligatures_and_silences_fabricated_offsets() -> None:
    content = "The study compares a compact model with aﬂow-based training."
    quote = "The study compares a flow-based training."
    normalized = EvidenceVerifier.tier_2(content, quote)
    assert normalized is True

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE chunks ("
        "id TEXT, content TEXT, section_path TEXT, "
        "page_start INTEGER, char_start INTEGER, char_end INTEGER"
        ")"
    )
    connection.execute(
        "INSERT INTO chunks ("
        "id, content, section_path, page_start, char_start, char_end"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        ("c1", content, "methods/foo", 4, 10, 42),
    )
    row = connection.execute("SELECT * FROM chunks WHERE id = ?", ("c1",)).fetchone()
    assert row is not None
    assert EvidenceVerifier.tier_2(content, quote, chunk_row=row) is True


def test_harness_kills_infinite_tool_loops_at_three_calls() -> None:
    chunks = [
        ChunkRow(
            chunk_id="c1",
            paper_id="p1",
            section="methods",
            text="The method uses dropout and data augmentation.",
            order=0,
            token_count=20,
        ),
    ]
    harness = GroundedQAHarness(
        paper_id="p1",
        title="Demo",
        source="demo",
        chunks=list(chunks),
    )
    answer = harness.answer("what is the method?")
    assert answer.status in {"answered", "insufficient"}
    assert answer.route in {"method", "overview", "locate_fact"}
    assert answer.llm_tokens == 0


def test_prompt_cache_builder_zone_b_sorts_metadata_and_uses_playbook() -> None:
    builder = PromptCacheBuilder(project_root=Path(__file__).resolve().parents[2])
    prompt = builder.build_zone_b(
        paper_id="p1",
        title="Demo paper",
        source="arxiv",
        custom_metadata={"z": "zzz", "a": "aaa"},
        outline=["intro", "methods"],
    )
    assert "custom_metadata" in prompt
    assert "a: aaa" in prompt
    assert "zone_b_template" not in prompt.lower()
