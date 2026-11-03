from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import pytest
from arie_core.contracts import Claim, Evidence
from arie_core.harness import EvidenceVerifier, reciprocal_rank_fusion


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


def test_reciprocal_rank_fusion_rewards_items_ranked_high_in_every_list() -> None:
    fused = reciprocal_rank_fusion(["c1", "c2", "c3"], ["c3", "c1", "c2"])
    assert max(fused, key=fused.__getitem__) == "c1"
    assert fused["c1"] == 1 / 61 + 1 / 62
