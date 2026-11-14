from __future__ import annotations

import asyncio
import math
import os
import re
import time
from pathlib import Path
from typing import cast

import pytest
from arie_api.runtime import Document, ResearchRuntime, context
from arie_core.harness.orchestrator import answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.plugin_api import Embedder
from arie_core.registry.manager import ArieConfig, RegistryManager

ROOT = Path(__file__).resolve().parents[2]
# Word groups stand in for meaning: "long inputs" and "context length" share a concept.
CONCEPTS = (
    {"long", "inputs", "context", "length", "tokens"},
    {"translation", "bleu", "french"},
    {"limitations", "scanned", "unsupported"},
)
PASSAGES = (
    "We extend the context length to 128K tokens with position interpolation.",
    "Our model attains 41.8 BLEU on English-to-French translation.",
    "The method requires readable text; scanned images are unsupported.",
)


class ConceptEmbedding:
    """Deterministic stand-in for the worker pool: one dimension per concept."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    async def embed(self, texts: list[str], kill_after: float = 60.0) -> list[list[float]]:
        del kill_after
        if self.fail:
            raise RuntimeError("model unavailable")
        self.calls.append(texts)
        vectors: list[list[float]] = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            vector = [float(len(words & concept)) for concept in CONCEPTS] + [0.3]
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors

    def close(self) -> None:
        pass


def runtime_at(tmp_path: Path) -> tuple[ResearchRuntime, ConceptEmbedding]:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "concept", "model": "v1", "min_score": 0.6},
        vector_store={"use": "exact", "path": str(tmp_path / "vectors.db")},
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))
    embedding = ConceptEmbedding()
    runtime.embedding = embedding  # pyright: ignore[reportAttributeAccessIssue]
    return runtime, embedding


def paper(paper_id: str, texts: tuple[str, ...]) -> Document:
    return dict(
        id=paper_id,
        title=f"Paper {paper_id}",
        source="arxiv",
        abstract="",
        status="INDEXED",
        published_at="2026-10-01T00:00:00+00:00",
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        dedup_hash=f"hash-{paper_id}",
        metadata={"primary_category": "cs.CL"},
        pages=[{"number": 1, "text": " ".join(texts)}],
        chunks=[
            dict(
                paper_id=paper_id,
                chunk_id=f"{paper_id}:{i}",
                page_start=1,
                char_start=0,
                char_end=len(text),
                text=text,
            )
            for i, text in enumerate(texts)
        ],
    )


async def settle(runtime: ResearchRuntime, status: str) -> Document:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        report = await runtime.vector_status()
        if report["status"] == status:
            return report
        await asyncio.sleep(0.02)
    raise AssertionError(await runtime.vector_status())


@pytest.mark.asyncio
async def test_backfill_search_reembed_and_delete(tmp_path: Path) -> None:
    runtime, embedding = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper("a", PASSAGES))
        await runtime.save_paper(paper("b", ("Long context tokens appear in another paper.",)))
        assert (await runtime.vector_status())["status"] == "indexing"
        indexer = asyncio.create_task(runtime.run_vector_index())
        try:
            report = await settle(runtime, "healthy")
            assert report["vectors"] == 4 and report["papers_indexed"] == 2
            assert report["embedder"] == "concept:v1" and report["size_bytes"] > 0
            assert embedding.calls[0] == ["warm-up"]

            ranked = await runtime.semantic_search("a", ["how does it handle long inputs?"])
            assert ranked == ["a:0"]  # other paper's match excluded; weak matches dropped
            assert await runtime.semantic_search("a", ["quantum bananas?"]) == []

            # A new embedder model makes every paper stale until it is re-embedded.
            runtime.embedder_id = "concept:v2"
            assert await runtime.semantic_search("a", ["long inputs"]) == []
            runtime.vector_wakeup.set()
            assert (await settle(runtime, "healthy"))["vectors"] == 4

            # A re-synced paper with new chunks is re-embedded after sync signals the indexer.
            await runtime.save_paper(paper("b", ("Translation to French.", "Scanned pages.")))
            runtime.vector_wakeup.set()
            assert (await settle(runtime, "healthy"))["vectors"] == 4 - 1 + 2

            await runtime.delete_paper("b")
            report = await runtime.vector_status()
            assert report["vectors"] == 3 and report["papers_total"] == 1
        finally:
            indexer.cancel()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_one_paper_is_embedded_only_when_its_vectors_are_not_current(
    tmp_path: Path,
) -> None:
    runtime, embedding = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper("a", PASSAGES))
        await runtime.ensure_paper_vectors("a")
        assert len(embedding.calls) == 1
        await runtime.ensure_paper_vectors("a")
        assert len(embedding.calls) == 1
        runtime.embedder_id = "concept:v2"
        await runtime.ensure_paper_vectors("a")
        assert len(embedding.calls) == 2
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_long_table_row_groups_map_back_to_one_chunk(tmp_path: Path) -> None:
    runtime, embedding = runtime_at(tmp_path)
    rows = [f"| run-{i:03d} | translation | {i}.5 |" for i in range(200)]
    rows[150] = "| run-150 | translation bleu french | 41.8 |"
    table = "Table 2: Runs.\n\n| Run | Task | Score |\n| --- | --- | --- |\n" + "\n".join(rows)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper("a", ("Scanned pages are unsupported.", table)))
        await runtime.ensure_paper_vectors("a")
        embedded = embedding.calls[-1]
        groups = [text for text in embedded if text.startswith("Table 2: Runs.")]
        assert len(groups) > 3
        assert all("| Run | Task | Score |" in text for text in groups)
        assert runtime.vector_store is not None
        query = (await embedding.embed(["french"]))[0]
        stored = await runtime.vector_store.search(
            context("t", 5.0), query, 50, filters={"paper_id": "a", "model": "concept:v1"}
        )
        assert {hit.payload["chunk_id"] for hit in stored} == {"a:0", "a:1"}
        assert {hit.payload["kind"] for hit in stored} == {"prose", "table"}
        assert "a:1#t3" in {hit.id for hit in stored}
        # Every group matches "translation"; the chunk comes back once.
        assert await runtime.semantic_search("a", ["french translation bleu"]) == ["a:1"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_indexing_waits_for_a_question_but_the_questions_own_search_does_not(
    tmp_path: Path,
) -> None:
    runtime, embedding = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        await runtime.save_paper(paper("a", PASSAGES))
        await runtime.ensure_paper_vectors("a")
        indexed = len(embedding.calls)
        with runtime.foreground.hold():
            reindex = asyncio.create_task(runtime.index_paper_vectors("a"))
            # The question embeds its own query while the background run is held back.
            ranked = await asyncio.wait_for(
                runtime.semantic_search("a", ["how does it handle long inputs?"]), 5.0
            )
            assert ranked == ["a:0"]
            await asyncio.sleep(0.05)
            assert not reindex.done() and len(embedding.calls) == indexed + 1
        await asyncio.wait_for(reindex, 5.0)
        assert len(embedding.calls) == indexed + 2  # the query, then the one-batch re-embed
        assert await runtime.semantic_search("a", ["long inputs"]) == ["a:0"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_embedder_failure_is_reported_and_keyword_search_remains(tmp_path: Path) -> None:
    runtime, embedding = runtime_at(tmp_path)
    await runtime.initialize()
    embedding.fail = True
    try:
        await runtime.save_paper(paper("a", PASSAGES))
        indexer = asyncio.create_task(runtime.run_vector_index())
        try:
            report = await settle(runtime, "degraded")
            assert "model unavailable" in report["detail"] and report["papers_indexed"] == 0
            assert await runtime.semantic_search("a", ["long inputs"]) == []
        finally:
            indexer.cancel()
    finally:
        runtime.close()


ROWS = [
    EvidenceChunk("a", f"a:{i}", text, 1, 0, len(text), "Method") for i, text in enumerate(PASSAGES)
]


@pytest.mark.asyncio
async def test_vector_hits_answer_paraphrased_questions_within_the_bound_paper() -> None:
    question = "How big can a prompt get?"
    keyword_only = await answer_question(context("keyword"), "a", question, ROWS)
    assert not keyword_only.answer.excerpts

    async def semantic(queries: object) -> list[str]:
        del queries
        return ["foreign:0", "a:0"]  # IDs outside the bound paper are ignored

    run = await answer_question(context("hybrid"), "a", question, ROWS, semantic=semantic)
    assert [excerpt.chunk_id for excerpt in run.answer.excerpts] == ["a:0"]
    assert run.answer.excerpts[0].quote in PASSAGES[0]

    async def broken(queries: object) -> list[str]:
        raise RuntimeError("store offline")

    fallback = await answer_question(context("broken"), "a", question, ROWS, semantic=broken)
    assert not fallback.answer.excerpts and fallback.guardrail.allowed


@pytest.mark.asyncio
async def test_fastembed_refuses_to_load_in_the_api_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARIE_EMBED_WORKER", raising=False)
    assert "ARIE_EMBED_WORKER" not in os.environ
    with pytest.raises(RuntimeError, match="arie-worker"):
        embedder = RegistryManager(project_root=ROOT).build_plugin("embedder", "fastembed")
        await cast(Embedder, embedder).embed(context("api"), ["text"])


def test_review_evidence_fits_budget_with_verbatim_sentences() -> None:
    from arie_core.harness.answer import fit_evidence
    from arie_core.harness.reader import Excerpt

    passage = (
        "The method weights each sample by mutual information. "
        "Training uses eight GPUs for two days. "
        "Samples with unstable predictions receive lower weight."
    )
    claim = Excerpt("Unstable predictions get lower weight.", passage, 1, 0, len(passage), "a:0")
    assert fit_evidence([claim], 10_000) == [passage]
    fitted = fit_evidence([claim], 60)[0]
    assert fitted == "Samples with unstable predictions receive lower weight."
    assert fitted in passage and fit_evidence([claim], 5) == [""]
