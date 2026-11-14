"""Library retention: keep the newest ``sync.keep_papers`` papers, never ones with eval cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from arie_api.runtime import Document, ResearchRuntime, context
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import SyncConfig, papers_to_prune

ROOT = Path(__file__).resolve().parents[2]


def test_papers_beyond_the_newest_n_are_listed_and_protected_ones_never() -> None:
    papers = [
        ("a", "2026-10-01"),
        ("b", "2026-10-30"),
        ("c", "2026-10-30"),
        ("d", "2026-09-01"),
        ("q", "2018-01-01"),
    ]
    assert papers_to_prune(papers, {"q"}, 2) == ["a", "d"]  # ties keep the higher id: c, b
    assert papers_to_prune(papers, set(), 10) == []
    assert papers_to_prune(papers, {"q"}, 1) == ["b", "a", "d"]


def test_keep_papers_is_off_by_default_and_validated() -> None:
    assert SyncConfig().keep_papers is None
    with pytest.raises(ValueError):
        SyncConfig.model_validate({"keep_papers": 0})


def paper(paper_id: str, published: str) -> Document:
    return dict(
        id=paper_id,
        title=f"Paper {paper_id}",
        source="arxiv",
        abstract="",
        status="INDEXED",
        published_at=published,
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        dedup_hash=f"hash-{paper_id}",
        metadata={},
        pages=[{"number": 1, "text": "Text."}],
        chunks=[],
    )


@pytest.mark.asyncio
async def test_prune_library_deletes_the_oldest_and_keeps_papers_with_eval_cases(
    tmp_path: Path,
) -> None:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        vector_store={"use": "none"},
    )
    sync = config.sync.model_copy(update={"keep_papers": 2})
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins, "sync": sync}))
    await runtime.initialize()
    try:
        for paper_id, published in (
            ("new", "2026-10-30T00:00:00+00:00"),
            ("mid", "2026-10-01T00:00:00+00:00"),
            ("old", "2026-08-20T00:00:00+00:00"),
            ("qasper", "2018-05-01T00:00:00+00:00"),
        ):
            await runtime.save_paper(paper(paper_id, published))
        await runtime.db.execute(
            context("case"),
            "INSERT INTO arie_eval_cases (case_id, paper_id, case_json, created_at) "
            "VALUES ('c1', 'qasper', '{}', '2026-11-13')",
        )
        assert await runtime.prune_library() == ["old"]
        assert await runtime.get_paper("old") is None
        for kept in ("new", "mid", "qasper"):
            assert await runtime.get_paper(kept) is not None
        assert await runtime.prune_library() == []
    finally:
        runtime.close()
