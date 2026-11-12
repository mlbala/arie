"""The Qasper command: cases are saved per paper and imports are waited for, without network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from arie_api.eval_qasper import import_paper, load_dev, parse_args
from arie_api.runtime import ResearchRuntime
from arie_core.harness.paper_eval import PaperEvalCase
from arie_core.registry.manager import ArieConfig

ROOT = Path(__file__).resolve().parents[2]


def _chunk(paper_id: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "chunk_id": f"{paper_id}:0",
        "text": "x",
        "page_start": 1,
        "char_start": 0,
        "char_end": 1,
    }


def _case(case_id: str, origin: str = "qasper") -> PaperEvalCase:
    return PaperEvalCase(case_id=case_id, question="Why?", kind="fact", origin=origin)


def test_saving_cases_replaces_the_papers_cases_and_keeps_other_papers(tmp_path: Path) -> None:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
        reranker={"use": "none"},
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))

    async def scenario() -> None:
        await runtime.initialize()
        evaluations = runtime.evaluations
        await evaluations.save_cases("p1", [_case("a"), _case("b")])
        await evaluations.save_cases("p2", [_case("c", "generated")])
        await evaluations.save_cases("p1", [_case("d")])
        assert [case.case_id for case in await evaluations.cases("p1")] == ["d"]
        assert [case.origin for case in await evaluations.cases("p2")] == ["generated"]
        paper: dict[str, Any] = {"id": "p1", "chunks": [_chunk("p1"), _chunk("p2")]}
        assert [row.chunk_id for row in evaluations.chunks(paper)] == ["p1:0"]

    try:
        asyncio.run(scenario())
    finally:
        runtime.close()


class _FakeRuntime:
    """Just enough of ``ResearchRuntime`` for ``import_paper``: papers, one job, a job table."""

    def __init__(self, papers: dict[str, dict[str, Any]], job: dict[str, Any] | None) -> None:
        self.papers = papers
        self.job = job
        self.jobs: dict[str, dict[str, Any]] = {}
        self.imports: list[str] = []

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self.papers.get(paper_id)

    async def start_import(self, url: str) -> dict[str, Any]:
        assert self.job is not None
        self.imports.append(url)
        self.jobs[self.job["job_id"]] = self.job
        return self.job


def _import(runtime: _FakeRuntime, paper_id: str) -> tuple[str, str]:
    return asyncio.run(import_paper(cast(ResearchRuntime, runtime), paper_id))


def test_an_indexed_paper_is_not_imported_again() -> None:
    runtime = _FakeRuntime({"1503.00841": {"status": "INDEXED"}}, None)
    assert _import(runtime, "1503.00841") == ("1503.00841", "")
    assert runtime.imports == []


def test_a_completed_import_returns_the_library_id_and_a_failed_one_its_reason() -> None:
    done = {"job_id": "import-1", "status": "completed", "paper_id": "1503.00841"}
    runtime = _FakeRuntime({}, done)
    assert _import(runtime, "1503.00841") == ("1503.00841", "")
    assert runtime.imports == ["https://arxiv.org/pdf/1503.00841"]

    failed = {"job_id": "import-2", "status": "failed", "paper_id": None, "error": "No PDF"}
    assert _import(_FakeRuntime({}, failed), "1601.02403") == ("1601.02403", "No PDF")


def test_the_dev_split_is_read_from_the_cache_without_downloading(tmp_path: Path) -> None:
    dev: dict[str, dict[str, object]] = {"1503.00841": {"title": "T", "qas": []}}
    (tmp_path / "qasper-dev-v0.3.json").write_text(json.dumps(dev), encoding="utf-8")
    papers = asyncio.run(load_dev(tmp_path))
    assert list(papers) == ["1503.00841"] and papers["1503.00841"].title == "T"


def test_the_command_scores_strict_by_default_and_explain_on_request() -> None:
    assert parse_args(["30"]).mode == "strict"
    assert parse_args(["5", "--mode", "explain"]).mode == "explain"
    assert parse_args(["30"]).kind == "any"
    assert parse_args(["20", "--kind", "figure"]).kind == "figure"
    with pytest.raises(SystemExit):
        parse_args(["5", "--kind", "chart"])
    with pytest.raises(SystemExit):
        parse_args(["5", "--mode", "loud"])
