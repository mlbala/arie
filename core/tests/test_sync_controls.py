from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from arie_api.monitor import StorageMonitor
from arie_api.routes import create_app
from arie_api.runtime import ResearchRuntime
from arie_core.plugin_api import Ctx, ScrapedPaperInput, SourcePage
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import SyncConfig
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("settings", "start", "end"),
    [
        ({"mode": "year", "year": 2025}, "2025-01-01", "2025-12-31"),
        ({"mode": "month", "year": 2024, "month": 2}, "2024-02-01", "2024-02-29"),
        ({"mode": "since", "start_month": "2026-01"}, "2026-01-01", "2026-11-02"),
        (
            {"mode": "range", "start_month": "2025-01", "end_month": "2025-06"},
            "2025-01-01",
            "2025-06-30",
        ),
        (
            {"mode": "range", "start_month": "2026-01", "end_month": "2026-12"},
            "2026-01-01",
            "2026-11-02",
        ),
    ],
)
def test_publication_windows_include_last_day(
    settings: dict[str, object],
    start: str,
    end: str,
) -> None:
    since, until = SyncConfig.model_validate(settings).window(NOW)
    assert since.isoformat().startswith(start)
    assert until.isoformat().startswith(end)
    assert until <= NOW and since.tzinfo == UTC
    if end != "2026-11-02":
        assert (until.hour, until.minute, until.second, until.microsecond) == (23, 59, 59, 999999)


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "month", "year": 2025},
        {"mode": "year", "year": 2025, "month": 1},
        {"mode": "range", "start_month": "2025-06", "end_month": "2025-01"},
        {"mode": "since", "start_month": "2025-13"},
        {"max_papers": 0},
        {"batch_size": 101},
        {"year": 2025},
        {"mode": "since", "start_month": "0000-01"},
    ],
)
def test_rejects_ambiguous_or_unbounded_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SyncConfig.model_validate(settings)


def test_future_only_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="future"):
        SyncConfig(mode="year", year=2027).window(NOW)


def runtime_at(tmp_path: Path) -> ResearchRuntime:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        sources=["arxiv"],
    )
    return ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))


@pytest.mark.asyncio
async def test_limit_resume_keeps_window_page_offset_and_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    seen: list[str] = []
    bounds: list[tuple[str, str]] = []

    class Source:
        allowed_hosts = ("arxiv.org",)

        async def fetch_page(
            self,
            ctx: Ctx,
            since: str,
            until: str,
            cursor: str | None,
            limit: int,
        ) -> SourcePage:
            del ctx, cursor
            assert limit == 3
            bounds.append((since, until))
            return SourcePage(
                [
                    ScrapedPaperInput(
                        paper_id=f"paper-{i}",
                        title="AI agents",
                        source="arxiv",
                        pdf_url=f"https://arxiv.org/pdf/{i}",
                        published_at="2025-06-01T00:00:00Z",
                    )
                    for i in range(3)
                ],
                None,
            )

    async def ingest(job: dict[str, Any], paper: ScrapedPaperInput, hosts: tuple[str, ...]) -> None:
        del hosts
        seen.append(paper.paper_id)
        job["processed"] += 1
        job["indexed"] += 1

    def source_instance(*args: object, **kwargs: object) -> Source:
        return Source()

    monkeypatch.setattr(runtime.manager, "instantiate_plugin", source_instance)
    monkeypatch.setattr(runtime, "_supervised_paper", ingest)
    settings = SyncConfig(mode="year", year=2025, batch_size=3, max_papers=1)
    try:
        job = await runtime.start_sync(settings=settings)
        await runtime._run_job(job)  # pyright: ignore[reportPrivateUsage]
        assert job["status"] == "interrupted" and job["page_offset"] == 1
        runtime.config.sync = SyncConfig(batch_size=1, max_papers=10)
        resumed = await runtime.start_sync(resume_job_id=job["job_id"])
        await runtime._run_job(resumed)  # pyright: ignore[reportPrivateUsage]
        assert resumed["page_offset"] == 2
        resumed = await runtime.start_sync(resume_job_id=job["job_id"])
        await runtime._run_job(resumed)  # pyright: ignore[reportPrivateUsage]
        assert resumed["status"] == "completed"
        assert seen == ["paper-0", "paper-1", "paper-2"]
        assert resumed["discovered"] == resumed["processed"] == 3
        assert len(set(bounds)) == 1
    finally:
        runtime.close()


def paper(identifier: str, title: str, published: str) -> dict[str, Any]:
    return dict(
        id=identifier,
        title=title,
        abstract="Research with reproducible evidence.",
        source="arxiv",
        status="INDEXED",
        published_at=published,
        relevance_gate=True,
        overridden_by_user=0,
        status_reason="Indexed",
        metadata={"primary_category": "cs.AI", "version_id": identifier},
        pages=[{"number": 1, "text": "Large language models use agent tools."}],
        chunks=[{"paper_id": identifier, "text": "evidence"}],
    )


@pytest.mark.asyncio
async def test_library_filters_topics_deletion_and_storage(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    original = paper("first", "LLM agents", "2025-06-30T23:59:00+00:00")
    pdf = tmp_path / "papers" / "original.pdf"
    pdf.write_bytes(b"%PDF- test file")
    original["file_path"] = str(pdf)
    try:
        await runtime.save_paper(original)
        await runtime.save_paper(paper("second", "Vision", "2025-07-01T00:00:00+00:00"))
        result = await runtime.list_papers(month="2025-06", domain="agents", sort="oldest")
        assert [item["id"] for item in result["items"]] == ["first"]
        tags = result["items"][0]["tags"]
        assert {tag["tag"] for tag in tags} >= {"ai", "agents", "llm"}
        assert next(tag for tag in tags if tag["tag"] == "agents")["quote"] == "agents"
        assert not (await runtime.list_papers(search="100%"))["items"]
        monitor = StorageMonitor(runtime)
        snapshot = await monitor.snapshot()
        assert snapshot["database"]["status"] == "healthy"
        assert snapshot["file_storage"]["files"] == 1
        assert snapshot["file_storage"]["size_bytes"] == len(b"%PDF- test file")
        assert snapshot["vector_store"]["status"] == "disabled"
        assert snapshot["vector_store"]["size_bytes"] is None
        await runtime.delete_paper("first", pdf_only=True)
        assert not pdf.exists()
        retained = await runtime.get_paper("first")
        assert retained and retained["pages"] and retained["pdf_deleted"]
        await runtime.delete_paper("first")
        assert await runtime.get_paper("first") is None
        assert [row["id"] for row in (await runtime.list_papers())["items"]] == ["second"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_delete_restores_pdf_on_database_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    document = paper("safe", "AI", "2025-01-01T00:00:00+00:00")
    pdf = tmp_path / "papers" / "safe.pdf"
    pdf.write_bytes(b"%PDF- keep me")
    document["file_path"] = str(pdf)
    await runtime.save_paper(document)

    async def fail(*args: object, **kwargs: object) -> int:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(runtime.db, "execute", fail)
    try:
        with pytest.raises(RuntimeError, match="database unavailable"):
            await runtime.delete_paper("safe")
        assert pdf.read_bytes() == b"%PDF- keep me"
        assert await runtime.get_paper("safe") is not None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_settings_save_preserves_other_yaml_and_import_route_exists(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    path = tmp_path / "arie.yaml"
    path.write_text(
        "# Keep this comment\nprofile: hybrid\nsync:\n  initial_months: 6\n"
        "# Keep plugin guidance\nplugins: {}\nfeatures: []\n"
    )
    runtime.root = tmp_path
    settings = SyncConfig(mode="range", start_month="2025-01", end_month="2025-06", max_papers=10)
    try:
        await runtime.save_sync_settings(settings)
        assert ArieConfig.from_yaml(path).sync == settings
        assert (
            "# Keep this comment" in path.read_text()
            and "# Keep plugin guidance" in path.read_text()
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/imports", json={"url": "https://arxiv.org/pdf/1706.03762"}
            )
            assert response.status_code == 202, response.text
            assert (await client.get(f"/v1/imports/{response.json()['job_id']}")).status_code == 200
            assert (await client.get("/v1/papers?month=2025-13")).status_code == 422
            assert (await client.get("/v1/papers?domain=invalid")).status_code == 400
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_library_2026_windows_include_date_only_boundaries(tmp_path: Path) -> None:
    runtime = runtime_at(tmp_path)
    await runtime.initialize()
    try:
        for key, date in (
            ("old", "2025-12-31T23:59:59+00:00"),
            ("jan-first", "2026-01-01"),
            ("jan-last", "2026-01-31T23:59:59Z"),
            ("feb-first", "2026-02-01T00:00:00+00:00"),
            ("recent", datetime.now(UTC).isoformat()),
        ):
            await runtime.save_paper(paper(key, "AI research", date))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test"
        ) as client:

            async def ids(query: str) -> set[str]:
                response = await client.get(f"/v1/papers?{query}")
                assert response.status_code == 200, response.text
                return {row["id"] for row in response.json()["items"]}

            assert await ids("month=2026-01") == {"jan-first", "jan-last"}
            assert await ids("year=2026") == {"jan-first", "jan-last", "feb-first", "recent"}
            assert await ids("start_month=2026-01&end_month=2026-01") == {
                "jan-first",
                "jan-last",
            }
            assert "recent" in await ids("months=6")
            assert "old" not in await ids("months=6")
            assert "old" not in await ids("start_month=2026-02")
            assert await ids("month=9998-12") == set()
            for query in (
                "year=2026&month=2026-01",
                "end_month=2026-01",
                "start_month=2026-02&end_month=2026-01",
            ):
                assert (await client.get(f"/v1/papers?{query}")).status_code == 400
            assert (await client.get("/v1/papers?months=0")).status_code == 422
    finally:
        runtime.close()
