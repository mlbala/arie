from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
import pytest
from arie_api.export_openapi import export_openapi
from arie_api.mcp_server import create_scope_handle, verify_scope_handle
from arie_api.routes import create_app
from arie_api.runtime import ResearchRuntime
from arie_core.registry.manager import ArieConfig
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    "We propose Cedar, a model for grounded verification. "
    "The method verifies source quotes before producing an answer."
)


class _APIClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...
    def post(self, url: str, *, json: object) -> httpx.Response: ...


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_APIClient]:
    """Exercise real local API data without credentials, network calls or background sync."""
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
        sources=["arxiv"],
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))
    initialize = runtime.initialize
    instance = runtime.instance

    async def seed() -> None:
        await initialize()
        for identifier, status in (("paper-1", "INDEXED"), ("paper-2", "SKIPPED")):
            await runtime.save_paper(
                dict(
                    id=identifier,
                    title="Cedar research",
                    source="arxiv",
                    status=status,
                    published_at="2026-01-01",
                    abstract=SOURCE,
                    relevance_gate=status == "INDEXED",
                    overridden_by_user=0,
                    status_reason=status,
                    pages=[{"text": SOURCE}],
                    chunks=[
                        dict(
                            paper_id=identifier,
                            chunk_id=identifier + ":0",
                            text=SOURCE,
                            page_start=1,
                            char_start=0,
                            char_end=len(SOURCE),
                            section="Abstract",
                        )
                    ],
                    size_limit_mb=25,
                    allowed_hosts=["export.arxiv.org", "arxiv.org"],
                )
            )

    async def idle() -> None:
        await asyncio.Event().wait()

    async def finite_events() -> AsyncIterator[str]:
        yield 'event: update\ndata: {"job_id":"fixture"}\n\n'

    def without_model(slot: str, name: str | None = None) -> object:
        return object() if slot == "llm" else instance(slot, name)

    monkeypatch.setattr(runtime, "initialize", seed)
    monkeypatch.setattr(runtime, "instance", without_model)
    monkeypatch.setattr(runtime, "run", idle)
    monkeypatch.setattr(runtime, "run_imports", idle)
    monkeypatch.setattr(runtime, "events", finite_events)
    with TestClient(create_app(runtime)) as connected:
        yield cast(_APIClient, connected)


def test_papers_feed_and_detail(client: _APIClient) -> None:
    feed = client.get("/v1/papers")
    assert feed.status_code == 200
    payload = feed.json()
    assert "items" in payload
    assert payload["items"]
    first = payload["items"][0]
    assert first["id"]

    detail = client.get(f"/v1/papers/{first['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] in {"INDEXED", "SKIPPED"}
    assert body["page_count"] == 1
    assert body["id"] == first["id"]
    assert "file_path" not in body


def test_question_stream_emits_verified_claims_only(client: _APIClient) -> None:
    response = client.post(
        "/v1/papers/paper-1/questions",
        json={"question": "What is the main contribution?"},
    )
    assert response.status_code == 200
    text = response.text
    assert "event: progress" in text
    assert "event: claim" in text
    claims = [
        json.loads(block.split("data: ")[1])
        for block in text.split("\n\n")
        if block.startswith("event: claim\n")
    ]
    assert claims and all(item["verified"] is True for item in claims)
    for item in claims:
        claim = item["claim"]
        citation = claim["citation"]
        assert SOURCE[citation["char_start"] : citation["char_end"]] == claim["quote"]
        assert "tier_1" in claim["tier_checks"]
    assert "event: done" in text


def test_override_preserves_protected_fields(client: _APIClient) -> None:
    before = client.get("/v1/papers/paper-2").json()
    excessive = client.post(
        "/v1/papers/paper-2/override",
        json={"size_limit_mb": 42, "relevance_override": True},
    )
    assert excessive.status_code == 422
    blocked = client.post(
        "/v1/papers/paper-2/override",
        json={"size_limit_mb": 25, "relevance_override": True},
    )
    assert blocked.status_code == 409
    after = client.get("/v1/papers/paper-2").json()
    assert after == before
    assert after["size_limit_mb"] == 25
    assert after["allowed_hosts"] == ["export.arxiv.org", "arxiv.org"]


def test_sync_events_plugins_and_health(client: _APIClient) -> None:
    sync = client.post("/v1/sync", json={"source": "arxiv"})
    assert sync.status_code == 202
    assert sync.json()["status"] == "queued"
    assert sync.json()["single_flight"] is True

    events = client.get("/v1/events")
    assert events.status_code == 200
    assert "event: update" in events.text

    plugins = client.get("/v1/plugins")
    assert plugins.status_code == 200
    plugin_slots = [entry["slot"] for entry in plugins.json()["plugins"]]
    assert "source" in plugin_slots

    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True


def test_mcp_scope_handle_isolated_and_ordered(client: _APIClient) -> None:
    handle = create_scope_handle("paper-1")
    assert verify_scope_handle(handle) == "paper-1"

    listed = client.post(
        "/mcp/p/" + handle,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert listed.status_code == 200
    tools = listed.json()["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == sorted(names)
    assert all("paper_id" not in str(tool["inputSchema"]) for tool in tools)

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_paper",
            "arguments": {"query": "grounded verification"},
            "handle": handle,
        },
    }
    result = client.post("/mcp/p/" + handle, json=payload)
    assert result.status_code == 200
    body = result.json()
    assert body["result"]["structuredContent"]["paper_id"] == "paper-1"
    assert body["result"]["_meta"]["protocolVersion"] == "2026-07-28"

    wrong_handle = create_scope_handle("paper-2")
    mismatched = client.post(
        "/mcp/p/" + handle,
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_paper",
                "arguments": {"query": "grounded verification"},
                "handle": wrong_handle,
            },
        },
    )
    assert mismatched.status_code == 400


def test_openapi_export(tmp_path: Path) -> None:
    target = tmp_path / "openapi.json"
    schema: dict[str, Any] = export_openapi(target)
    assert "openapi" in schema
    assert "/v1/papers" in schema["paths"]
    assert target.exists()
