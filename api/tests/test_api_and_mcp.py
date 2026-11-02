from __future__ import annotations

from arie_api.export_openapi import export_openapi
from arie_api.mcp_server import create_scope_handle, verify_scope_handle
from arie_api.routes import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_papers_feed_and_detail() -> None:
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
    assert len(body["verified_5_part_brief"]) == 5


def test_question_stream_emits_verified_claims_only() -> None:
    response = client.post(
        "/v1/papers/paper-1/questions",
        json={"question": "What is the main contribution?"},
    )
    assert response.status_code == 200
    text = response.text
    assert "event: progress" in text
    assert "event: claim" in text
    assert "passed_verifier_tiers" in text
    assert "event: done" in text


def test_override_preserves_protected_fields() -> None:
    response = client.post(
        "/v1/papers/paper-2/override",
        json={"size_limit_mb": 42, "relevance_override": True},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["overridden_by_user"] == 1
    assert body["size_limit_mb"] == 42
    assert body["relevance_override"] is True
    assert body["allowed_hosts"] == ["export.arxiv.org", "arxiv.org"]

    detail = client.get("/v1/papers/paper-2")
    assert detail.status_code == 200
    assert detail.json()["size_limit_mb"] == 42
    assert detail.json()["allowed_hosts"] == ["export.arxiv.org", "arxiv.org"]


def test_sync_events_plugins_and_health() -> None:
    sync = client.post("/v1/sync", json={"source": "arxiv"})
    assert sync.status_code == 202
    assert sync.json()["qos"] == "QoS_Background"

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


def test_mcp_scope_handle_isolated_and_ordered() -> None:
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


def test_openapi_export(tmp_path) -> None:
    target = tmp_path / "openapi.json"
    schema = export_openapi(target)
    assert "openapi" in schema
    assert "/v1/papers" in schema["paths"]
    assert target.exists()
