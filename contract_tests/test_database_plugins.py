"""Postgres-profile and MCP-SQL database plugins: paper isolation (I1) and declared hosts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
REAL_CLIENT = httpx.AsyncClient


def database(name: str, **kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin("database", name, **kwargs)


@pytest.mark.asyncio
async def test_postgres_profile_isolates_papers_and_bridges_notify(tmp_path: Path) -> None:
    db = database("postgres", dsn=f"sqlite:///{tmp_path / 'pg.sqlite3'}")
    for paper_id, texts in (("p1", ["alpha", "beta"]), ("p2", ["gamma"])):
        await db.upsert_paper_and_chunks(
            {"paper_id": paper_id, "title": paper_id, "source": "arxiv", "pdf_url": "u"},
            [
                {"chunk_id": f"{paper_id}-{i}", "paper_id": paper_id, "text": text, "order": i}
                for i, text in enumerate(texts)
            ],
        )
    rows = await db.get_paper_chunks("p1")
    assert len(rows) == 2 and {row["paper_id"] for row in rows} == {"p1"}

    received: list[dict[str, object]] = []
    unsubscribe = db.listen("paper_indexed", received.append)
    db.notify("paper_indexed", {"paper_id": "p1"})
    unsubscribe()
    db.notify("paper_indexed", {"paper_id": "p2"})
    assert received == [{"paper_id": "p1"}]


def mcp_server(
    monkeypatch: pytest.MonkeyPatch, tools: list[str], rows: list[dict[str, object]]
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/tools"):
            return httpx.Response(200, json={"tools": [{"name": name} for name in tools]})
        body = json.loads(request.content)
        return httpx.Response(200, json={"result": {"rows": rows, "echo": body["name"]}})

    def client(**kwargs: Any) -> httpx.AsyncClient:
        assert kwargs.get("follow_redirects") is False
        return REAL_CLIENT(timeout=kwargs.get("timeout"), transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


@pytest.mark.asyncio
async def test_mcp_sql_requires_sql_tools_and_post_filters_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, object]] = [
        {"paper_id": "p1", "content": "ours"},
        {"paper_id": "p2", "content": "a faulty server leaked this row"},
    ]
    seen = mcp_server(monkeypatch, ["sql_query", "sql_execute", "sql_transaction"], rows)
    db = database("mcp-sql", server_url="http://127.0.0.1:8000/mcp")
    await db.startup()
    chunks = await db.get_paper_chunks(paper_id="p1")
    assert chunks == [{"paper_id": "p1", "content": "ours"}]
    assert {request.url.host for request in seen} == {"127.0.0.1"}

    mcp_server(monkeypatch, ["sql_query"], rows)
    with pytest.raises(RuntimeError, match="sql_execute"):
        await db.startup()


@pytest.mark.asyncio
async def test_mcp_sql_refuses_undeclared_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = mcp_server(monkeypatch, [], [])
    db = database("mcp-sql", server_url="https://sql.example.com/mcp")
    with pytest.raises(ValueError, match="not allowed"):
        await db.query(sql="SELECT 1")
    assert seen == []
