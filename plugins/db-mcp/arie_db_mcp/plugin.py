from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from arie_core.plugin_api import plugin


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _as_object_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast(list[object], value)


class StatelessMcpClient:
    """Stateless MCP client wrapper for SQL tool calls."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: str = "streamable-http",
        headers: Mapping[str, str] | None = None,
        allowed_hosts: Sequence[str] | None = None,
        command: Sequence[str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.transport = str(transport).strip().lower()
        self.headers = dict(headers or {})
        self.allowed_hosts = tuple(
            str(host)
            .strip()
            .lower()
            .removeprefix("http://")
            .removeprefix("https://")
            .split("/", 1)[0]
            .split(":", 1)[0]
            for host in (allowed_hosts or ())
            if str(host).strip()
        )
        self.command = tuple(str(part) for part in (command or ()))

    @staticmethod
    def _host_from_url(url: str | None) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        if parsed.hostname:
            return parsed.hostname.lower()
        raw_host = url.split("//", 1)[-1].split("/", 1)[0]
        return raw_host.split(":", 1)[0].lower()

    def _ensure_allowed_host(self, url: str | None) -> None:
        if not url:
            return
        host = self._host_from_url(url)
        if self.allowed_hosts and host and host not in self.allowed_hosts:
            raise ValueError(f"Host {host!r} not allowed for MCP SQL server")

    async def _http_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any:
        if not self.base_url:
            raise ValueError("base_url is required")
        self._ensure_allowed_host(self.base_url)
        url = self.base_url.rstrip("/") + path
        self._ensure_allowed_host(url)

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.request(
                method,
                url,
                json=json_body,
                headers=self.headers,
            )
            response.raise_for_status()
            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

    async def _stdio_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any:
        del path
        if not self.command:
            raise ValueError("command is required for stdio transport")

        payload: dict[str, object] = {
            "jsonrpc": "2.0",
            "id": f"mcp-{uuid4().hex}",
            "method": method,
            "params": dict(json_body or {}),
        }

        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None
        stdout, stderr = await process.communicate(json.dumps(payload).encode("utf-8"))

        if process.returncode not in (0, None):
            error_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MCP stdio process failed: {error_text or process.returncode}")
        if not stdout:
            return None

        response = json.loads(stdout.decode("utf-8"))
        if isinstance(response, Mapping):
            response_mapping = cast(Mapping[str, Any], response)
            if "result" in response_mapping:
                result_value = response_mapping.get("result")
                if result_value is not None:
                    return result_value
            if "content" in response_mapping:
                content_value = response_mapping.get("content")
                if content_value is not None:
                    return content_value
        return cast(Any, response)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> Any:
        if self.transport in {"streamable-http", "http"}:
            return await self._http_request(method, path, json_body=json_body)
        if self.transport == "stdio":
            return await self._stdio_request(method, path, json_body=json_body)
        raise ValueError(f"Unsupported MCP transport: {self.transport!r}")

    async def list_tools(self) -> list[str]:
        if self.transport == "stdio":
            payload = await self._request("tools/list", "/tools", json_body={})
        else:
            payload = await self._request("GET", "/tools")

        payload_mapping = _as_mapping(payload)
        if payload_mapping is None:
            return []

        tools_value = _as_object_list(payload_mapping.get("tools"))
        if tools_value is None:
            return []

        names: list[str] = []
        for tool in tools_value:
            tool_mapping = _as_mapping(tool)
            if tool_mapping is None:
                continue
            name = tool_mapping.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        if self.transport == "stdio":
            payload = await self._request(
                "tools/call",
                "/tools/call",
                json_body={"name": name, **kwargs},
            )
        else:
            payload = await self._request(
                "POST",
                "/tools/call",
                json_body={"name": name, **kwargs},
            )

        payload_mapping = _as_mapping(payload)
        if payload_mapping is None:
            return payload

        if "result" in payload_mapping:
            return payload_mapping["result"]
        if "content" in payload_mapping:
            return payload_mapping["content"]
        return payload


@plugin(slot="database", name="mcp-sql", api=">=1,<2")
class McpSqlDatabasePlugin:
    """Database plugin that wraps a stateless MCP SQL server."""

    allowed_hosts = ("127.0.0.1", "localhost")

    def __init__(
        self,
        *,
        server_url: str | None = None,
        transport: str = "streamable-http",
        allowed_hosts: Sequence[str] | None = None,
        command: Sequence[str] | None = None,
    ) -> None:
        self.server_url = server_url or "http://127.0.0.1:8000/mcp"
        self.client = StatelessMcpClient(
            base_url=self.server_url,
            transport=transport,
            allowed_hosts=allowed_hosts or self.allowed_hosts,
            command=command,
        )

    async def startup(self) -> None:
        required = {"sql_query", "sql_execute", "sql_transaction"}
        tools = set(await self.client.list_tools())
        missing = sorted(required - tools)
        if missing:
            raise RuntimeError(f"Missing MCP SQL tools: {missing}")

    async def get_paper_chunks(self, *, paper_id: str) -> list[dict[str, Any]]:
        result = await self.client.call_tool(
            "sql_query",
            sql="SELECT * FROM paper_chunks WHERE paper_id = ?",
            params=[paper_id],
        )
        rows_value: object = result
        rows_mapping = _as_mapping(rows_value)
        if rows_mapping is not None:
            rows_value = rows_mapping.get("rows") or []
        rows = _as_object_list(rows_value) or []
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row_mapping = _as_mapping(row)
            if row_mapping is None:
                continue
            candidate = dict(row_mapping)
            if candidate.get("paper_id") == paper_id:
                filtered.append(candidate)
        return filtered

    async def upsert_paper_and_chunks(
        self,
        *,
        paper_id: str,
        paper: Mapping[str, Any],
        chunks: Sequence[Mapping[str, Any]],
    ) -> None:
        paper_row: dict[str, Any] = {
            "paper_id": paper_id,
            "title": paper.get("title"),
            "source": paper.get("source"),
            "abstract": paper.get("abstract"),
            "json": json.dumps(paper),
        }
        chunk_rows: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_rows.append(
                {
                    "paper_id": paper_id,
                    "chunk_index": chunk.get("chunk_index"),
                    "content": chunk.get("content"),
                    "metadata": json.dumps(chunk.get("metadata") or {}),
                }
            )

        records = [paper_row, *chunk_rows]
        payload = {
            "sql_transaction": {
                "statements": [
                    (
                        "INSERT INTO papers (paper_id, title, source, abstract, json) "
                        "VALUES (:paper_id, :title, :source, :abstract, :json) "
                        "ON CONFLICT (paper_id) DO UPDATE SET "
                        "title = EXCLUDED.title, source = EXCLUDED.source, "
                        "abstract = EXCLUDED.abstract, json = EXCLUDED.json"
                    ),
                    (
                        "INSERT INTO paper_chunks (paper_id, chunk_index, content, metadata) "
                        "VALUES (:paper_id, :chunk_index, :content, :metadata) "
                        "ON CONFLICT (paper_id, chunk_index) DO UPDATE SET "
                        "content = EXCLUDED.content, metadata = EXCLUDED.metadata"
                    ),
                ],
                "params": records,
            }
        }
        await self.client.call_tool("sql_transaction", **payload)

    async def query(
        self,
        *,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        result = await self.client.call_tool("sql_query", sql=sql, params=list(params or ()))
        value: object = result
        mapping = _as_mapping(value)
        if mapping is not None:
            value = mapping.get("rows") or []
        rows = _as_object_list(value) or []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            row_mapping = _as_mapping(row)
            if row_mapping is not None:
                normalized.append(dict(row_mapping))
        return normalized


__all__ = ["McpSqlDatabasePlugin", "StatelessMcpClient"]
