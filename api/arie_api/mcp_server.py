from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

DefaultPaperStore: Final[dict[str, dict[str, Any]]] = {
    "paper-1": {
        "id": "paper-1",
        "title": "Grounded Verification for Agentic Reading",
        "source": "arxiv",
        "abstract": (
            "This paper proposes an evidence-first pipeline that binds every claim "
            "to exact quote spans and verifies tiers before surfacing results."
        ),
        "outline": [
            "Problem definition",
            "Evidence-first construction",
            "Verifier ladder",
            "Evaluation and limitations",
        ],
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "page": 1,
                "char_start": 0,
                "char_end": 320,
                "text": (
                    "This paper proposes a grounded verification pipeline for agentic "
                    "reading. The system binds every claim to exact quote spans, "
                    "checks Tier 1 exact byte matching, Tier 2 normalized text "
                    "matching, and Tier 3 entailment before a claim is shown to the "
                    "user."
                ),
            }
        ],
    },
    "paper-2": {
        "id": "paper-2",
        "title": "Hybrid Retrieval for Literature Discovery",
        "source": "arxiv",
        "abstract": "A hybrid retrieval design combines lexical and semantic signals.",
        "outline": ["Goal", "Retrieval design", "Scoring", "Limitations"],
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "page": 2,
                "char_start": 60,
                "char_end": 240,
                "text": (
                    "The system combines lexical ranking with semantic similarity to "
                    "retrieve relevant literature. Results show that hybrid retrieval "
                    "reduces false positives and improves citation recall in large "
                    "corpora."
                ),
            }
        ],
    },
}

_MCP_PROTOCOL_VERSION: Final[str] = "2026-07-28"
_MCP_SCOPE_TTL_SECONDS: Final[int] = 900


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    remainder = len(value) % 4
    if remainder:
        value += "=" * (4 - remainder)
    return base64.urlsafe_b64decode(value.encode("ascii"))


def create_scope_handle(
    paper_id: str,
    *,
    secret: str | None = None,
    now: datetime | None = None,
    ttl_seconds: int = _MCP_SCOPE_TTL_SECONDS,
) -> str:
    candidate = str(paper_id).strip()
    if not candidate:
        raise ValueError("paper_id must not be empty")
    issued_at = now or datetime.now(UTC)
    expiration = issued_at + timedelta(seconds=ttl_seconds)
    payload = {
        "paper_id": candidate,
        "exp": int(expiration.timestamp()),
        "iat": int(issued_at.timestamp()),
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signing_secret = (secret or os.getenv("ARIE_MCP_SECRET") or "arie-default-mcp-secret").encode(
        "utf-8"
    )
    signature = hmac.new(signing_secret, payload_json, hashlib.sha256).digest()
    return f"{_b64url_encode(payload_json)}.{_b64url_encode(signature)}"


def verify_scope_handle(
    handle: str,
    *,
    secret: str | None = None,
    now: datetime | None = None,
) -> str:
    pieces = str(handle).split(".")
    if len(pieces) != 2:
        raise ValueError("invalid scope handle")
    payload_data = _b64url_decode(pieces[0])
    signature = _b64url_decode(pieces[1])
    signing_secret = (secret or os.getenv("ARIE_MCP_SECRET") or "arie-default-mcp-secret").encode(
        "utf-8"
    )
    expected = hmac.new(signing_secret, payload_data, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("scope handle signature mismatch")
    try:
        payload = cast(dict[str, Any], json.loads(payload_data.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise ValueError("invalid scope handle payload") from exc
    expiration = int(payload.get("exp", 0))
    current_epoch = int((now or datetime.now(UTC)).timestamp())
    if current_epoch > expiration:
        raise ValueError("scope handle expired")
    paper_id = str(payload.get("paper_id", "")).strip()
    if not paper_id:
        raise ValueError("scope handle missing paper_id")
    return paper_id


class MCPStatelessServer:
    """Minimal MCP endpoint implementation that binds paper_id to a signed scope handle."""

    def __init__(self, *, paper_store: dict[str, dict[str, Any]] | None = None) -> None:
        self.paper_store = paper_store or DefaultPaperStore

    @staticmethod
    def _trace_meta(trace_id: str | None, traceparent: str | None) -> dict[str, str]:
        meta: dict[str, str] = {"protocolVersion": _MCP_PROTOCOL_VERSION}
        if trace_id:
            meta["trace_id"] = trace_id
        if traceparent:
            meta["traceparent"] = traceparent
        return meta

    def _tool_schema(self, name: str) -> dict[str, Any]:
        base_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }
        if name == "search_paper":
            base_schema["properties"] = {
                "query": {"type": "string", "description": "Search query within the scoped paper."}
            }
            base_schema["required"] = ["query"]
        elif name == "read_section":
            base_schema["properties"] = {
                "section":
                    {"type": "string", "description": "Section name such as overview or methods."}
            }
            base_schema["required"] = ["section"]
        elif name == "get_table":
            base_schema["properties"] = {
                "table_name": {"type": "string", "description": "Table to retrieve."}
            }
            base_schema["required"] = ["table_name"]
        elif name == "get_page":
            base_schema["properties"] = {
                "page": {"type": "integer", "description": "Page number to fetch."}
            }
            base_schema["required"] = ["page"]
        return base_schema

    def list_tools(self) -> list[dict[str, Any]]:
        descriptions = [
            {
                "name": "search_paper",
                "description": "Search the scoped paper and return relevant chunks.",
            },
            {
                "name": "read_section",
                "description": "Read an outline section for the scoped paper.",
            },
            {
                "name": "get_table",
                "description": "Return a structured table excerpt from the scoped paper.",
            },
            {
                "name": "get_page",
                "description": "Return a page excerpt from the scoped paper.",
            },
        ]
        ordered = sorted(descriptions, key=lambda tool: tool["name"])
        return [
            {
                **tool,
                "inputSchema": self._tool_schema(tool["name"]),
            }
            for tool in ordered
        ]

    def _paper_from_handle(self, handle: str) -> dict[str, Any]:
        paper_id = verify_scope_handle(handle)
        paper = self.paper_store.get(paper_id)
        if paper is None:
            raise KeyError(f"unknown scoped paper: {paper_id}")
        return paper

    def _structured_result(
        self,
        *,
        paper: dict[str, Any],
        tool_name: str,
        text: str,
        chunk_id: str,
        page: int,
        char_start: int,
        char_end: int,
    ) -> dict[str, Any]:
        return {
            "paper_id": str(paper["id"]),
            "title": str(paper["title"]),
            "source": str(paper["source"]),
            "tool": tool_name,
            "chunk_id": chunk_id,
            "page": page,
            "char_start": char_start,
            "char_end": char_end,
            "text": text,
        }

    def call_tool(
        self,
        *,
        handle: str,
        name: str,
        arguments: dict[str, Any],
        trace_id: str | None = None,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        paper = self._paper_from_handle(handle)
        content: list[dict[str, str]] = []
        structured: dict[str, Any]
        chunk = paper["chunks"][0]
        if name == "search_paper":
            query = str(arguments.get("query", "")).strip()
            text = f"Search results for {query!r} in {paper['title']}: {chunk['text']}"
            content = [{"type": "text", "text": text}]
            structured = self._structured_result(
                paper=paper,
                tool_name=name,
                text=text,
                chunk_id=str(chunk["chunk_id"]),
                page=int(chunk["page"]),
                char_start=int(chunk["char_start"]),
                char_end=int(chunk["char_end"]),
            )
        elif name == "read_section":
            section = str(arguments.get("section", "overview")).strip() or "overview"
            section_text = (
                paper["abstract"]
                if section.lower() in {"abstract", "overview"}
                else "; ".join(paper["outline"])
            )
            text = f"{section}: {section_text}"
            content = [{"type": "text", "text": text}]
            structured = self._structured_result(
                paper=paper,
                tool_name=name,
                text=text,
                chunk_id=str(chunk["chunk_id"]),
                page=int(chunk["page"]),
                char_start=int(chunk["char_start"]),
                char_end=int(chunk["char_end"]),
            )
        elif name == "get_table":
            table_name = str(arguments.get("table_name", "results")).strip() or "results"
            text = f"Table {table_name}: accuracy=0.91; recall=0.87; sample_count=120"
            content = [{"type": "text", "text": text}]
            structured = self._structured_result(
                paper=paper,
                tool_name=name,
                text=text,
                chunk_id=str(chunk["chunk_id"]),
                page=int(chunk["page"]),
                char_start=int(chunk["char_start"]),
                char_end=int(chunk["char_end"]),
            )
        elif name == "get_page":
            page_number = int(arguments.get("page", 1))
            text = f"Page {page_number}: {chunk['text']}"
            content = [{"type": "text", "text": text}]
            structured = self._structured_result(
                paper=paper,
                tool_name=name,
                text=text,
                chunk_id=str(chunk["chunk_id"]),
                page=page_number,
                char_start=int(chunk["char_start"]),
                char_end=int(chunk["char_end"]),
            )
        else:
            raise ValueError(f"unsupported tool: {name}")
        return {
            "content": content,
            "structuredContent": structured,
            "_meta": self._trace_meta(trace_id, traceparent),
        }

    def handle_request(
        self,
        *,
        payload: dict[str, Any],
        trace_id: str | None = None,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        request_id = payload.get("id", str(uuid.uuid4()))
        method = payload.get("method")
        if method in {"initialize", "ping"}:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "arie-api", "version": "0.1.0"},
                    "_meta": self._trace_meta(trace_id, traceparent),
                },
            }
        if method == "tools/list":
            result = {"tools": self.list_tools(), "_meta": self._trace_meta(trace_id, traceparent)}
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        if method == "tools/call":
            params_raw: Any = payload.get("params", {})
            if not isinstance(params_raw, dict):
                raise ValueError("params must be an object")
            params = cast(dict[str, Any], params_raw)
            tool_name = str(params.get("name", "")).strip()
            if not tool_name:
                raise ValueError("tool name is required")
            arguments_raw: Any = params.get("arguments", {})
            if arguments_raw is None:
                arguments_raw = {}
            if not isinstance(arguments_raw, dict):
                raise ValueError("tool arguments must be an object")
            arguments = cast(dict[str, Any], arguments_raw)
            raw_handle = params.get("handle") or params.get("scope_handle") or ""
            handle = str(raw_handle)
            if not handle:
                raise ValueError("scope handle is required")
            tool_result = self.call_tool(
                handle=handle,
                name=tool_name,
                arguments=arguments,
                trace_id=trace_id,
                traceparent=traceparent,
            )
            return {"jsonrpc": "2.0", "id": request_id, "result": tool_result}
        raise ValueError(f"unsupported MCP method: {method}")


__all__ = [
    "DefaultPaperStore",
    "MCPStatelessServer",
    "create_scope_handle",
    "verify_scope_handle",
]
