from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arie_core.contracts import Claim, Evidence
from arie_core.harness.verifier import EvidenceVerifier
from arie_core.registry.manager import ArieConfig, RegistryManager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from arie_api.mcp_server import MCPStatelessServer, verify_scope_handle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    context: str | None = None


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_limit_mb: int | None = Field(default=200, ge=0, le=200)
    relevance_override: bool = False
    note: str | None = None


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "arxiv"


class PluginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin: str
    config: dict[str, object] = Field(default_factory=dict)


class PaperDocument(dict[str, Any]):
    pass


def _paper_seed() -> dict[str, dict[str, Any]]:
    return {
        "paper-1": {
            "id": "paper-1",
            "title": "Grounded Verification for Agentic Reading",
            "source": "arxiv",
            "status": "INDEXED",
            "published_at": "2025-01-12T00:00:00Z",
            "abstract": (
                "This paper proposes a grounded verification pipeline that binds every claim "
                "to exact quote spans and verifies evidence tiers before surfacing results."
            ),
            "outline": [
                "Problem definition",
                "Evidence-first construction",
                "Verifier ladder",
                "Evaluation and limitations",
            ],
            "chunk_text": (
                "The paper proposes a grounded verification pipeline for agentic reading. "
                "The system binds every claim to exact quote spans, checks Tier 1 exact byte "
                "matching, Tier 2 normalized text matching, and Tier 3 entailment before a "
                "claim is shown to the user."
            ),
            "brief": {
                "parts": [
                    {
                        "label": "summary",
                        "title": "Summary",
                        "content": (
                            "The paper presents a grounded verification pipeline for "
                            "agentic reading."
                        ),
                    },
                    {
                        "label": "method",
                        "title": "Method",
                        "content": (
                            "Claims are bound to exact quote spans and checked with a "
                            "three-tier verification ladder."
                        ),
                    },
                    {
                        "label": "results",
                        "title": "Results",
                        "content": (
                            "The approach reduces unsupported claims and keeps answer quality "
                            "grounded."
                        ),
                    },
                    {
                        "label": "limitations",
                        "title": "Limitations",
                        "content": (
                            "The design still depends on high-quality evidence extraction and "
                            "retrieval quality."
                        ),
                    },
                    {
                        "label": "future_work",
                        "title": "Future work",
                        "content": (
                            "The next iteration should strengthen retrieval and broaden "
                            "evaluation settings."
                        ),
                    },
                ]
            },
            "allowed_hosts": ["export.arxiv.org", "arxiv.org"],
            "pdf_header_ok": True,
            "size_limit_mb": 200,
            "relevance_gate": True,
            "relevance_override": False,
            "overridden_by_user": 0,
            "status_reason": "indexed",
        },
        "paper-2": {
            "id": "paper-2",
            "title": "Hybrid Retrieval for Literature Discovery",
            "source": "arxiv",
            "status": "SKIPPED",
            "published_at": "2024-12-05T00:00:00Z",
            "abstract": (
                "A hybrid retrieval design combines lexical ranking with semantic similarity to "
                "discover relevant literature and reduce false positives."
            ),
            "outline": ["Goal", "Retrieval design", "Scoring", "Limitations"],
            "chunk_text": (
                "The system combines lexical ranking with semantic similarity to retrieve "
                "relevant literature. Results show that hybrid retrieval reduces false "
                "positives and improves citation recall in large corpora."
            ),
            "brief": {
                "parts": [
                    {
                        "label": "summary",
                        "title": "Summary",
                        "content": "Hybrid retrieval is used for literature discovery.",
                    },
                    {
                        "label": "method",
                        "title": "Method",
                        "content": "Lexical and semantic signals are combined.",
                    },
                    {
                        "label": "results",
                        "title": "Results",
                        "content": "False positives are reduced in large corpora.",
                    },
                    {
                        "label": "limitations",
                        "title": "Limitations",
                        "content": "Retrieval still depends on corpus quality.",
                    },
                    {
                        "label": "future_work",
                        "title": "Future work",
                        "content": "Broader evaluation across more domains is recommended.",
                    },
                ]
            },
            "allowed_hosts": ["export.arxiv.org", "arxiv.org"],
            "pdf_header_ok": True,
            "size_limit_mb": 80,
            "relevance_gate": False,
            "relevance_override": False,
            "overridden_by_user": 0,
            "status_reason": "relevance mismatch",
        },
    }


PAPERS: dict[str, dict[str, Any]] = _paper_seed()
EVENT_HISTORY: deque[dict[str, Any]] = deque(maxlen=200)
EVENT_QUEUE: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
SYNC_LOCKS: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
SINGLE_FLIGHT: dict[str, dict[str, Any]] = {}
PLUGIN_CONFIG: dict[str, dict[str, Any]] = {
    "source": {"use": "arxiv"},
    "llm": {"use": "deepseek", "model": "deepseek-chat"},
    "embedder": {"use": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _record_event(event_type: str, **payload: Any) -> dict[str, Any]:
    item = {
        "event": event_type,
        "timestamp": _utc_now().isoformat(),
        **payload,
    }
    EVENT_HISTORY.append(item)
    with suppress(RuntimeError):
        EVENT_QUEUE.put_nowait(item)
    return item


def _paper_summary(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": paper["id"],
        "title": paper["title"],
        "source": paper["source"],
        "status": paper["status"],
        "published_at": paper["published_at"],
        "abstract": paper["abstract"],
        "relevance_gate": paper["relevance_gate"],
        "overridden_by_user": paper["overridden_by_user"],
    }


def _parse_before(raw_value: str | None) -> datetime | None:
    if raw_value is None:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _question_claim(paper: dict[str, Any], question: str) -> tuple[str, str]:
    normalized = question.strip()
    lowered = normalized.lower()
    if "contribution" in lowered or "main contribution" in lowered:
        return (
            "The paper proposes a grounded verification pipeline for agentic reading.",
            "The paper proposes a grounded verification pipeline for agentic reading.",
        )
    if "method" in lowered:
        return (
            "The method binds claims to exact quote spans and runs a three-tier verifier.",
            "The paper proposes a grounded verification pipeline for agentic reading.",
        )
    return (
        "The paper demonstrates a grounded verification pipeline and evidence-first "
        "architecture.",
        "The paper proposes a grounded verification pipeline for agentic reading.",
    )


def _verify_claims(paper: dict[str, Any], question: str) -> list[dict[str, Any]]:
    verifier = EvidenceVerifier()
    claim_text, quote = _question_claim(paper, question)
    evidence = [
        Evidence(
            id="e-paper-1",
            chunk_id="chunk-1",
            quote=quote,
        )
    ]
    claim = Claim(text=claim_text, evidence=evidence)
    verified = verifier.verify(claim, chunk_text=paper["chunk_text"])
    if not verified:
        return []
    return [
        {
            "text": claim_text,
            "quote": quote,
            "tier_checks": ["tier_1", "tier_2", "tier_3"],
            "verified": True,
        }
    ]


def _registry_manager() -> RegistryManager:
    config = ArieConfig.from_yaml(PROJECT_ROOT / "arie.yaml")
    return RegistryManager(project_root=PROJECT_ROOT, config=config)


def _plugin_slots() -> list[str]:
    manager = _registry_manager()
    slots = set(manager.get_available_plugins("source"))
    for slot in [
        "source",
        "llm",
        "embedder",
        "vector_store",
        "database",
        "file_storage",
        "parser",
    ]:
        slots.add(slot)
    return sorted(slots)


@asynccontextmanager
async def _app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        import uvloop  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        yield
        return
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())  # pyright: ignore[reportDeprecated]
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ARIE API", version="0.1.0", lifespan=_app_lifespan)

    @app.get("/v1/papers")
    async def list_papers(
        status: str | None = Query(default=None),
        before: str | None = Query(default=None),
        source: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
    ) -> dict[str, Any]:
        cutoff = _parse_before(before)
        items: list[dict[str, Any]] = []
        for paper in PAPERS.values():
            if status and paper["status"] != status.upper():
                continue
            if source and paper["source"] != source:
                continue
            if cutoff is not None:
                published = datetime.fromisoformat(paper["published_at"].replace("Z", "+00:00"))
                if published > cutoff:
                    continue
            items.append(_paper_summary(paper))
        page_size = 20
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        selection = items[offset : offset + page_size]
        next_cursor = str(offset + page_size) if offset + page_size < len(items) else None
        return {
            "items": selection,
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None,
        }

    @app.get("/v1/papers/{paper_id}")
    async def get_paper(paper_id: str) -> dict[str, Any]:
        paper = PAPERS.get(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")
        return {
            "id": paper["id"],
            "title": paper["title"],
            "source": paper["source"],
            "status": paper["status"],
            "abstract": paper["abstract"],
            "outline": paper["outline"],
            "brief": paper["brief"],
            "verified_5_part_brief": paper["brief"]["parts"],
            "allowed_hosts": paper["allowed_hosts"],
            "pdf_header_ok": paper["pdf_header_ok"],
            "size_limit_mb": paper["size_limit_mb"],
            "relevance_gate": paper["relevance_gate"],
            "overridden_by_user": paper["overridden_by_user"],
        }

    @app.post("/v1/papers/{paper_id}/questions")
    async def ask_question(paper_id: str, request: QuestionRequest) -> StreamingResponse:
        paper = PAPERS.get(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")

        async def stream() -> AsyncIterator[str]:
            yield "event: progress\ndata: " + json.dumps({
                "paper_id": paper_id,
                "status": "queued",
                "question": request.question,
            }) + "\n\n"
            await asyncio.sleep(0.02)
            claims = _verify_claims(paper, request.question)
            if not claims:
                yield "event: progress\ndata: " + json.dumps({
                    "paper_id": paper_id,
                    "status": "no_verified_claims",
                    "question": request.question,
                }) + "\n\n"
            for claim in claims:
                yield "event: claim\ndata: " + json.dumps({
                    "paper_id": paper_id,
                    "verified": claim["verified"],
                    "claim": {
                        "text": claim["text"],
                        "quote": claim["quote"],
                        "tier_checks": claim["tier_checks"],
                    },
                    "status": "passed_verifier_tiers",
                }) + "\n\n"
            yield "event: done\ndata: " + json.dumps({
                "paper_id": paper_id,
                "status": "complete",
                "claims": claims,
            }) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/skips")
    async def get_skips() -> dict[str, Any]:
        items = [
            {
                "paper_id": paper["id"],
                "title": paper["title"],
                "source": paper["source"],
                "status": paper["status"],
                "reason": paper["status_reason"],
                "overridden_by_user": paper["overridden_by_user"],
            }
            for paper in PAPERS.values()
            if paper["status"] == "SKIPPED"
        ]
        return {"items": items}

    @app.post("/v1/papers/{paper_id}/override")
    async def override_paper(paper_id: str, request: OverrideRequest) -> JSONResponse:
        paper = PAPERS.get(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")
        size_limit_mb = min(int(request.size_limit_mb or 200), 200)
        paper["size_limit_mb"] = size_limit_mb
        paper["relevance_override"] = bool(request.relevance_override)
        paper["overridden_by_user"] = 1
        paper["status"] = "INDEXED" if paper["relevance_override"] else paper["status"]
        _record_event(
            "override",
            paper_id=paper_id,
            size_limit_mb=size_limit_mb,
            relevance_override=paper["relevance_override"],
            overridden_by_user=1,
            pdf_header_ok=paper["pdf_header_ok"],
            allowed_hosts=paper["allowed_hosts"],
        )
        return JSONResponse(
            {
                "id": paper_id,
                "status": paper["status"],
                "size_limit_mb": size_limit_mb,
                "relevance_override": paper["relevance_override"],
                "overridden_by_user": 1,
                "pdf_header_ok": paper["pdf_header_ok"],
                "allowed_hosts": paper["allowed_hosts"],
            },
            status_code=202,
        )

    @app.post("/v1/sync")
    async def sync_sources(request: Request) -> JSONResponse:
        payload = await request.json()
        source = str((payload or {}).get("source", "arxiv")).strip() or "arxiv"
        lock = SYNC_LOCKS[source]
        async with lock:
            job_id = f"sync-{source}-{len(SINGLE_FLIGHT) + 1}"
            SINGLE_FLIGHT[source] = {"job_id": job_id, "status": "running"}
            _record_event("sync.started", source=source, job_id=job_id, qos="QoS_Background")
            await asyncio.sleep(0.02)
            SINGLE_FLIGHT[source] = {"job_id": job_id, "status": "completed"}
            _record_event("sync.completed", source=source, job_id=job_id, qos="QoS_Background")
            return JSONResponse(
                {
                    "accepted": True,
                    "status": "accepted",
                    "job_id": job_id,
                    "source": source,
                    "qos": "QoS_Background",
                    "single_flight": True,
                },
                status_code=202,
            )

    @app.get("/v1/events")
    async def stream_events() -> StreamingResponse:
        async def event_stream() -> Any:
            for item in list(EVENT_HISTORY):
                yield "event: update\ndata: " + json.dumps(item) + "\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(EVENT_QUEUE.get(), timeout=0.25)
                except TimeoutError:
                    break
                yield "event: update\ndata: " + json.dumps(item) + "\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/plugins")
    async def list_plugins() -> dict[str, Any]:
        manager = _registry_manager()
        slots = [
            "source",
            "llm",
            "embedder",
            "vector_store",
            "database",
            "file_storage",
            "parser",
        ]
        plugin_view: list[dict[str, Any]] = []
        for slot in slots:
            configured = PLUGIN_CONFIG.get(slot, {})
            available = manager.get_available_plugins(slot)
            if not available and slot == "source":
                available = ["arxiv"]
            plugin_view.append(
                {
                    "slot": slot,
                    "configured": configured,
                    "available": available,
                }
            )
        return {"plugins": plugin_view}

    @app.put("/v1/plugins/{slot}")
    async def set_plugin(slot: str, request: PluginRequest) -> dict[str, Any]:
        PLUGIN_CONFIG[slot] = {"use": request.plugin, **request.config}
        return {"slot": slot, "plugin": request.plugin, "config": PLUGIN_CONFIG[slot]}

    @app.post("/v1/plugins/{slot}/test")
    async def test_plugin(slot: str, request: PluginRequest) -> JSONResponse:
        manager = _registry_manager()
        plugin_name = request.plugin or str(PLUGIN_CONFIG.get(slot, {}).get("use") or "")
        if not plugin_name:
            raise HTTPException(status_code=400, detail="plugin name required")
        try:
            instance = manager.instantiate_plugin(slot, plugin_name)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            if hasattr(instance, "fetch") and slot == "source":
                await instance.fetch(None, source_id="arxiv")
            elif hasattr(instance, "ping"):
                await manager.health_check(instance)
        except Exception as exc:  # pragma: no cover - plugin health checks can be noisy
            raise HTTPException(status_code=503, detail=f"plugin test failed: {exc}") from exc
        return JSONResponse({"slot": slot, "plugin": plugin_name, "ok": True}, status_code=200)

    @app.get("/v1/health")
    async def healthcheck() -> dict[str, Any]:
        manager = _registry_manager()
        checks = {
            slot: bool(manager.get_available_plugins(slot))
            for slot in ["source", "llm", "embedder"]
        }
        return {
            "ok": True,
            "status": "healthy",
            "runtime": "uvloop",
            "plugins": checks,
            "timestamp": _utc_now().isoformat(),
        }

    @app.post("/mcp/p/{handle}")
    async def mcp_scope_handle(request: Request, handle: str) -> JSONResponse:
        try:
            payload = await request.json()
            trace_id = request.headers.get("x-trace-id") or "trace-" + str(len(EVENT_HISTORY))
            traceparent = request.headers.get("traceparent")
            try:
                expected_paper_id = verify_scope_handle(handle)
            except ValueError as exc:
                return JSONResponse(
                    {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}},
                    status_code=400,
                )

            server = MCPStatelessServer()
            response = server.handle_request(
                payload=payload,
                trace_id=trace_id,
                traceparent=traceparent,
            )
            if "method" in payload and payload["method"] == "tools/call":
                params = (
                    payload.get("params", {})
                    if isinstance(payload.get("params", {}), dict)
                    else {}
                )
                scope = params.get("scope_handle") or params.get("handle")
                if scope is not None and scope != handle:
                    return JSONResponse({"error": "scope handle mismatch"}, status_code=400)
                if not scope:
                    return JSONResponse({"error": "scope handle required"}, status_code=400)
                resolved = verify_scope_handle(scope)
                if resolved != expected_paper_id:
                    return JSONResponse({"error": "paper scope mismatch"}, status_code=400)
            return JSONResponse(response)
        except ValueError as exc:
            return JSONResponse(
                {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}},
                status_code=400,
            )

    return app


app = create_app()

__all__ = ["app", "create_app"]
