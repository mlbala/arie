from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, cast
from urllib.parse import quote

from arie_core.harness.intent import DialogueTurn
from arie_core.harness.orchestrator import answer_question
from arie_core.harness.reader import EvidenceChunk
from arie_core.plugin_api import LLM
from arie_core.registry.sync import SyncConfig
from arie_core.sched import supervised
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from arie_api.mcp_server import MCPStatelessServer, verify_scope_handle
from arie_api.monitor import StorageMonitor
from arie_api.runtime import Document, ResearchRuntime, context

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QuestionTurn:
    """Bounded, untrusted pointers to earlier questions in the selected paper's chat."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    question: Annotated[str, Field(min_length=1, max_length=700)]
    source_ids: Annotated[
        tuple[Annotated[str, Field(max_length=256)], ...], Field(max_length=8)
    ] = ()


class QuestionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    context: str | None = Field(default=None, max_length=2000)
    history: tuple[QuestionTurn, ...] = Field(default=(), max_length=4)


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = "all"
    settings: SyncConfig | None = None
    resume_job_id: str | None = None


@dataclass(frozen=True)
class ImportRequest:
    """Validate the public import boundary without a mutable runtime model."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    url: Annotated[str, Field(min_length=1, max_length=2048)]


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size_limit_mb: int | None = Field(default=25, ge=0, le=25)
    relevance_override: bool = False
    note: str | None = None


class PluginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plugin: str
    config: dict[str, object] = Field(default_factory=dict)


def _event(kind: str, **payload: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


def create_app(runtime: ResearchRuntime | None = None) -> FastAPI:
    service = runtime
    monitor: StorageMonitor | None = None

    def active() -> ResearchRuntime:
        if service is None:
            raise HTTPException(status_code=503, detail="Research service is starting")
        return service

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        nonlocal service
        service = service or ResearchRuntime(PROJECT_ROOT)
        await service.initialize()
        async with asyncio.TaskGroup() as group:
            worker = group.create_task(supervised(service.run)())
            imports = group.create_task(supervised(service.run_imports)())
            try:
                yield
            finally:
                worker.cancel()
                imports.cancel()
                service.close()

    app = FastAPI(title="ARIE API", version="0.1.0", lifespan=lifespan)

    async def document(paper_id: str) -> Document:
        paper = await active().get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "ARIE API", "status": "ok", "version": "0.1.0"}

    @app.get("/v1/papers")
    async def list_papers(
        status: str | None = None,
        source: str | None = None,
        cursor: int = Query(default=0, ge=0),
        month: str | None = Query(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"),
        year: int | None = Query(default=None, ge=1991, le=9998),
        months: int | None = Query(default=None, ge=1, le=120),
        start_month: str | None = Query(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"),
        end_month: str | None = Query(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"),
        domain: str | None = None,
        search: str | None = Query(default=None, max_length=200),
        sort: Literal["newest", "oldest", "domain"] = "newest",
    ) -> Document:
        try:
            return await active().list_papers(
                cursor,
                status,
                source,
                month,
                domain,
                search,
                sort,
                year=year,
                months=months,
                start_month=start_month,
                end_month=end_month,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/papers/{paper_id:path}/pdf")
    async def paper_pdf(paper_id: str) -> FileResponse:
        paper = await document(paper_id)

        def resolve_pdf() -> Path:
            path = Path(str(paper.get("file_path", ""))).resolve()
            if not path.is_relative_to(active().storage_root.resolve()) or not path.is_file():
                raise HTTPException(status_code=404, detail="Extracted PDF is unavailable")
            return path

        path = await asyncio.to_thread(resolve_pdf)
        return FileResponse(path, media_type="application/pdf")

    @app.get("/v1/papers/{paper_id:path}/pages/{page_number}")
    async def paper_page(paper_id: str, page_number: int) -> Document:
        paper = await document(paper_id)
        pages = paper.get("pages", [])
        if page_number < 1 or page_number > len(pages):
            raise HTTPException(status_code=404, detail="Page not found")
        return {**pages[page_number - 1], "paper_id": paper_id}

    @app.post("/v1/papers/{paper_id:path}/questions")
    async def ask_question(paper_id: str, request: QuestionRequest) -> StreamingResponse:
        paper = await document(paper_id)
        if paper["status"] not in {"INDEXED", "BRIEFED"}:
            raise HTTPException(
                status_code=409, detail="Wait for PDF extraction and indexing to finish"
            )

        async def stream() -> AsyncIterator[str]:
            try:
                async with asyncio.timeout(25.0):
                    ctx = context(uuid.uuid4().hex, 25.0)
                    yield _event(
                        "progress",
                        paper_id=paper_id,
                        status="planning",
                        message="Planning what to read in this paper",
                    )
                    rows = [
                        EvidenceChunk(**row)
                        for row in paper.get("chunks", [])
                        if row.get("paper_id") == paper_id
                    ]
                    history = [
                        DialogueTurn(turn.question, turn.source_ids) for turn in request.history
                    ]
                    if not history and request.context:
                        history = [DialogueTurn(request.context)]
                    llm: LLM | None = cast(LLM, active().instance("llm"))
                    if not getattr(llm, "api_key", None):
                        llm = None
                    run = await answer_question(
                        ctx,
                        paper_id,
                        request.question,
                        rows,
                        llm,
                        project_root=PROJECT_ROOT,
                        history=history,
                    )
                    plan, reading, result = run.plan, run.reading, run.answer
                    excerpts = result.excerpts
                    mode = result.mode
                    yield _event(
                        "progress",
                        paper_id=paper_id,
                        status="verification",
                        message="Source quotations matched; summary evidence review completed"
                        if result.reviewed_claims
                        else "Checking exact quotes and page locations",
                    )
                    yield _event(
                        "answer",
                        paper_id=paper_id,
                        mode=mode,
                        context_question=plan.topic_question,
                        heading="Detailed explanation"
                        if plan.depth == "detailed"
                        else "Brief answer"
                        if plan.depth == "brief"
                        else "Paper summary"
                        if plan.overview
                        else "Answer",
                        evaluation={
                            "source_of_truth": "saved_pdf_text",
                            "task": plan.task,
                            "planner": run.planner_status,
                            "guardrail": run.guardrail.code or "passed",
                            "required_aspects": list(plan.aspects),
                            "covered_aspects": result.covered_aspects,
                            "depth": plan.depth,
                            "follow_up": plan.follow_up,
                            "reading_steps": list(reading.actions),
                            "model_calls": result.model_calls,
                            "reviewed_claims": result.reviewed_claims,
                            "rejected_claims": result.rejected_claims,
                            "accepted_claims": len(excerpts),
                            "answer_relevant": result.answer_relevant,
                            "fallback_reason": result.fallback_reason,
                        },
                    )
                    for excerpt in excerpts:
                        yield _event(
                            "claim",
                            paper_id=paper_id,
                            verified=True,
                            claim={
                                "text": excerpt.text,
                                "label": excerpt.label,
                                "paragraph": excerpt.paragraph,
                                "quote": excerpt.quote,
                                "citation": {
                                    "page_start": excerpt.page_start,
                                    "char_start": excerpt.char_start,
                                    "char_end": excerpt.char_end,
                                    "chunk_id": excerpt.chunk_id,
                                    "label": f"p. {excerpt.page_start}",
                                },
                                "tier_checks": list(excerpt.checks),
                            },
                        )
                    if not excerpts:
                        yield _event(
                            "abstention",
                            paper_id=paper_id,
                            message=(
                                run.guardrail.message
                                or "I could not find verified evidence in this paper "
                                "to answer that question."
                            ),
                        )
                    yield _event(
                        "done",
                        paper_id=paper_id,
                        status="complete",
                        mode=mode,
                        claims_count=len(excerpts),
                        tool_calls=len(reading.actions),
                        model_calls=result.model_calls,
                        input_bytes=result.input_bytes,
                        output_tokens=result.output_tokens,
                    )
            except TimeoutError:
                yield _event(
                    "error",
                    paper_id=paper_id,
                    message="The 25-second answer deadline was reached. Try a narrower question.",
                )
            except Exception:
                logger.exception("Paper answer failed for %s", paper_id)
                yield _event(
                    "error",
                    paper_id=paper_id,
                    message="The answer could not be completed. Please try again.",
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/papers/{paper_id:path}")
    async def get_paper(paper_id: str) -> Document:
        paper = await document(paper_id)
        result = {
            key: value
            for key, value in paper.items()
            if key not in {"file_path", "pages", "chunks"}
        }
        result["page_count"] = len(paper.get("pages", []))
        result["revision"] = paper.get("dedup_hash", "")
        result["pdf_url"] = (
            None if paper.get("pdf_deleted") else f"/v1/papers/{quote(paper_id, safe='')}/pdf"
        )
        return result

    @app.delete("/v1/papers/{paper_id:path}")
    async def delete_paper(paper_id: str, pdf_only: bool = False) -> Document:
        try:
            await active().delete_paper(paper_id, pdf_only)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if monitor is not None:
            monitor.cached = None
        return {"deleted": True, "paper_id": paper_id, "pdf_only": pdf_only}

    @app.post("/v1/sync")
    async def sync_sources(request: SyncRequest) -> JSONResponse:
        try:
            job = await active().start_sync(request.source, request.settings, request.resume_job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({**job, "accepted": True, "single_flight": True}, status_code=202)

    @app.get("/v1/sync")
    async def sync_status() -> Document:
        jobs = [job for job in active().jobs.values() if job.get("kind") != "import"]
        return {
            "job": jobs[-1] if jobs else None,
            "initial_months": active().config.sync.initial_months,
            "settings": active().config.sync.model_dump(exclude_none=True),
            "sources": active().config.plugins.get("sources", []),
        }

    @app.put("/v1/sync/settings")
    async def save_sync_settings(settings: SyncConfig) -> Document:
        try:
            await active().save_sync_settings(settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Could not save arie.yaml") from exc
        return {"settings": settings.model_dump(exclude_none=True)}

    @app.get("/v1/sync/{job_id}")
    async def sync_job(job_id: str) -> Document:
        job = active().jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Sync job not found")
        return job

    @app.post("/v1/sync/{job_id}/pause")
    async def pause_sync(job_id: str) -> Document:
        job = active().jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Sync job not found")
        if job["status"] in {"running", "queued"}:
            await active().update_job(
                job, pause_requested=True, message="Pausing after the current paper finishes"
            )
        return job

    @app.post("/v1/imports")
    async def import_paper(request: ImportRequest) -> JSONResponse:
        try:
            job = await active().start_import(request.url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(job, status_code=202)

    @app.get("/v1/imports/{job_id}")
    async def import_status(job_id: str) -> Document:
        job = active().jobs.get(job_id)
        if job is None or job.get("kind") != "import":
            raise HTTPException(status_code=404, detail="Import job not found")
        return job

    @app.get("/v1/events")
    async def stream_events() -> StreamingResponse:
        return StreamingResponse(active().events(), media_type="text/event-stream")

    @app.get("/v1/skips")
    async def get_skips() -> Document:
        rows = await active().db.fetch(
            context("audit"),
            "SELECT summary_json FROM arie_library "
            "WHERE status IN ('SKIPPED', 'FAILED', 'QUARANTINED') "
            "ORDER BY published_at DESC LIMIT 200",
        )
        papers = [json.loads(str(row["summary_json"])) for row in rows]
        return {
            "items": [
                {**paper, "paper_id": paper["id"], "reason": paper["status_reason"]}
                for paper in papers
            ]
        }

    @app.post("/v1/papers/{paper_id:path}/override")
    async def override_paper(paper_id: str, request: OverrideRequest) -> JSONResponse:
        paper = await document(paper_id)
        del request
        if paper["status"] == "SKIPPED":
            raise HTTPException(
                status_code=409, detail="This paper did not pass ingestion safety gates"
            )
        job = await active().start_sync(paper["source"])
        return JSONResponse({"job_id": job["job_id"], "status": "queued"}, status_code=202)

    @app.get("/v1/plugins")
    async def list_plugins() -> Document:
        config = active().config.plugins
        slots = ["source", "llm", "embedder", "vector_store", "database", "file_storage", "parser"]
        return {
            "plugins": [
                {
                    "slot": slot,
                    "configured": (
                        {"use": "all", "sources": config.get("sources", [])}
                        if slot == "source"
                        else {
                            key: value
                            for key, value in cast(dict[str, object], config.get(slot, {})).items()
                            if key not in {"api_key", "password", "secret", "token"}
                        }
                    ),
                    "available": active().manager.get_available_plugins(slot),
                    **({"providers": active().providers.options()} if slot == "llm" else {}),
                }
                for slot in slots
            ]
        }

    @app.put("/v1/plugins/{slot}")
    async def set_plugin(slot: str, request: PluginRequest) -> Document:
        if slot != "llm":
            raise HTTPException(
                status_code=409, detail="Change this slot in arie.yaml and restart ARIE"
            )
        try:
            return await active().providers.switch(request.plugin, request.config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail="Could not save provider selection to arie.yaml"
            ) from exc

    @app.post("/v1/plugins/{slot}/test")
    async def test_plugin(slot: str, request: PluginRequest) -> Document:
        try:
            if slot == "llm":
                await active().providers.check(request.plugin, request.config)
                return {"ok": True, "slot": slot, "plugin": request.plugin}
            instance = active().manager.build_plugin(slot, request.plugin, **request.config)
            ping = getattr(instance, "ping", None)
            if callable(ping):
                result = ping()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "slot": slot, "plugin": request.plugin}

    @app.get("/v1/storage")
    async def storage_status() -> Document:
        nonlocal monitor
        if monitor is None:
            monitor = StorageMonitor(active())
        return await monitor.snapshot()

    @app.get("/v1/health")
    async def healthcheck() -> Document:
        await active().db.fetch(context("health"), "SELECT 1")
        return {"ok": True, "status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

    @app.post("/mcp/p/{handle}")
    async def mcp_scope_handle(request: Request, handle: str) -> JSONResponse:
        try:
            raw_payload: Any = await request.json()
            payload: dict[str, Any] = cast(
                dict[str, Any],
                raw_payload if isinstance(raw_payload, dict) else {},
            )
            trace_id = request.headers.get("x-trace-id") or "trace-" + uuid.uuid4().hex
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
            if payload.get("method") == "tools/call":
                params_raw: Any = payload.get("params", {})
                params: dict[str, Any] = {}
                if isinstance(params_raw, Mapping):
                    mapping = cast(Mapping[str, Any], params_raw)
                    params = {str(key): value for key, value in mapping.items()}
                scope_value = params.get("scope_handle")
                scope: str | None
                if isinstance(scope_value, str):
                    scope = scope_value
                else:
                    handle_value = params.get("handle")
                    scope = handle_value if isinstance(handle_value, str) else None
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
