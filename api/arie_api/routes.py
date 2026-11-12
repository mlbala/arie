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
from typing import Annotated, Any, ClassVar, Literal, cast, get_args
from urllib.parse import quote

import msgspec
from arie_core.harness.intent import DialogueTurn
from arie_core.harness.session import EXPLAIN_SECONDS, JUDGE_SECONDS
from arie_core.registry.sync import SyncConfig
from arie_core.sched import EventLoopLagMonitor, supervised
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from arie_api.explain_stream import explain_events
from arie_api.mcp_server import MCPStatelessServer, verify_scope_handle
from arie_api.monitor import StorageMonitor
from arie_api.providers import embedder_options
from arie_api.runtime import Document, ResearchRuntime, context

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# A stored PDF is untrusted bytes (ADR 0013). The UI reads it with pdf.js, which is unaffected;
# opened directly, the browser may not sniff it as another type or run anything in it.
PDF_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox; default-src 'none'",
}


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
    # None takes config ``answer.default_mode``. Explain mode reads the paper's saved chat for
    # its history and ignores ``history`` and ``context``.
    mode: Literal["strict", "explain"] | None = None


# Why an answer was rated down; ``arie_core.harness.feedback.REASONS`` lists the same names.
Reason = Literal[
    "wrong",
    "not_in_paper",
    "too_shallow",
    "too_long",
    "missing_citation",
    "did_not_answer",
]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # null clears the rating.
    rating: Literal["up", "down"] | None
    # The reader's comment on a down rating.
    note: str = Field(default="", max_length=500)
    # Why it was rated down; only with a down rating.
    reasons: tuple[Reason, ...] = Field(default=(), max_length=len(get_args(Reason)))


class FeedbackFactsRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # What a correct answer must contain; each fact must appear in source_sentence if given.
    key_facts: tuple[Annotated[str, Field(max_length=300)], ...] = Field(min_length=1, max_length=3)
    # The paper's own sentence that holds the facts; scores whether the answer cited it.
    source_sentence: str = Field(default="", max_length=600)


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # Write new questions instead of reusing this paper's saved ones.
    regenerate: bool = False
    # Also answer every question without the reranker, to measure what it changes.
    compare_reranker: bool = False
    # Which answer path to score.
    mode: Literal["strict", "explain"] = "strict"


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
            vectors = group.create_task(supervised(service.run_vector_index)())
            guard = group.create_task(supervised(service.run_guard)())
            guides = group.create_task(supervised(service.run_guides)())
            models = group.create_task(supervised(service.models.warm)())
            evaluations = group.create_task(supervised(service.evaluations.serve)())
            lag = EventLoopLagMonitor(
                threshold_ms=service.config.runtime_qos.slow_callback_threshold_ms
            )
            lag_monitor = group.create_task(supervised(lag.run)())
            try:
                yield
            finally:
                for task in (
                    worker, imports, vectors, guard, guides, models, evaluations, lag_monitor
                ):
                    task.cancel()
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

    @app.get("/v1/papers/{paper_id:path}/chat")
    async def get_chat(paper_id: str) -> Document:
        """Saved turns for this paper, oldest first, each with the events it streamed."""
        await document(paper_id)
        return {"paper_id": paper_id, "turns": await active().chat_history(paper_id)}

    @app.delete("/v1/chat")
    async def clear_chats() -> Document:
        """Delete the saved chat of every paper. Papers, guides and ratings are untouched."""
        return {"deleted": await active().clear_chat_history()}

    @app.get("/v1/papers/{paper_id:path}/guide")
    async def get_guide(paper_id: str) -> Document:
        """The paper's cited guide, with a citation per marker, or its build status.

        ``status`` is ``ready`` (``guide`` and ``citations`` set), ``pending`` (queued, or
        failed and waiting for a retry; ``error`` names the last failure) or ``missing``.
        """
        paper = await document(paper_id)
        return await active().guide_view(paper)

    @app.put("/v1/papers/{paper_id:path}/chat/{turn_id}/feedback")
    async def rate_answer(paper_id: str, turn_id: str, request: FeedbackRequest) -> Document:
        """Rate a saved answer; answers rated down join the paper's evaluation for review."""
        await document(paper_id)
        try:
            return await active().evaluations.save_feedback(
                paper_id, turn_id, request.rating, request.note, request.reasons
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/feedback/agreement")
    async def feedback_agreement() -> Document:
        """How often the judge's verdict matched the reader's ratings, overall and per task."""
        return await active().evaluations.agreement()

    @app.put("/v1/papers/{paper_id:path}/chat/{turn_id}/feedback/facts")
    async def add_feedback_facts(
        paper_id: str, turn_id: str, request: FeedbackFactsRequest
    ) -> Document:
        """Attach key facts to an answer rated down, so its question is scored in evaluations."""
        await document(paper_id)
        try:
            return await active().evaluations.save_feedback_facts(
                paper_id, turn_id, request.key_facts, request.source_sentence
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/papers/{paper_id:path}/eval")
    async def paper_evaluation(paper_id: str) -> Document:
        """This paper's evaluation questions, recent runs and answer feedback."""
        await document(paper_id)
        return await active().evaluations.overview(paper_id)

    @app.post("/v1/papers/{paper_id:path}/eval/runs")
    async def start_paper_evaluation(paper_id: str, request: EvalRunRequest) -> JSONResponse:
        """Evaluate answers on this paper, with the paper's own text as the ground truth."""
        await document(paper_id)
        try:
            run = await active().evaluations.start(
                paper_id,
                regenerate=request.regenerate,
                compare_reranker=request.compare_reranker,
                mode=request.mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(run, status_code=202)

    @app.get("/v1/papers/{paper_id:path}/eval/runs/{run_id}")
    async def paper_evaluation_run(paper_id: str, run_id: str) -> Document:
        run = await active().evaluations.get_run(paper_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return run

    @app.get("/v1/papers/{paper_id:path}/pdf")
    async def paper_pdf(paper_id: str) -> FileResponse:
        paper = await document(paper_id)

        def resolve_pdf() -> Path:
            path = Path(str(paper.get("file_path", ""))).resolve()
            if not path.is_relative_to(active().storage_root.resolve()) or not path.is_file():
                raise HTTPException(status_code=404, detail="Extracted PDF is unavailable")
            return path

        path = await asyncio.to_thread(resolve_pdf)
        return FileResponse(path, media_type="application/pdf", headers=PDF_HEADERS)

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
        if paper["status"] != "INDEXED":
            raise HTTPException(
                status_code=409, detail="Wait for PDF extraction and indexing to finish"
            )

        async def stream() -> AsyncIterator[str]:
            # The answer as streamed, saved once complete so a reload shows the same turn.
            saved: list[Document] = []
            turn_id = uuid.uuid4().hex

            def emit(kind: str, **payload: Any) -> str:
                saved.append({"event": kind, "data": payload})
                return _event(kind, **payload)

            try:
                async with asyncio.timeout(25.0):
                    ctx = context(uuid.uuid4().hex, 25.0)
                    yield _event(
                        "progress",
                        paper_id=paper_id,
                        status="planning",
                        message="Planning what to read in this paper",
                    )
                    history = [
                        DialogueTurn(turn.question, turn.source_ids) for turn in request.history
                    ]
                    if not history and request.context:
                        history = [DialogueTurn(request.context)]
                    run = await active().answer_paper_question(
                        ctx, paper, request.question, history
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
                    yield emit(
                        "answer",
                        paper_id=paper_id,
                        mode=mode,
                        trace_id=ctx.trace_id,
                        context_question=plan.topic_question,
                        heading="Paper summary"
                        if plan.overview and not plan.follow_up
                        else "Detailed explanation"
                        if plan.depth == "detailed"
                        else "Brief answer"
                        if plan.depth == "brief"
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
                            "reranked": reading.reranked,
                            "model_calls": result.model_calls,
                            "reviewed_claims": result.reviewed_claims,
                            "rejected_claims": result.rejected_claims,
                            "accepted_claims": len(excerpts),
                            "answer_relevant": result.answer_relevant,
                            "fallback_reason": result.fallback_reason,
                            "trace": msgspec.to_builtins(result.trace),
                        },
                    )
                    for excerpt in excerpts:
                        yield emit(
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
                        yield emit(
                            "abstention",
                            paper_id=paper_id,
                            message=(
                                run.guardrail.message
                                or "I could not find verified evidence in this paper "
                                "to answer that question."
                            ),
                        )
                    yield emit(
                        "done",
                        paper_id=paper_id,
                        turn_id=turn_id,
                        status="complete",
                        mode=mode,
                        claims_count=len(excerpts),
                        tool_calls=len(reading.actions),
                        model_calls=result.model_calls,
                        input_bytes=result.input_bytes,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                    try:
                        await active().save_chat_turn(paper_id, request.question, saved, turn_id)
                    except Exception:
                        # History is a convenience; the answer was already delivered.
                        logger.warning("Could not save chat turn for %s", paper_id, exc_info=True)
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

        async def explain_stream() -> AsyncIterator[str]:
            # The events as streamed (the text saved as one delta), kept for a reload.
            saved: list[Document] = []
            turn_id = uuid.uuid4().hex
            try:
                async with asyncio.timeout(EXPLAIN_SECONDS + JUDGE_SECONDS):
                    async for kind, payload in explain_events(
                        active(), paper, request.question, turn_id, saved
                    ):
                        yield _event(kind, **payload)
                    try:
                        await active().save_chat_turn(paper_id, request.question, saved, turn_id)
                    except Exception:
                        # History is a convenience; the answer was already delivered.
                        logger.warning("Could not save chat turn for %s", paper_id, exc_info=True)
            except TimeoutError:
                yield _event(
                    "error",
                    paper_id=paper_id,
                    message="The explanation deadline was reached. Try a narrower question.",
                )
            except Exception:
                logger.exception("Paper explanation failed for %s", paper_id)
                yield _event(
                    "error",
                    paper_id=paper_id,
                    message="The answer could not be completed. Please try again.",
                )

        mode = request.mode or active().config.answer.default_mode
        return StreamingResponse(
            explain_stream() if mode == "explain" else stream(),
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
        slots = [
            "source",
            "llm",
            "embedder",
            "vector_store",
            "database",
            "file_storage",
            "parser",
            "entailment",
            "reranker",
            "guard",
        ]
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
                    **(
                        {
                            "providers": embedder_options(
                                active().config, active().manager.get_available_plugins(slot)
                            ),
                            "effective": {
                                "embedder": active().embedder_id,
                                "reason": active().embedder_reason,
                            },
                        }
                        if slot == "embedder"
                        else {}
                    ),
                }
                for slot in slots
            ]
        }

    @app.put("/v1/plugins/{slot}")
    async def set_plugin(slot: str, request: PluginRequest) -> Document:
        if slot not in {"llm", "embedder"}:
            raise HTTPException(
                status_code=409, detail="Change this slot in arie.yaml and restart ARIE"
            )
        try:
            if slot == "embedder":
                return await active().switch_embedder(request.plugin)
            result = await active().providers.switch(request.plugin, request.config)
            # Automatic embedding follows the newly selected model provider.
            active().apply_embedder()
            return {**result, "embedder": active().embedder_id}
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
            if slot == "embedder":
                await active().check_embedder(request.plugin)
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
