"""The events of an explain-mode answer, as ``(kind, payload)`` pairs for the question route.

Strict mode's events are built in the route. Explain mode's live here: the text arrives as
``delta`` events, then ``answer``, ``citations``, ``checks``, ``judge`` and ``done`` describe it.
A reload shows the same turn from the saved events, with the text saved as one ``delta``.
SPEC 8.9 and 11.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing

import msgspec
from arie_core.harness.explain import ExplainResult

from arie_api.runtime import Document, ResearchRuntime

# What the reader is told when an explanation produced no text, by reason.
_FAILURES = {
    "model_unavailable": (
        "Explain mode needs a language model and none is available. "
        "Add an API key in Settings, then ask again."
    ),
    "first_token_timeout": "The model did not start answering in time. Please try again.",
    "stall_timeout": "The model stopped answering. Please try again.",
    "total_timeout": "The explanation took too long. Try a narrower question.",
    "input_budget": "This paper does not fit the model's context window.",
}
_FAILED = "The explanation could not be completed. Please try again."
# done.status for each result mode; a refusal by the guard is a complete answer.
_STATUS = {
    "explain": "complete",
    "explain_partial": "partial",
    "explain_failed": "failed",
    "abstention": "complete",
}


async def explain_events(
    runtime: ResearchRuntime,
    paper: Document,
    question: str,
    turn_id: str,
    saved: list[Document],
) -> AsyncIterator[tuple[str, Document]]:
    """Stream one explanation and append the events a reload needs to ``saved``.

    Order: ``progress``, ``delta`` (one per piece of text), ``answer``, ``citations`` and
    ``checks`` (only if text arrived), ``progress`` and ``judge`` (only if ``judge.chat`` is
    ``all`` and the answer cites something), ``abstention`` (only if no text arrived) and
    ``done``. ``answer`` and the events after it come last because the trace is complete only
    when the stream has ended; ``checks`` is the sentence-by-sentence verdict of
    ``check_answer`` and ``judge`` the model's verdict on each cited sentence (SPEC 8.9).
    """
    paper_id = str(paper["id"])

    def emit(kind: str, **payload: object) -> tuple[str, Document]:
        saved.append({"event": kind, "data": payload})
        return kind, payload

    yield (
        "progress",
        {
            "paper_id": paper_id,
            "status": "reading",
            "message": "Reading the whole paper",
        },
    )
    turns = await runtime.explain_turns(paper_id)
    result: ExplainResult | None = None
    trace_id = f"explain:{uuid.uuid4().hex}"
    stream = runtime.explain_paper_question(trace_id, paper, question, turns)
    async with aclosing(stream) as items:
        async for item in items:
            if isinstance(item, str):
                yield "delta", {"paper_id": paper_id, "text": item}
            else:
                result = item
    if result is None:
        raise RuntimeError("The explain stream ended without a result")

    text = result.text
    if text:
        # One saved delta replays the same text as the streamed pieces.
        saved.append({"event": "delta", "data": {"paper_id": paper_id, "text": text}})
    mode = "explained_answer" if text else "abstention"
    trace = msgspec.to_builtins(result.trace)
    yield emit(
        "answer",
        paper_id=paper_id,
        mode=mode,
        heading="Explanation",
        prompt_version=result.trace.prompt_version,
        trace_id=trace_id,
        evaluation={
            "source_of_truth": "saved_pdf_text",
            "task": "explain",
            "depth": result.depth,
            "paged": result.paged,
            "guardrail": result.fallback_reason if result.mode == "abstention" else "passed",
            "complete": result.mode == "explain",
            "fallback_reason": result.fallback_reason,
            "unknown_markers": list(result.unknown_markers),
            "first_token_seconds": result.first_token_seconds,
            "trace": trace,
        },
    )

    citations: list[Document] = []
    if text:
        # A citation is checked when a sentence cites it and no sentence that does is flagged.
        flagged = result.checks.flagged_markers()
        cited_markers = {m for sentence in result.checks.sentences for m in sentence.markers}
        chunks = {
            str(row["chunk_id"]): row
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        }
        for cited in result.citations:
            row = chunks.get(cited.chunk_id)
            if row is not None:
                citations.append(
                    {
                        "marker": cited.marker,
                        "chunk_id": cited.chunk_id,
                        "page": row["page_start"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        # The span is the whole chunk: an explain answer cites chunks, not quotes.
                        "checked": cited.marker in cited_markers and cited.marker not in flagged,
                    }
                )
        yield emit("citations", paper_id=paper_id, citations=citations)
        yield emit("checks", paper_id=paper_id, **msgspec.to_builtins(result.checks))
        if runtime.config.judge.chat == "all" and result.checks.cited:
            yield (
                "progress",
                {"paper_id": paper_id, "status": "judging", "message": "Checking the citations"},
            )
            judgement = await runtime.judge_explanation(trace_id, result)
            yield emit("judge", paper_id=paper_id, **msgspec.to_builtins(judgement))
    else:
        yield emit(
            "abstention",
            paper_id=paper_id,
            message=result.message or _FAILURES.get(result.fallback_reason, _FAILED),
        )

    usage = {
        "prompt": result.trace.tokens.prompt,
        "completion": result.trace.tokens.completion,
        "cache_hit": result.trace.tokens.cache_hit,
        "cache_miss": result.trace.tokens.cache_miss,
    }
    yield emit(
        "done",
        paper_id=paper_id,
        turn_id=turn_id,
        status=_STATUS[result.mode],
        mode=mode,
        claims_count=len(citations),
        tool_calls=0,
        input_tokens=usage["prompt"],
        output_tokens=usage["completion"],
        usage=usage,
        first_token_seconds=result.first_token_seconds,
    )
