"""LLM plugins: streams end with a usage delta that carries the cache counts (no network).

Each test serves a canned response through ``httpx.MockTransport``. The wire shapes follow
the providers' own documentation: DeepSeek puts the usage on its last content chunk, OpenAI
sends it on a final chunk with an empty ``choices``, and Gemini repeats ``usageMetadata`` on
its chunks with the full count on the last.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from arie_core.budget import BudgetHandle
from arie_core.contracts import LLMDelta, LLMRequest, TokenUsage
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
REAL_CLIENT = httpx.AsyncClient
REQUEST = LLMRequest(prompt="Explain the paper.", system_prompt="Be brief.")


@dataclass
class Ctx:
    deadline: float = field(default_factory=lambda: time.monotonic() + 30)
    trace_id: str = "trace"
    budget: BudgetHandle = field(default_factory=BudgetHandle)


def llm(name: str, **kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin(
        "llm", name, api_key="test-key", **kwargs
    )


def sse(*events: dict[str, Any] | str) -> bytes:
    return "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n" for event in events
    ).encode()


def serve(
    monkeypatch: pytest.MonkeyPatch, *, body: bytes | None = None, json_body: Any = None
) -> list[httpx.Request]:
    """Answer every request with an SSE body (or a JSON one) and record what was sent."""
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if body is not None:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=json_body)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return REAL_CLIENT(timeout=kwargs.get("timeout"), transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


async def collect(plugin: Any) -> list[LLMDelta]:
    return [delta async for delta in plugin.stream(Ctx(), REQUEST)]


def sent(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def chat_chunk(content: str, **extra: Any) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": content}}], "usage": None, **extra}


# DeepSeek ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepseek_stream_ends_with_cache_hit_and_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "prompt_cache_hit_tokens": 8,
        "prompt_cache_miss_tokens": 2,
    }
    seen = serve(
        monkeypatch,
        body=sse(
            chat_chunk("Hel"),
            chat_chunk("lo"),
            # DeepSeek's usage rides on the last content chunk, which has empty content.
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}], "usage": usage},
            "[DONE]",
        ),
    )
    deltas = await collect(llm("deepseek"))
    assert [d.text for d in deltas] == ["Hel", "lo", ""]
    assert [d.finish for d in deltas] == [False, False, True]
    assert deltas[-1].usage == TokenUsage(prompt=10, completion=3, cache_hit=8, cache_miss=2)
    assert all(d.usage is None for d in deltas[:-1])
    body = sent(seen[0])
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_deepseek_complete_reports_cache_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(
        monkeypatch,
        json_body={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_cache_hit_tokens": 8,
                "prompt_cache_miss_tokens": 2,
            },
        },
    )
    response = await llm("deepseek").complete(Ctx(), REQUEST)
    assert response.usage == TokenUsage(prompt=10, completion=3, cache_hit=8, cache_miss=2)


# OpenAI and OpenAI-compatible -------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_ends_with_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(
        monkeypatch,
        body=sse(
            chat_chunk("Hel"),
            chat_chunk("lo"),
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": None},
            # OpenAI sends the usage on its own chunk with an empty `choices`.
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 6},
                },
            },
            "[DONE]",
        ),
    )
    deltas = await collect(llm("openai"))
    assert [d.text for d in deltas] == ["Hel", "lo", ""]
    assert deltas[-1].finish is True
    assert deltas[-1].usage == TokenUsage(prompt=10, completion=3, cache_hit=6, cache_miss=4)
    body = sent(seen[0])
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_openai_complete_reports_cache_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(
        monkeypatch,
        json_body={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        },
    )
    response = await llm("openai").complete(Ctx(), REQUEST)
    assert response.usage == TokenUsage(prompt=10, completion=3, cache_hit=6, cache_miss=4)


@pytest.mark.asyncio
async def test_openai_compatible_server_without_cache_stats_leaves_them_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(
        monkeypatch,
        body=sse(
            chat_chunk("Hi"),
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
            "[DONE]",
        ),
    )
    deltas = await collect(llm("openai-compatible", base_url="http://llm.test/v1"))
    assert deltas[-1].usage == TokenUsage(prompt=10, completion=3, cache_hit=0, cache_miss=0)


@pytest.mark.asyncio
async def test_openai_stream_with_no_text_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, body=sse({"choices": [], "usage": None}, "[DONE]"))
    with pytest.raises(ValueError, match="no text"):
        await collect(llm("openai"))


# Gemini -----------------------------------------------------------------------------------


def gemini_chunk(text: str, usage: dict[str, int] | None = None, *, thought: bool = False) -> Any:
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    chunk: dict[str, Any] = {"candidates": [{"content": {"parts": [part]}}]}
    if usage is not None:
        chunk["usageMetadata"] = usage
    return chunk


@pytest.mark.asyncio
async def test_gemini_stream_ends_with_cached_content_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(
        monkeypatch,
        body=sse(
            gemini_chunk("Hel", {"promptTokenCount": 10}),
            gemini_chunk("hidden reasoning", thought=True),
            gemini_chunk(
                "lo",
                {
                    "promptTokenCount": 10,
                    "cachedContentTokenCount": 7,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                },
            ),
        ),
    )
    deltas = await collect(llm("gemini"))
    assert [d.text for d in deltas] == ["Hel", "lo", ""]
    assert deltas[-1].finish is True
    # The last usageMetadata is the total; promptTokenCount already includes the cached part.
    assert deltas[-1].usage == TokenUsage(prompt=10, completion=5, cache_hit=7, cache_miss=3)
    assert seen[0].url.path.endswith(":streamGenerateContent")
    assert seen[0].url.params["alt"] == "sse"


@pytest.mark.asyncio
async def test_gemini_without_a_cached_count_means_every_prompt_token_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(
        monkeypatch,
        json_body={
            **gemini_chunk("{}"),
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3},
        },
    )
    response = await llm("gemini").complete(Ctx(), REQUEST)
    assert response.usage == TokenUsage(prompt=10, completion=3, cache_hit=0, cache_miss=10)


# All plugins ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "events"),
    [
        ("deepseek", [chat_chunk("Hi"), "[DONE]"]),
        ("openai", [chat_chunk("Hi"), "[DONE]"]),
        ("gemini", [gemini_chunk("Hi")]),
    ],
)
async def test_a_stream_without_usage_ends_with_none(
    monkeypatch: pytest.MonkeyPatch, name: str, events: list[dict[str, Any] | str]
) -> None:
    serve(monkeypatch, body=sse(*events))
    deltas = await collect(llm(name))
    assert deltas[-1].finish is True
    assert deltas[-1].usage is None
