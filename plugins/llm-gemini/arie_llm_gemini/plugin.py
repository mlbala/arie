from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from arie_core.plugin_api import (
    Ctx,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    configured_api_key,
    plugin,
)


def _visible_text(body: dict[str, Any]) -> str:
    """The answer text of one response or stream chunk, without thought parts."""
    candidates = cast(list[dict[str, Any]], body.get("candidates") or [])
    parts = cast(
        list[dict[str, Any]],
        candidates[0].get("content", {}).get("parts", []) if candidates else [],
    )
    return "".join(
        str(part["text"])
        for part in parts
        if isinstance(part.get("text"), str) and not part.get("thought")
    )


def _token_usage(usage: dict[str, Any]) -> TokenUsage:
    # promptTokenCount already includes the cached tokens, and cachedContentTokenCount is
    # left out when nothing was cached, so the misses are the difference.
    prompt = int(usage.get("promptTokenCount", 0))
    hit = int(usage.get("cachedContentTokenCount", 0))
    return TokenUsage(
        prompt=prompt,
        completion=int(usage.get("candidatesTokenCount", 0))
        + int(usage.get("thoughtsTokenCount", 0)),
        cache_hit=hit,
        cache_miss=max(0, prompt - hit),
    )


@plugin(slot="llm", name="gemini")
class GeminiLLM:
    """Call Gemini's native JSON endpoint; honor deadlines and account for thinking tokens."""

    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash-lite",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
            raise ValueError("Gemini model must be a model name, without paths or query parameters")
        self.model = model
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.api_key = configured_api_key(
            api_key
            if api_key is not None
            else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        )

    def _payload(self, request: LLMRequest) -> dict[str, object]:
        if request.tools:
            raise ValueError("This adapter only supports bounded text requests without tools")
        if self.model.startswith("gemini-2.5-"):
            thinking: dict[str, object] = {"thinkingBudget": 128 if "pro" in self.model else 0}
        else:
            thinking = {"thinkingLevel": "minimal"}
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
                "maxOutputTokens": min(700, request.max_output_tokens),
                "thinkingConfig": thinking,
            },
        }
        if request.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return payload

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        payload = self._payload(request)
        remaining = ctx.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Gemini request deadline has expired")
        async with asyncio.timeout(remaining):
            async with httpx.AsyncClient(
                timeout=min(23.0, remaining), follow_redirects=False
            ) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                )
                response.raise_for_status()
                body = cast(dict[str, Any], response.json())
        text = _visible_text(body)
        if not text.strip():
            raise ValueError("Gemini returned no text; check model access or output limits")
        return LLMResponse(text, _token_usage(body["usageMetadata"]))

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        """Yield text deltas, then one final delta with the usage (None if none was sent)."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        payload = self._payload(request)
        remaining = ctx.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Gemini request deadline has expired")
        usage: TokenUsage | None = None
        emitted = False
        async with asyncio.timeout(remaining):
            async with (
                httpx.AsyncClient(timeout=min(23.0, remaining), follow_redirects=False) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse",
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = cast(dict[str, Any], json.loads(line[5:].strip()))
                    # Early chunks can carry a partial count; the last one is the total.
                    if isinstance(chunk.get("usageMetadata"), dict):
                        usage = _token_usage(chunk["usageMetadata"])
                    text = _visible_text(chunk)
                    if text:
                        emitted = True
                        yield LLMDelta(text=text)
        if not emitted:
            raise ValueError("Gemini returned no text; check model access or output limits")
        yield LLMDelta(finish=True, usage=usage)
