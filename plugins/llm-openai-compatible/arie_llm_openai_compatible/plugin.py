from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any, cast
from urllib.parse import urlsplit

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


@plugin(slot="llm", name="openai-compatible")
class OpenAICompatibleLLM:
    """Call an OpenAI-compatible endpoint with bounded JSON output; propagate HTTP failures."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = configured_api_key(
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )

    @property
    def supports_json_schema(self) -> bool:
        return True

    @property
    def supports_tool_calls(self) -> bool:
        return False

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        if request.tools:
            raise ValueError("This adapter only supports bounded text requests without tools")
        remaining = ctx.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenAI request deadline has expired")
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        token_field = (
            "max_completion_tokens"
            if urlsplit(self.base_url).hostname == "api.openai.com"
            else "max_tokens"
        )
        async with asyncio.timeout(remaining):
            async with httpx.AsyncClient(
                timeout=min(23.0, remaining), follow_redirects=False
            ) as client:
                response = await client.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        token_field: min(700, request.max_output_tokens),
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                body = cast(dict[str, Any], response.json())
        text = body["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI returned no text")
        usage = body["usage"]
        return LLMResponse(
            text,
            TokenUsage(
                prompt=int(usage["prompt_tokens"]),
                completion=int(usage["completion_tokens"]),
                cache_hit=int(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)),
            ),
        )

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        response = await self.complete(ctx, request)
        yield LLMDelta(text=response.text, finish=True)


@plugin(slot="llm", name="openai")
class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI's built-in provider alias; uses OPENAI_API_KEY and the configured model."""
