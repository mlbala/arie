from __future__ import annotations

import asyncio
import os
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


@plugin(slot="llm", name="deepseek")
class DeepSeekLLM:
    """Call the configured provider under the request deadline; never echo a prompt as an answer."""

    def __init__(
        self,
        *,
        model: str = "deepseek-flash",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.api_key = configured_api_key(
            api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        )

    @property
    def supports_json_schema(self) -> bool:
        return False

    @property
    def supports_tool_calls(self) -> bool:
        return False

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        async with asyncio.timeout(max(0.01, ctx.deadline - time.monotonic())):
            async with httpx.AsyncClient(timeout=23.0) as client:
                response = await client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": min(700, request.max_output_tokens),
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                        # Hidden reasoning would spend the bounded output budget (I3).
                        "thinking": {"type": "disabled"},
                    },
                )
                response.raise_for_status()
                body = cast(dict[str, Any], response.json())
        usage = body.get("usage", {})
        return LLMResponse(
            text=str(body["choices"][0]["message"]["content"]),
            usage=TokenUsage(
                prompt=int(usage.get("prompt_tokens", 0)),
                completion=int(usage.get("completion_tokens", 0)),
                cache_hit=int(usage.get("prompt_cache_hit_tokens", 0)),
            ),
        )

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        response = await self.complete(ctx, request)
        yield LLMDelta(text=response.text, finish=True)
