from __future__ import annotations

import asyncio
import json
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


def _token_usage(usage: dict[str, Any]) -> TokenUsage:
    # DeepSeek reports prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens.
    return TokenUsage(
        prompt=int(usage.get("prompt_tokens", 0)),
        completion=int(usage.get("completion_tokens", 0)),
        cache_hit=int(usage.get("prompt_cache_hit_tokens", 0)),
        cache_miss=int(usage.get("prompt_cache_miss_tokens", 0)),
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

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        messages = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(700, request.max_output_tokens),
            "response_format": {"type": "json_object"},
            "temperature": 0,
            # Hidden reasoning would spend the bounded output budget (I3).
            "thinking": {"type": "disabled"},
        }
        if stream:
            # A stream is prose (explain mode's Markdown) with the caller's own output budget.
            # JSON mode would also be refused: DeepSeek wants the word "json" in the prompt.
            del payload["response_format"]
            payload["max_tokens"] = request.max_output_tokens
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        async with asyncio.timeout(max(0.01, ctx.deadline - time.monotonic())):
            async with httpx.AsyncClient(timeout=23.0) as client:
                response = await client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._payload(request, stream=False),
                )
                response.raise_for_status()
                body = cast(dict[str, Any], response.json())
        return LLMResponse(
            text=str(body["choices"][0]["message"]["content"]),
            usage=_token_usage(body.get("usage", {})),
        )

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        """Yield text deltas, then one final delta with the usage (None if none was sent)."""
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        usage: TokenUsage | None = None
        async with asyncio.timeout(max(0.01, ctx.deadline - time.monotonic())):
            async with (
                httpx.AsyncClient(timeout=23.0) as client,
                client.stream(
                    "POST",
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._payload(request, stream=True),
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = cast(dict[str, Any], json.loads(data))
                    # DeepSeek puts the usage on the last content chunk; every earlier
                    # chunk has `"usage": null`.
                    if isinstance(chunk.get("usage"), dict):
                        usage = _token_usage(chunk["usage"])
                    for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
                        delta: dict[str, Any] = choice.get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            yield LLMDelta(text=text)
        yield LLMDelta(finish=True, usage=usage)
