from __future__ import annotations

import asyncio
import json
import time
from typing import cast

from arie_core.contracts import LLMRequest
from arie_core.plugin_api import LLM, Ctx


def failure_reason(exc: Exception) -> str:
    """Expose a failure category without provider messages, credentials or response bodies."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"provider_http_{status}" if isinstance(status, int) else type(exc).__name__


class ModelSession:
    """Share hard limits across planning, writing and review; no executable model tools.

    UTF-8 bytes conservatively bound input tokens. Invalid JSON, unexpected tools,
    provider errors and exhausted limits fail closed to the caller's source fallback.
    """

    def __init__(self, ctx: Ctx, llm: LLM | None) -> None:
        self.ctx = ctx
        self.llm = llm
        self.calls = self.input_bytes = self.output_tokens = 0
        self.unavailable_reason = ""

    async def call(self, system: str, payload: object, limit: int, seconds: float) -> object:
        if self.llm is None:
            raise ValueError("Model is unavailable")
        prompt = json.dumps(payload, ensure_ascii=False)
        size = len((system + prompt).encode("utf-8"))
        if self.input_bytes + size > 11300 or self.calls >= 3:
            raise ValueError("Research input or call budget exhausted")
        if limit < 1 or self.output_tokens + limit > 700:
            raise ValueError("Research output budget exhausted")
        remaining = min(seconds, self.ctx.deadline - time.monotonic() - 0.1)
        if remaining <= 0:
            raise TimeoutError("Research deadline exhausted")
        self.ctx.budget.reserve(size)
        self.input_bytes += size
        self.calls += 1
        async with asyncio.timeout(remaining):
            response = await self.llm.complete(
                self.ctx,
                LLMRequest(prompt=prompt, system_prompt=system, tools=[], max_output_tokens=limit),
            )
        self.output_tokens += response.usage.completion
        if response.usage.completion > limit or self.output_tokens > 700:
            raise ValueError("Model exceeded the output budget")
        self.ctx.budget.consume(response.usage.completion)
        if response.tool_calls or len(response.text.encode("utf-8")) > 8 * limit:
            raise ValueError("Unexpected tools or oversized model response")
        return cast(object, json.loads(response.text))
