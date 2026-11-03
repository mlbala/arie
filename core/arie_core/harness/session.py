from __future__ import annotations

import asyncio
import json
import time
from typing import cast

from arie_core.contracts import LLMRequest
from arie_core.plugin_api import LLM, Ctx

# I3: at most 12,000 input tokens per answer. A call is admitted on an estimate of 3 UTF-8
# bytes per token (paper text measured ~4.2 on DeepSeek); the running total is then
# corrected with the provider's reported prompt tokens, and exceeding the cap fails closed.
INPUT_BUDGET_TOKENS = 12_000
BYTES_PER_TOKEN = 3


def estimate_tokens(size_bytes: int) -> int:
    return -(-size_bytes // BYTES_PER_TOKEN)


def failure_reason(exc: Exception) -> str:
    """Expose a failure category without provider messages, credentials or response bodies."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"provider_http_{status}" if isinstance(status, int) else type(exc).__name__


class ModelSession:
    """Share hard limits across planning, writing and review; no executable model tools.

    Input tokens are estimated before and provider-counted after each call. Invalid JSON,
    unexpected tools,
    provider errors and exhausted limits fail closed to the caller's source fallback.
    """

    def __init__(self, ctx: Ctx, llm: LLM | None) -> None:
        self.ctx = ctx
        self.llm = llm
        self.calls = self.input_bytes = self.input_tokens = self.output_tokens = 0
        self.unavailable_reason = ""

    def remaining_input(self, system: str, payload: object) -> int:
        """Bytes still admissible after sending ``system`` and ``payload`` (negative if over)."""
        size = len((system + json.dumps(payload, ensure_ascii=False)).encode("utf-8"))
        return (INPUT_BUDGET_TOKENS - self.input_tokens) * BYTES_PER_TOKEN - size

    async def call(self, system: str, payload: object, limit: int, seconds: float) -> object:
        if self.llm is None:
            raise ValueError("Model is unavailable")
        prompt = json.dumps(payload, ensure_ascii=False)
        size = len((system + prompt).encode("utf-8"))
        estimate = estimate_tokens(size)
        if self.input_tokens + estimate > INPUT_BUDGET_TOKENS or self.calls >= 3:
            raise ValueError("Research input or call budget exhausted")
        if limit < 1 or self.output_tokens + limit > 700:
            raise ValueError("Research output budget exhausted")
        remaining = min(seconds, self.ctx.deadline - time.monotonic() - 0.1)
        if remaining <= 0:
            raise TimeoutError("Research deadline exhausted")
        self.ctx.budget.reserve(estimate)
        self.input_bytes += size
        self.input_tokens += estimate
        self.calls += 1
        async with asyncio.timeout(remaining):
            response = await self.llm.complete(
                self.ctx,
                LLMRequest(prompt=prompt, system_prompt=system, tools=[], max_output_tokens=limit),
            )
        undercount = max(0, response.usage.prompt - estimate)
        if response.usage.prompt > 0:
            self.input_tokens += response.usage.prompt - estimate
        if self.input_tokens > INPUT_BUDGET_TOKENS:
            raise ValueError("Provider-counted input exceeded the 12,000-token budget")
        # The request budget has no refunds; charge it only for an undercount.
        self.ctx.budget.reserve(undercount)
        self.output_tokens += response.usage.completion
        if response.usage.completion > limit or self.output_tokens > 700:
            raise ValueError("Model exceeded the output budget")
        self.ctx.budget.consume(response.usage.completion)
        if response.tool_calls or len(response.text.encode("utf-8")) > 8 * limit:
            raise ValueError("Unexpected tools or oversized model response")
        return cast(object, json.loads(response.text))
