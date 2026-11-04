from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import cast

import msgspec

from arie_core.contracts import LLMDelta, LLMRequest, TokenUsage
from arie_core.harness.trace import STAGES, AnswerTrace, StageSeconds
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


def model_name(llm: LLM | None) -> str:
    """The provider model the plugin was configured with, or "" when it does not say."""
    name: object = getattr(llm, "model", "")
    return name if isinstance(name, str) else ""


class StreamBudget(msgspec.Struct, frozen=True):
    """Seconds a streaming call may take: to its first text, between texts, and in total."""

    first_token: float = 10.0
    stall: float = 15.0
    total: float = 90.0


# Explain mode's writer call (ADR 0010).
EXPLAIN_BUDGET = StreamBudget()
# An explain request's deadline: the streaming budget plus a few seconds for the guard, the
# pack and the paged-pack choice. Strict mode's request deadline is 25 s.
EXPLAIN_SECONDS = EXPLAIN_BUDGET.total + 5.0

# The judge's streamed calls (S13): a verdict list for one answer, or one pairwise verdict. A
# judge call is its own request with its own budget, so a slow judge never eats the writer's.
JUDGE_BUDGET = StreamBudget(first_token=15.0, stall=15.0, total=45.0)
JUDGE_SECONDS = JUDGE_BUDGET.total + 5.0


class StreamStopped(Exception):
    """A streaming call stopped by its budget; ``reason`` is the fallback reason to record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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
        self.cache_hit = self.cache_miss = 0
        # The usage the provider reported on a streaming call's final delta, if any.
        self.stream_usage: TokenUsage | None = None
        self.unavailable_reason = ""
        # For the answer trace: the configured model (kept even if the LLM is later
        # switched off), the prompt file version in use, and seconds per pipeline stage.
        self.model = model_name(llm)
        self.prompt_version = ""
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        """Add the time spent inside the block to ``name``, even if the block raises."""
        started = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.monotonic() - started

    def trace(self, mode: str, fallback_reason: str, total: float) -> AnswerTrace:
        """The trace so far; ``total`` is never less than the sum of the stages that ran."""
        seconds = {name: round(self.stages.get(name, 0.0), 3) for name in STAGES}
        seconds["total"] = round(max(total, sum(seconds.values())), 3)
        return AnswerTrace(
            seconds=StageSeconds(**seconds),
            tokens=TokenUsage(
                self.input_tokens, self.output_tokens, self.cache_hit, self.cache_miss
            ),
            prompt_version=self.prompt_version,
            model=self.model,
            mode=mode,
            fallback_reason=fallback_reason,
        )

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
        self.cache_hit += response.usage.cache_hit
        self.cache_miss += response.usage.cache_miss
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

    async def stream(
        self, request: LLMRequest, *, input_limit: int, budget: StreamBudget
    ) -> AsyncGenerator[str]:
        """Yield the text of one streaming call within ``budget``; explain mode's writer path.

        Fails closed with ``StreamStopped`` before any model call when the session already
        streamed, no model is set, the estimated input exceeds ``input_limit`` tokens, or the
        request budget refuses it. While streaming, the first text must arrive within
        ``first_token`` seconds, each next one within ``stall`` seconds of asking for it, and
        all of it within ``total`` (capped by the request deadline). Deltas without text are
        not progress. Tokens are counted from the final delta's usage; without one, input
        stays at its estimate and output is estimated from the text.
        """
        if self.calls:
            raise StreamStopped("call_budget")
        if self.llm is None:
            raise StreamStopped("model_unavailable")
        size = len(((request.system_prompt or "") + request.prompt).encode("utf-8"))
        estimate = estimate_tokens(size)
        if estimate > input_limit or request.max_output_tokens < 1:
            raise StreamStopped("input_budget")
        try:
            self.ctx.budget.reserve(estimate)
        except (RuntimeError, ValueError):
            raise StreamStopped("request_budget") from None
        self.input_bytes += size
        self.input_tokens += estimate
        self.calls += 1

        started = time.monotonic()
        end = min(started + budget.total, self.ctx.deadline - 0.1)
        waiting_since = started
        streamed = 0
        usage: TokenUsage | None = None
        deltas = aiter(self.llm.stream(self.ctx, request))
        try:
            while True:
                if streamed:
                    due, reason = waiting_since + budget.stall, "stall_timeout"
                else:
                    due, reason = waiting_since + budget.first_token, "first_token_timeout"
                if end <= due:
                    due, reason = end, "total_timeout"
                wait = due - time.monotonic()
                if wait <= 0:
                    raise StreamStopped(reason)
                try:
                    async with asyncio.timeout(wait):
                        delta: LLMDelta = await anext(deltas)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise StreamStopped(reason) from None
                if delta.usage is not None:
                    usage = delta.usage
                if delta.tool_calls:
                    raise StreamStopped("unexpected_tools")
                if delta.text:
                    streamed += len(delta.text.encode("utf-8"))
                    if streamed > 8 * request.max_output_tokens:
                        raise StreamStopped("output_budget")
                    yield delta.text
                    # The stall clock measures the provider, not the caller's time between pulls.
                    waiting_since = time.monotonic()
        finally:
            if isinstance(deltas, AsyncGenerator):
                await cast(AsyncGenerator[LLMDelta], deltas).aclose()
            # Counted on every exit, so a stopped stream's trace still shows what it used.
            self.stream_usage = usage
            if usage is None:
                usage = TokenUsage(0, estimate_tokens(streamed))
            self.cache_hit += usage.cache_hit
            self.cache_miss += usage.cache_miss
            if usage.prompt > 0:
                self.input_tokens += usage.prompt - estimate
            self.output_tokens += usage.completion
        try:
            # The request budget has no refunds; charge it only for an undercount.
            self.ctx.budget.reserve(max(0, usage.prompt - estimate))
            self.ctx.budget.consume(usage.completion)
        except (RuntimeError, ValueError):
            raise StreamStopped("request_budget") from None
        if usage.completion > request.max_output_tokens:
            raise StreamStopped("output_budget")
