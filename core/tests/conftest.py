"""Test doubles shared by more than one core test file."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse, TokenUsage
from arie_core.plugin_api import Ctx

# One step of a scripted stream: a text delta, or seconds to wait before the next step.
StreamStep = str | float
SCRIPTED_USAGE = TokenUsage(1, 30)


class ScriptedModel:
    """An LLM that answers ``complete`` with scripted JSON and ``stream`` with scripted steps.

    Each ``stream`` call plays the next script in ``streams``, then sends a final delta with
    ``usage`` (``None`` plays a provider that reports no usage).
    """

    def __init__(
        self,
        responses: list[object] | None = None,
        *,
        streams: Sequence[Sequence[StreamStep]] = (),
        usage: TokenUsage | None = SCRIPTED_USAGE,
    ) -> None:
        self.responses = list(responses or [])
        self.streams = [list(script) for script in streams]
        self.usage = usage
        self.requests: list[LLMRequest] = []

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        del ctx
        self.requests.append(request)
        assert not request.tools
        return LLMResponse(json.dumps(self.responses.pop(0)), SCRIPTED_USAGE)

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        del ctx
        self.requests.append(request)
        assert not request.tools
        for step in self.streams.pop(0):
            if isinstance(step, str):
                yield LLMDelta(step)
            else:
                await asyncio.sleep(step)
        yield LLMDelta(finish=True, usage=self.usage)
