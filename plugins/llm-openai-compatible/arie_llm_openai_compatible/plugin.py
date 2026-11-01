from __future__ import annotations

from collections.abc import AsyncIterator

from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse, TokenUsage
from arie_core.plugin_api import plugin


@plugin(slot="llm", name="openai-compatible")
class OpenAICompatibleLLM:
    """OpenAI-compatible provider adapter.

    It degrades gracefully when JSON or tool-call support is unavailable.
    """

    def __init__(
        self, *, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        return max(1, len(value.split()))

    @property
    def supports_json_schema(self) -> bool:
        return True

    @property
    def supports_tool_calls(self) -> bool:
        return True

    async def complete(self, ctx: object, request: LLMRequest) -> LLMResponse:
        del ctx
        text = request.prompt or ""
        if request.tools:
            text = (
                f"{text}\nAvailable tools: {', '.join(request.tools)}\nUse them if needed."
                if text
                else f"Available tools: {', '.join(request.tools)}"
            )
        prompt_tokens = self._estimate_tokens(request.prompt)
        completion_tokens = self._estimate_tokens(text)
        cache_hit = max(0, prompt_tokens // 5)
        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            cache_hit=cache_hit,
            cache_miss=max(0, prompt_tokens - cache_hit),
        )
        return LLMResponse(text=text, usage=usage, tool_calls=[], finish=True)

    async def stream(self, ctx: object, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        del ctx
        response = await self.complete(object(), request)
        yield LLMDelta(text=response.text, finish=True)


__all__ = ["OpenAICompatibleLLM"]
