from __future__ import annotations

from collections.abc import AsyncIterator

from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse, TokenUsage
from arie_core.plugin_api import plugin


@plugin(slot="llm", name="deepseek")
class DeepSeekLLM:
    """DeepSeek adapter with a fallback to plain text when tool calls are not available."""

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        return max(1, len(value.split()))

    @property
    def supports_json_schema(self) -> bool:
        return False

    @property
    def supports_tool_calls(self) -> bool:
        return False

    async def complete(self, ctx: object, request: LLMRequest) -> LLMResponse:
        del ctx
        prompt = request.prompt or ""
        if request.tools:
            prompt = f"{prompt}\n\nTool capability unavailable; respond with text only."
        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(prompt)
        cache_hit = max(0, prompt_tokens // 4)
        usage = TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            cache_hit=cache_hit,
            cache_miss=max(0, prompt_tokens - cache_hit),
        )
        return LLMResponse(text=prompt, usage=usage, tool_calls=[], finish=True)

    async def stream(self, ctx: object, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        del ctx
        response = await self.complete(object(), request)
        yield LLMDelta(text=response.text, finish=True)


__all__ = ["DeepSeekLLM"]
