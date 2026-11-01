from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable

Listener = Callable[[dict[str, object]], object]


class EventBus:
    """Tiny in-process event bus for core orchestration and audit hooks."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def subscribe(self, event: str, callback: Listener) -> Callable[[], None]:
        self._listeners[event].append(callback)

        def _unsubscribe() -> None:
            listeners = self._listeners.get(event, [])
            if callback in listeners:
                listeners.remove(callback)

        return _unsubscribe

    def publish(self, event: str, payload: dict[str, object] | None = None) -> list[object]:
        results: list[object] = []
        for callback in list(self._listeners.get(event, [])):
            result = callback(payload or {})
            if asyncio.iscoroutine(result):
                raise TypeError("async listeners require emit_async()")
            results.append(result)
        return results

    async def emit_async(
        self,
        event: str,
        payload: dict[str, object] | None = None,
    ) -> list[object]:
        results: list[object] = []
        for callback in list(self._listeners.get(event, [])):
            result = callback(payload or {})
            if asyncio.iscoroutine(result):
                result = await result
            results.append(result)
        return results

    async def publish_async(
        self,
        event: str,
        payload: dict[str, object] | None = None,
    ) -> list[object]:
        return await self.emit_async(event, payload)


__all__ = ["EventBus"]
