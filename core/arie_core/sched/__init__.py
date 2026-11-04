from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import TypeVar

T = TypeVar("T")


def supervised[T](
    func: Callable[..., Awaitable[T]],
    *args: object,
    **kwargs: object,
) -> Callable[[], Coroutine[object, object, T]]:
    """Wrap a task so TaskGroup failures remain easy to supervise and debug."""

    async def _runner() -> T:
        return await func(*args, **kwargs)

    return _runner


class EventLoopLagMonitor:
    """Warn when event-loop callbacks run later than ``threshold_ms``.

    ``run`` loops until cancelled; start it inside the caller's ``asyncio.TaskGroup``
    with ``supervised()`` so its lifetime is owned by structured concurrency.
    """

    def __init__(
        self,
        *,
        threshold_ms: float = 15.0,
        interval_ms: float = 100.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.threshold_ms = threshold_ms
        self.interval_ms = interval_ms
        self._logger = logger or logging.getLogger("arie_core.sched")

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            before = loop.time()
            await asyncio.sleep(self.interval_ms / 1000.0)
            after = loop.time()
            lag_ms = (after - before - self.interval_ms / 1000.0) * 1000.0
            if lag_ms > self.threshold_ms:
                self._logger.warning(
                    "Event loop lag %.2fms exceeded threshold %.2fms",
                    lag_ms,
                    self.threshold_ms,
                )


__all__ = [
    "EventLoopLagMonitor",
    "supervised",
]
