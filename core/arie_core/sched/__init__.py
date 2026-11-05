from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Generator
from contextlib import contextmanager
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


class ForegroundGate:
    """Let background work wait while any question is being answered (SPEC section 9).

    A question holds the gate with ``hold()`` for as long as it answers or streams.
    Background submitters (PDF parsing, embedding, guide builds) ``await wait_quiet()``
    before they start new work; work already running is never interrupted. Never wait
    on the gate from inside an answer: it would wait for itself.
    """

    def __init__(self) -> None:
        self.active = 0
        self._quiet = asyncio.Event()
        self._quiet.set()

    @contextmanager
    def hold(self) -> Generator[None]:
        """Mark one question in flight until the block ends, even if it raises."""
        self.active += 1
        self._quiet.clear()
        try:
            yield
        finally:
            self.active -= 1
            if not self.active:
                self._quiet.set()

    async def wait_quiet(self) -> None:
        """Return once no question is in flight; a question that starts first holds us again."""
        while self.active:
            await self._quiet.wait()


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
    "ForegroundGate",
    "supervised",
]
