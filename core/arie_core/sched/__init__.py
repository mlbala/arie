from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from typing import TypeVar

QoS_UserInteractive = "user_interactive"
QoS_UserInitiated = "user_initiated"
QoS_Utility = "utility"
QoS_Background = "background"

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


class TokenBucket:
    """Simple token-bucket rate limiter with a steady refill rate."""

    def __init__(self, rate_per_second: float, *, capacity: float = 1.0) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self.rate_per_second = rate_per_second
        self.capacity = capacity
        self.tokens = capacity
        self.updated_at = time.monotonic()

    def acquire(self, *, amount: float = 1.0) -> float:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
        self.updated_at = now
        if self.tokens >= amount:
            self.tokens -= amount
            return 0.0
        deficit = (amount - self.tokens) / self.rate_per_second
        self.tokens = 0.0
        self.updated_at = now + deficit
        return deficit


class HostRateLimiter:
    """Per-host limiter with a default arXiv policy of 1 request / 3s."""

    def __init__(self, *, limits: dict[str, float] | None = None) -> None:
        self._limits = {"export.arxiv.org": 1.0 / 3.0}
        if limits is not None:
            self._limits.update(limits)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, host: str) -> None:
        if host not in self._buckets:
            rate = self._limits.get(host, 1.0)
            self._buckets[host] = TokenBucket(rate, capacity=1.0)
        bucket = self._buckets[host]
        while True:
            async with self._lock:
                wait_for = bucket.acquire(amount=1.0)
            if wait_for <= 0:
                return
            await asyncio.sleep(wait_for)


class QoSPriorityGate:
    """Prioritize interactive and user-initiated work ahead of utility and background work."""

    def __init__(self, *, max_llm_concurrency: int = 4) -> None:
        self.max_llm_concurrency = max_llm_concurrency
        self._overall = asyncio.Semaphore(max_llm_concurrency)
        self._lane_semaphores: dict[str, asyncio.Semaphore] = {
            QoS_UserInteractive: asyncio.Semaphore(1),
            QoS_UserInitiated: asyncio.Semaphore(2),
            QoS_Utility: asyncio.Semaphore(max_llm_concurrency),
            QoS_Background: asyncio.Semaphore(max_llm_concurrency),
        }
        self._in_use: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self._waiters: list[tuple[int, int, str]] = []
        self._counter = 0

    @staticmethod
    def priority_for(lane: str) -> int:
        return {
            QoS_UserInteractive: 0,
            QoS_UserInitiated: 1,
            QoS_Utility: 2,
            QoS_Background: 3,
        }.get(lane, 99)

    async def acquire(self, lane: str) -> None:
        priority = self.priority_for(lane)
        async with self._lock:
            self._counter += 1
            self._waiters.append((priority, self._counter, lane))
            self._waiters.sort(key=lambda item: (item[0], item[1]))

        try:
            while True:
                async with self._lock:
                    if not self._waiters:
                        break
                    if self._waiters[0][2] != lane:
                        await asyncio.sleep(0.01)
                        continue
                    if lane in {QoS_UserInteractive, QoS_UserInitiated}:
                        # User work should be granted before lower-priority work.
                        self._waiters.pop(0)
                        self._in_use[lane] += 1
                        break
                    if self._in_use[QoS_Background] < self.max_llm_concurrency:
                        self._waiters.pop(0)
                        self._in_use[lane] += 1
                        break
                await asyncio.sleep(0.01)
            await self._overall.acquire()
            await self._lane_semaphores[lane].acquire()
        except Exception:
            async with self._lock:
                self._in_use[lane] = max(0, self._in_use.get(lane, 0) - 1)
            raise

    def release(self, lane: str) -> None:
        self._lane_semaphores[lane].release()
        self._overall.release()
        self._in_use[lane] = max(0, self._in_use.get(lane, 0) - 1)


class EventLoopLagMonitor:
    """Checks callback lag and warns when it exceeds the configured threshold."""

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
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
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

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


__all__ = [
    "EventLoopLagMonitor",
    "HostRateLimiter",
    "QoS_Background",
    "QoS_UserInitiated",
    "QoS_UserInteractive",
    "QoS_Utility",
    "QoSPriorityGate",
    "TokenBucket",
    "supervised",
]
