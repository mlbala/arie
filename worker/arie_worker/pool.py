from __future__ import annotations

import asyncio
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import psutil
from arie_core.budget import BudgetHandle

# Default per-worker memory cap. Only the PDF parser pool may raise it, through
# ``plugins.parser.max_rss_mb``; embedding and entailment pools always use this value.
DEFAULT_MAX_RSS_MB = 1536
MAX_RSS = DEFAULT_MAX_RSS_MB * 1024 * 1024
# Niceness for background pools (PDF extraction, embedding). Linux then gives a nice-0
# process (the API, the claim checker) about nine times their CPU share under contention,
# while background work still uses every idle core.
BACKGROUND_NICE = 10


def _lower_priority(increment: int) -> None:
    """Worker initializer: raise this process's niceness (needs no privileges)."""
    if increment:
        os.nice(increment)


@dataclass(frozen=True)
class WorkerContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


class SupervisedPool:
    """Run CPU work in recycled spawn processes; kill hung or memory-heavy workers.

    Workers recycle after ``tasks_per_child`` tasks (default 20). A call that times out,
    is cancelled, or lets any worker's RSS exceed ``max_rss_mb`` (default 1536, 1.5 GB)
    kills the whole pool, which is rebuilt on the next call; the caller sees
    ``TimeoutError``, ``CancelledError`` or ``MemoryError``. ``nice`` lowers the OS
    priority of every worker so background pools yield the CPU to interactive work.
    """

    def __init__(
        self,
        processes: int,
        label: str,
        tasks_per_child: int = 20,
        max_rss_mb: int = DEFAULT_MAX_RSS_MB,
        nice: int = 0,
    ) -> None:
        if max_rss_mb <= 0:
            raise ValueError("max_rss_mb must be positive")
        if not 0 <= nice <= 19:
            raise ValueError("nice must be 0-19")
        self.processes = processes
        self.nice = nice
        self.label = label
        self.tasks_per_child = max(1, tasks_per_child)
        self.max_rss_mb = max_rss_mb
        self._max_rss = max_rss_mb * 1024 * 1024
        self._pool: ProcessPoolExecutor | None = None
        self._lock = asyncio.Lock()

    async def submit[T](self, kill_after: float, func: Callable[..., T], *args: object) -> T:
        """Run ``func(*args)`` in a worker; kill the pool if it runs past ``kill_after`` s."""
        async with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.processes,
                    mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=self.tasks_per_child,
                    initializer=_lower_priority,
                    initargs=(self.nice,),
                )
            pool = self._pool
            future = asyncio.get_running_loop().run_in_executor(pool, func, *args)
            try:
                async with asyncio.timeout(kill_after):
                    while not future.done():
                        processes = pool._processes  # pyright: ignore[reportPrivateUsage]
                        # The executor replaces recycled workers from its own thread;
                        # copy (atomic) before iterating or the dict changes underfoot.
                        for process in dict(processes or {}).values():
                            try:
                                if (
                                    process.pid
                                    and psutil.Process(process.pid).memory_info().rss
                                    > self._max_rss
                                ):
                                    raise MemoryError(
                                        f"{self.label} worker exceeded its "
                                        f"{self.max_rss_mb} MB memory limit"
                                    )
                            except psutil.NoSuchProcess:
                                pass
                        await asyncio.sleep(0.05)
                    return await future
            except BaseException:
                pool.kill_workers()
                self._pool = None
                raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.kill_workers()
            self._pool = None
