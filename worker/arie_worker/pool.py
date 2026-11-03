from __future__ import annotations

import asyncio
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import psutil
from arie_core.budget import BudgetHandle

MAX_RSS = 1536 * 1024 * 1024


@dataclass(frozen=True)
class WorkerContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


class SupervisedPool:
    """Run CPU work in recycled spawn processes; kill hung or memory-heavy workers.

    Workers recycle after ``tasks_per_child`` tasks (default 20). A call that times out,
    is cancelled, or exceeds the 1.5 GB RSS cap kills the whole pool, which is rebuilt
    on the next call.
    """

    def __init__(self, processes: int, label: str, tasks_per_child: int = 20) -> None:
        self.processes = processes
        self.label = label
        self.tasks_per_child = max(1, tasks_per_child)
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
                                    and psutil.Process(process.pid).memory_info().rss > MAX_RSS
                                ):
                                    raise MemoryError(
                                        f"{self.label} worker exceeded the 1.5 GB RSS limit"
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
