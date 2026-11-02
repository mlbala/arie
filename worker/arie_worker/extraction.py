from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import psutil
from arie_core.budget import BudgetHandle
from arie_core.plugin_api import ParsedDoc, Parser
from arie_core.registry.manager import RegistryManager

MAX_RSS = 1536 * 1024 * 1024


@dataclass(frozen=True)
class WorkerContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


def _extract(project_root: str, parser_name: str, file_path: str) -> ParsedDoc:
    """Load the configured parser only in a disposable worker process."""
    os.environ["ARIE_PDF_WORKER"] = "1"
    manager = RegistryManager(project_root=project_root)
    parser = cast(Parser, manager.instantiate_plugin("parser", parser_name))
    ctx = WorkerContext(time.monotonic() + 60.0, Path(file_path).stem, BudgetHandle())
    return asyncio.run(parser.parse(ctx, file_path))


class ExtractionPool:
    """Recycle after 20 tasks; kill hung or memory-heavy workers before replacing the pool."""

    def __init__(self, project_root: Path, parser_name: str, processes: int = 4) -> None:
        self.root = project_root
        self.parser_name = parser_name
        self.processes = processes
        self._pool: ProcessPoolExecutor | None = None
        self._lock = asyncio.Lock()

    async def parse(self, file_path: str) -> ParsedDoc:
        async with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.processes,
                    mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=20,
                )
            pool = self._pool
            future = asyncio.get_running_loop().run_in_executor(
                pool,
                _extract,
                str(self.root),
                self.parser_name,
                file_path,
            )
            try:
                async with asyncio.timeout(60.0):
                    while not future.done():
                        processes = pool._processes  # pyright: ignore[reportPrivateUsage]
                        for process in (processes or {}).values():
                            try:
                                if (
                                    process.pid
                                    and psutil.Process(process.pid).memory_info().rss > MAX_RSS
                                ):
                                    raise MemoryError("PDF worker exceeded the 1.5 GB RSS limit")
                            except psutil.NoSuchProcess:
                                pass
                        await asyncio.sleep(0.1)
                    return await future
            except BaseException:
                pool.kill_workers()
                self._pool = None
                raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.kill_workers()
            self._pool = None
