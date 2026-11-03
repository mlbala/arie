from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import ParsedDoc, Parser
from arie_core.registry.manager import RegistryManager

from arie_worker.pool import MAX_RSS, SupervisedPool, WorkerContext

# One parser per worker process, so heavy parsers (Docling) load their models once
# per process rather than once per paper; workers still recycle after 20 tasks.
_PARSERS: dict[tuple[str, str], Parser] = {}


def _extract(
    project_root: str, parser_name: str, settings: str, timeout: float, file_path: str
) -> ParsedDoc:
    """Load the configured parser only in a disposable worker process."""
    os.environ["ARIE_PDF_WORKER"] = "1"
    parser = _PARSERS.get((parser_name, settings))
    if parser is None:
        manager = RegistryManager(project_root=project_root)
        parser = cast(
            Parser, manager.instantiate_plugin("parser", parser_name, **json.loads(settings))
        )
        _PARSERS[(parser_name, settings)] = parser
    ctx = WorkerContext(time.monotonic() + timeout, Path(file_path).stem, BudgetHandle())
    return asyncio.run(parser.parse(ctx, file_path))


class ExtractionPool(SupervisedPool):
    """Recycle after 20 tasks; kill hung or memory-heavy workers before replacing the pool.

    ``settings`` are the parser's constructor options from ``arie.yaml``; a parse that
    runs past ``timeout_sec`` (default 60 s) kills the pool and fails that paper only.
    ``tasks_per_worker`` below 20 recycles workers sooner, for parsers such as Docling
    whose memory after one paper sits near the 1.5 GB RSS cap.
    """

    def __init__(
        self,
        project_root: Path,
        parser_name: str,
        processes: int = 4,
        settings: dict[str, object] | None = None,
        timeout_sec: float = 60.0,
        tasks_per_worker: int = 20,
    ) -> None:
        super().__init__(processes, "PDF", tasks_per_worker)
        self.root = project_root
        self.parser_name = parser_name
        self.settings = json.dumps(settings or {}, sort_keys=True)
        self.timeout_sec = timeout_sec

    async def parse(self, file_path: str) -> ParsedDoc:
        return await self.submit(
            self.timeout_sec,
            _extract,
            str(self.root),
            self.parser_name,
            self.settings,
            self.timeout_sec,
            file_path,
        )


__all__ = ["MAX_RSS", "ExtractionPool", "WorkerContext"]
