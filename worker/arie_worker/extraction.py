from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import ParsedDoc, Parser
from arie_core.registry.manager import PdfCheckConfig, RegistryManager
from pydantic import BaseModel, ConfigDict, Field

from arie_worker.pdf_check import check_pdf
from arie_worker.pool import (
    BACKGROUND_NICE,
    DEFAULT_MAX_RSS_MB,
    MAX_RSS,
    SupervisedPool,
    WorkerContext,
)

# One parser per worker process, so heavy parsers (Docling) load their models once
# per process rather than once per paper; workers still recycle after 20 tasks.
_PARSERS: dict[tuple[str, str], Parser] = {}


def _extract(
    project_root: str,
    parser_name: str,
    settings: str,
    timeout: float,
    file_path: str,
    pdf_check: PdfCheckConfig | None = None,
) -> ParsedDoc:
    """Check the PDF's structure, then parse it, only in a disposable worker process.

    The check runs before the parser is loaded, so a rejected file (``PdfRejected``)
    never reaches it.
    """
    os.environ["ARIE_PDF_WORKER"] = "1"
    if pdf_check is not None and pdf_check.enabled:
        check_pdf(file_path, pdf_check)
    parser = _PARSERS.get((parser_name, settings))
    if parser is None:
        manager = RegistryManager(project_root=project_root)
        parser = cast(
            Parser, manager.instantiate_plugin("parser", parser_name, **json.loads(settings))
        )
        _PARSERS[(parser_name, settings)] = parser
    ctx = WorkerContext(time.monotonic() + timeout, Path(file_path).stem, BudgetHandle())
    return asyncio.run(parser.parse(ctx, file_path))


class ExtractionLimits(BaseModel):
    """Worker limits read from ``plugins.parser``; every other key goes to the parser.

    Defaults suit PyMuPDF4LLM. Heavy parsers raise ``timeout_sec`` and ``max_rss_mb`` and
    lower ``tasks_per_worker``; values outside the bounds raise ``ValidationError`` at
    startup rather than silently killing every parse.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_sec: float = Field(default=60.0, gt=0, le=1800)
    tasks_per_worker: int = Field(default=20, ge=1, le=20)
    max_rss_mb: int = Field(default=DEFAULT_MAX_RSS_MB, ge=DEFAULT_MAX_RSS_MB, le=8192)

    @classmethod
    def pop_from(cls, parser_config: dict[str, object]) -> ExtractionLimits:
        """Remove the limit keys from ``parser_config`` and validate them."""
        keys = [key for key in cls.model_fields if key in parser_config]
        return cls.model_validate({key: parser_config.pop(key) for key in keys})


class ExtractionPool(SupervisedPool):
    """Parse PDFs in recycled workers; kill hung or memory-heavy workers, then rebuild.

    ``settings`` are the parser's constructor options from ``arie.yaml``. Each file first
    passes the structural check (``pdf_check``, from ``ingestion.pdf_check``). A parse that
    runs past ``limits.timeout_sec`` or lets a worker exceed ``limits.max_rss_mb`` kills
    the pool and fails that paper only. Workers recycle after ``limits.tasks_per_worker``
    papers, so a heavy parser (Docling) can use a fresh process for every paper.
    Workers run at background priority, and ``processes`` is capped to the CPUs this
    process may use, since each parser already spreads one paper over several threads.
    """

    def __init__(
        self,
        project_root: Path,
        parser_name: str,
        processes: int = 4,
        settings: dict[str, object] | None = None,
        limits: ExtractionLimits | None = None,
        pdf_check: PdfCheckConfig | None = None,
    ) -> None:
        limits = limits or ExtractionLimits()
        processes = max(1, min(processes, os.process_cpu_count() or 1))
        super().__init__(
            processes, "PDF", limits.tasks_per_worker, limits.max_rss_mb, BACKGROUND_NICE
        )
        self.root = project_root
        self.parser_name = parser_name
        self.settings = json.dumps(settings or {}, sort_keys=True)
        self.timeout_sec = limits.timeout_sec
        self.pdf_check = pdf_check or PdfCheckConfig()

    async def parse(self, file_path: str) -> ParsedDoc:
        return await self.submit(
            self.timeout_sec,
            _extract,
            str(self.root),
            self.parser_name,
            self.settings,
            self.timeout_sec,
            file_path,
            self.pdf_check,
        )


__all__ = ["MAX_RSS", "ExtractionLimits", "ExtractionPool", "WorkerContext"]
