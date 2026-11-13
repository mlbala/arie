from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import msgspec
from arie_core.budget import BudgetHandle
from arie_core.plugin_api import ParsedDoc, Parser
from arie_core.registry.manager import PdfCheckConfig, RegistryManager
from pydantic import BaseModel, ConfigDict, Field

from arie_worker.figure_crops import Box, CropLimits, Crops, render_crops
from arie_worker.figure_ocr import OcrLimits, OcrTexts, ocr_figures
from arie_worker.hidden_text import strip_hidden
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
    hidden_text: bool = False,
) -> ParsedDoc:
    """Check the PDF's structure, then parse it, only in a disposable worker process.

    The check runs before the parser is loaded, so a rejected file (``PdfRejected``)
    never reaches it. With ``hidden_text``, the parser reads a temporary copy without the
    text a reader cannot see; the copy keeps the file name, since parsers take the paper
    id from it, and is deleted after the parse, whether it succeeds or not.
    """
    os.environ["ARIE_PDF_WORKER"] = "1"
    if pdf_check is not None and pdf_check.enabled:
        check_pdf(file_path, pdf_check)
    if not hidden_text:
        return _parse(project_root, parser_name, settings, timeout, file_path)
    with tempfile.TemporaryDirectory(prefix="arie-hidden-") as folder:
        copy = Path(folder) / Path(file_path).name
        hidden = strip_hidden(file_path, copy)
        parsed = _parse(
            project_root, parser_name, settings, timeout, str(copy) if hidden else file_path
        )
    metadata = {**parsed.metadata, "hidden_text_chars": str(hidden)}
    return msgspec.structs.replace(parsed, metadata=metadata)


def _parse(
    project_root: str, parser_name: str, settings: str, timeout: float, file_path: str
) -> ParsedDoc:
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
    passes the structural check (``pdf_check``, from ``ingestion.pdf_check``); with
    ``hidden_text`` (``ingestion.hidden_text``) the parser reads a copy without hidden
    text. A parse that runs past ``limits.timeout_sec`` or lets a worker exceed
    ``limits.max_rss_mb`` kills the pool and fails that paper only. Workers recycle after
    ``limits.tasks_per_worker`` papers, so a heavy parser (Docling) can use a fresh process
    for every paper.
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
        hidden_text: bool = True,
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
        self.hidden_text = hidden_text

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
            self.hidden_text,
        )

    async def render_crops(
        self, file_path: str, figures: Sequence[tuple[int, Box]], limits: CropLimits | None = None
    ) -> Crops:
        """Render figure crops in a PDF worker (S24); never in the API process."""
        return await self.submit(
            self.timeout_sec, render_crops, file_path, list(figures), limits or CropLimits()
        )

    async def ocr_figures(
        self,
        file_path: str,
        figures: Sequence[tuple[int, Box]],
        model: str,
        limits: OcrLimits | None = None,
    ) -> OcrTexts:
        """OCR bitmap figures in a PDF worker (S25), under the parse time and memory limits.

        ``model`` is the English recognition model's path (ADR 0018).
        """
        return await self.submit(
            self.timeout_sec,
            ocr_figures,
            file_path,
            list(figures),
            limits or OcrLimits(),
            None,
            model,
        )


__all__ = ["MAX_RSS", "ExtractionLimits", "ExtractionPool", "WorkerContext"]
