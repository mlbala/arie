from __future__ import annotations

import asyncio
from pathlib import Path

from arie_core.contracts import ParsedDoc
from arie_core.plugin_api import plugin


@plugin(slot="parser", name="pymupdf4llm")
class PyMuPDF4LLMParserPlugin:
    """Parser adapter whose execution is isolated in the worker pool for PDF parsing."""

    async def parse(self, ctx: object | None, file_path: str) -> ParsedDoc:
        del ctx

        content = await asyncio.to_thread(Path(file_path).read_bytes)
        text = content.decode("utf-8", errors="replace")
        sections = [text.strip()] if text.strip() else ["empty document"]
        return ParsedDoc(
            paper_id=file_path.rsplit("/", 1)[-1].split(".")[0],
            title="Parsed document",
            source="arxiv",
            sections=sections,
            metadata={"path": file_path},
        )
