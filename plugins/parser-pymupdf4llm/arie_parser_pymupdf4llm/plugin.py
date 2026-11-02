from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

from arie_core.plugin_api import Ctx, ParsedDoc, plugin


@plugin(slot="parser", name="pymupdf4llm")
class PyMuPDF4LLMParserPlugin:
    """Extract page Markdown in worker processes; empty or image-only PDFs fail explicitly."""

    async def parse(self, ctx: Ctx, file_path: str) -> ParsedDoc:
        del ctx
        if os.environ.get("ARIE_PDF_WORKER") != "1":
            raise RuntimeError("PDF extraction must run inside arie-worker")
        module = importlib.import_module("pymupdf4llm")
        extract = cast(Callable[..., object], module.to_markdown)
        result = extract(file_path, page_chunks=True, show_progress=False)
        if not isinstance(result, list):
            raise ValueError("Parser did not return page-level extraction")
        pages = cast(list[dict[str, object]], result)
        texts = [str(page.get("text", "")) for page in pages]
        if not any(text.strip() for text in texts):
            raise ValueError("No text could be extracted; this PDF may require OCR")
        headings = [
            match.group(1).strip()
            for text in texts
            for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        ]
        return ParsedDoc(
            paper_id=Path(file_path).stem,
            title=headings[0] if headings else "Research paper",
            source="",
            sections=texts,
            metadata={"page_count": str(len(texts)), "outline_json": json.dumps(headings[:80])},
        )
