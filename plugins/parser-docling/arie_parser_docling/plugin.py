from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
from typing import Protocol, cast

from arie_core.plugin_api import Ctx, ParsedDoc, plugin


class _Document(Protocol):
    def num_pages(self) -> int: ...
    def export_to_markdown(self, *, page_no: int | None = None) -> str: ...


class _Result(Protocol):
    @property
    def document(self) -> _Document: ...


class _Converter(Protocol):
    def convert(self, source: str) -> _Result: ...



@plugin(slot="parser", name="docling")
class DoclingParserPlugin:
    """Extract page Markdown with Docling layout analysis, tables and OCR, in arie-worker.

    Unlike PyMuPDF4LLM, Docling runs OCR, so scanned and image-only PDFs yield text.
    It is markedly slower and heavier: the first run downloads its layout and OCR
    models, and a large paper can exceed the worker's 60 s or 1.5 GB limits. Failure
    modes: a missing ``docling`` install raises ``ModuleNotFoundError``; a document with
    no recoverable text raises ``ValueError``; calls outside a worker raise ``RuntimeError``.
    """

    def __init__(self, do_ocr: bool = True, do_table_structure: bool = True) -> None:
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        # Built once per instance: rebuilding reloads every layout and OCR model, which
        # slows each paper and grows worker memory past the 1.5 GB RSS cap.
        self._converter: _Converter | None = None

    async def parse(self, ctx: Ctx, file_path: str) -> ParsedDoc:
        del ctx
        if os.environ.get("ARIE_PDF_WORKER") != "1":
            raise RuntimeError("PDF extraction must run inside arie-worker")
        document = self._convert(file_path)
        texts = [
            document.export_to_markdown(page_no=page) for page in range(1, document.num_pages() + 1)
        ]
        if not any(text.strip() for text in texts):
            raise ValueError("No text could be extracted, even with OCR")
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
            metadata={
                "page_count": str(len(texts)),
                "outline_json": json.dumps(headings[:80]),
                "parser": "docling",
                "ocr": str(self.do_ocr).lower(),
            },
        )

    def _convert(self, file_path: str) -> _Document:
        converter = self._converter
        if converter is None:
            base = importlib.import_module("docling.datamodel.base_models")
            options_module = importlib.import_module("docling.datamodel.pipeline_options")
            converter_module = importlib.import_module("docling.document_converter")
            options = options_module.PdfPipelineOptions()
            options.do_ocr = self.do_ocr
            options.do_table_structure = self.do_table_structure
            converter = cast(
                _Converter,
                converter_module.DocumentConverter(
                    format_options={
                        base.InputFormat.PDF: converter_module.PdfFormatOption(
                            pipeline_options=options
                        )
                    }
                ),
            )
            self._converter = converter
        return converter.convert(file_path).document
