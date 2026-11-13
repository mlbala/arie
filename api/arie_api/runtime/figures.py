"""Figure capture: every stored paper's figures, crops and OCR text, kept current (S26).

ADR 0019, SPEC 6.2 and 9. Sync captures a new paper as it indexes it. Papers indexed before
a capture step existed, or before its current version, catch up here in the background:
one paper at a time in the PDF pool, each step behind the foreground gate (S16), the way
vectors and guard scores catch up. ``capture_version`` on the paper records which version of
each step ran, so a later change re-runs only that step. A paper whose PDF is not stored
cannot be captured; it is counted, so "no figures" is not read as "has none".
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from arie_core.pipeline.figures import (
    CAPTURE_KEY,
    CAPTURE_STEPS,
    attach_figures,
    capture_due,
    capture_record,
    ocr_chunks,
)
from arie_core.registry.manager import ArieConfig

from arie_api.runtime.base import Document, context, logger
from arie_api.runtime.vectors import VectorIndex

CAPTURE_RETRY_SECONDS = 300.0


def _chunk_ids(paper: Document) -> list[object]:
    return [chunk.get("chunk_id") for chunk in cast(list[Document], paper.get("chunks") or [])]


class FigureCapture(VectorIndex):
    """Capture steps for one paper, and the background catch-up over the library."""

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        super().__init__(root, config)
        self.capture_wakeup = asyncio.Event()
        # Set after each catch-up pass, so a caller can wait for one to finish.
        self.capture_idle = asyncio.Event()
        self.capture_errors: dict[str, str] = {}
        self.capture_no_pdf: set[str] = set()
        # Counts from the catch-up's last pass; the system pane reads them (None before it).
        self.capture_pending: int | None = None
        self.capture_total = 0

    async def capture_paper(
        self, document: Document, steps: Sequence[str]
    ) -> tuple[Counter[str], list[str]]:
        """Run ``steps`` (from ``capture_due``) on a paper and record each one that ran.

        ``figures`` parses the stored PDF again and keeps only its figures (ADR 0015);
        ``crops`` renders them (S24); ``ocr`` reads bitmap figures (S25). A failed step is
        logged and left unrecorded, so it runs again later; the others still run. When the
        paper has chunks, its OCR chunks follow its figure records. The caller saves.
        Returns the counts and the failed steps' errors.
        """
        paper_id = str(document.get("id", ""))
        counts: Counter[str] = Counter()
        errors: list[str] = []
        record = capture_record(document)
        for step in steps:
            try:
                await self.foreground.wait_quiet()
                if step == "figures":
                    pdf = await self.stored_pdf(document)
                    if pdf is None:
                        raise FileNotFoundError("no stored PDF")
                    parsed = await self.pool.parse(str(pdf))
                    await self.remove_figure_crops(document)
                    counts["captions_not_in_stored_text"] += attach_figures(
                        cast(dict[str, object], document), parsed.figures
                    )
                    counts["figures"] += len(parsed.figures)
                    # New records: their crops and OCR text are gone until those steps run.
                    record = {}
                elif step == "crops":
                    counts.update(await self.store_figure_crops(document))
                elif step == "ocr":
                    counts.update(await self.store_figure_ocr(document))
                else:
                    raise ValueError(f"unknown capture step {step}")
            except Exception as exc:
                logger.warning(
                    "Figure capture step %s failed for %s", step, paper_id, exc_info=True
                )
                errors.append(f"{step}: {exc or type(exc).__name__}")
                if step == "figures":
                    break
                continue
            record[step] = CAPTURE_STEPS[step]
        document[CAPTURE_KEY] = record
        chunks = cast(list[Document], document.get("chunks") or [])
        if chunks and ({"figures", "ocr"} & set(steps)):
            # After the page chunks, so their ids are unchanged (S25).
            kept = [chunk for chunk in chunks if not chunk.get("ocr")]
            added = (
                ocr_chunks(paper_id, document.get("figures") or [], len(kept))
                if self.config.ingestion.figure_ocr
                else []
            )
            document["chunks"] = [*kept, *added]
            counts["ocr_chunks"] += len(added)
        return counts, errors

    async def catch_up_paper(self, paper_id: str) -> Counter[str]:
        """Bring one indexed paper to the current capture version and save it."""
        paper = await self.get_paper(paper_id)
        if paper is None or paper.get("status") != "INDEXED":
            return Counter()
        steps = capture_due(paper, self.ocr_ready())
        if not steps:
            return Counter()
        if paper.get("pdf_deleted") or await self.stored_pdf(paper) is None:
            self.capture_no_pdf.add(paper_id)
            return Counter()
        before = _chunk_ids(paper)
        counts, errors = await self.capture_paper(paper, steps)
        await self.save_paper(paper)
        if _chunk_ids(paper) != before:
            # OCR chunks changed the paper's revision: re-embed and re-score it.
            self.vector_wakeup.set()
            self.guard_wakeup.set()
        counts["papers"] += 1
        if errors:
            raise RuntimeError("; ".join(errors))
        return counts

    async def _capture_backlog(self) -> tuple[list[str], int]:
        """Indexed papers behind the current capture version, newest first."""
        rows = await self.db.fetch(
            context("capture-backlog", 10.0),
            "SELECT paper_id, summary_json FROM arie_library WHERE status = 'INDEXED' "
            "ORDER BY published_at DESC",
        )
        ocr = self.ocr_ready()
        pending: list[str] = []
        for row in rows:
            paper_id = str(row["paper_id"])
            if paper_id in self.capture_no_pdf:
                continue
            summary = cast(Document, json.loads(str(row["summary_json"])))
            if CAPTURE_KEY in summary:
                due = capture_due({CAPTURE_KEY: summary[CAPTURE_KEY]}, ocr)
            else:
                # Saved before S26: read what the paper holds, once.
                paper = await self.get_paper(paper_id)
                due = capture_due(paper, ocr) if paper is not None else []
            if due:
                pending.append(paper_id)
        return pending, len(rows)

    async def run_figure_capture(self) -> None:
        """Keep every stored paper's figures captured: at startup, after each sync, on retry.

        Retries every 5 minutes while a paper failed or the OCR model is missing, so
        ``make ocr-model`` takes effect without a restart.
        """
        while True:
            self.capture_wakeup.clear()
            if self.config.ingestion.figure_ocr and self.ocr_problem is not None:
                await asyncio.to_thread(self.check_ocr)
            pending, total = await self._capture_backlog()
            self.capture_pending, self.capture_total = len(pending), total
            for paper_id in pending:
                try:
                    await self.catch_up_paper(paper_id)
                    self.capture_errors.pop(paper_id, None)
                except Exception as exc:
                    logger.warning("Figure capture failed for %s", paper_id, exc_info=True)
                    self.capture_errors[paper_id] = str(exc) or type(exc).__name__
                self.capture_pending = max(0, (self.capture_pending or 0) - 1)
            self.capture_pending = len(self.capture_errors)
            self.capture_idle.set()
            try:
                retry = self.capture_errors or self.ocr_problem is not None
                await asyncio.wait_for(
                    self.capture_wakeup.wait(), CAPTURE_RETRY_SECONDS if retry else None
                )
            except TimeoutError:
                pass

    def figure_capture_status(self) -> Document:
        """Figure capture for the system pane: progress, papers with no PDF, OCR state."""
        ocr = (
            "off"
            if not self.config.ingestion.figure_ocr
            else "on"
            if self.ocr_problem is None
            else "unavailable"
        )
        pending, total = self.capture_pending, self.capture_total
        no_pdf = len(self.capture_no_pdf)
        if pending is None:
            return {
                "status": "indexing",
                "detail": "Checking which papers need their figures captured.",
                "ocr": ocr,
                "ocr_problem": self.ocr_problem,
            }
        captured = total - pending - no_pdf
        parts = [f"{captured} of {total} papers captured."]
        if no_pdf:
            parts.append(f"{no_pdf} have no stored PDF, so their figures are unknown.")
        if self.ocr_problem is not None:
            parts.append(f"Figure OCR: {self.ocr_problem}.")
        elif ocr == "off":
            parts.append("Figure OCR is off.")
        if self.capture_errors:
            status = "degraded"
            parts.append(f"{len(self.capture_errors)} failed; retrying every 5 minutes.")
        elif self.ocr_problem is not None:
            status = "degraded"
        elif pending:
            status = "indexing"
        else:
            status = "healthy"
        return {
            "status": status,
            "detail": " ".join(parts),
            "ocr": ocr,
            "ocr_problem": self.ocr_problem,
            "papers_captured": captured,
            "papers_total": total,
            "papers_pending": pending,
            "papers_no_pdf": no_pdf,
        }


__all__ = ["CAPTURE_RETRY_SECONDS", "FigureCapture"]
