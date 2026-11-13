"""Capture figures of stored papers by hand: ``make figures-backfill``.

The background figure catch-up (``runtime/figures.py``, S26) does this on its own; these
targets stay for manual runs. Each INDEXED paper with a stored PDF and no ``figures`` key is
parsed again in the PDF worker pool; only the parse's figures are kept (ADR 0015), then its
crops (S24) and, with ``ingestion.figure_ocr`` on and its model present, OCR text (S25) are
captured. Pages, chunk text, ids and offsets are unchanged. ``--all`` also refreshes papers
that already have figures. ``--crops`` (``make figure-crops``) only renders crops and
``--ocr`` (``make figure-ocr``) only reads OCR text, for papers that already have figures.
Each run records its steps in ``capture_version``, so the catch-up does not repeat them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

from arie_core.pipeline.figures import CAPTURE_STEPS

from arie_api.runtime import ResearchRuntime
from arie_api.runtime.base import Document, context

ROOT = Path(__file__).resolve().parents[2]


def steps_for(
    paper: Document, refresh: bool, crops_only: bool, ocr_only: bool, ocr: bool
) -> list[str]:
    """The capture steps one run takes for ``paper``; empty when it skips the paper."""
    if crops_only or ocr_only:
        return ["crops" if crops_only else "ocr"] if paper.get("figures") else []
    if "figures" in paper and not refresh:
        return []
    return [step for step in CAPTURE_STEPS if ocr or step != "ocr"]


async def backfill(
    refresh: bool = False, crops_only: bool = False, ocr_only: bool = False
) -> Counter[str]:
    runtime = ResearchRuntime(ROOT)
    counts: Counter[str] = Counter()
    try:
        await runtime.initialize()
        ocr = runtime.ocr_ready()
        if ocr_only and not ocr:
            problem = runtime.ocr_problem or "ingestion.figure_ocr is off in arie.yaml"
            raise SystemExit(f"--ocr cannot run: {problem}")
        rows = await runtime.db.fetch(
            context("figures"),
            "SELECT paper_id FROM arie_library WHERE status = 'INDEXED' ORDER BY paper_id",
        )
        for row in rows:
            paper = await runtime.get_paper(str(row["paper_id"]))
            if paper is None:
                continue
            steps = steps_for(paper, refresh, crops_only, ocr_only, ocr)
            if not steps:
                counts["skipped_done"] += 1
                continue
            if paper.get("pdf_deleted") or await runtime.stored_pdf(paper) is None:
                counts["skipped_no_pdf"] += 1
                continue
            found, errors = await runtime.capture_paper(paper, steps)
            for error in errors:
                print(f"{paper['id']}: {error}", file=sys.stderr, flush=True)
            if any(error.startswith("figures:") for error in errors):
                counts["parse_failed"] += 1
                continue
            await runtime.save_paper(paper)
            counts.update(found)
            counts["papers"] += 1
            if "ocr" in steps:
                print(
                    f"{paper['id']}: OCR {found['ocr_texts']} of {found['ocr_targets']} "
                    f"figures, {found['ocr_ms']} ms",
                    file=sys.stderr,
                    flush=True,
                )
            if "figures" not in steps:
                continue
            chunks = cast(list[Document], paper.get("chunks") or [])
            figures = cast(list[Document], paper.get("figures") or [])
            counts["papers_with_figures"] += bool(figures)
            counts["caption_chunks"] += sum(
                "figure" in chunk and not chunk.get("figure_text") for chunk in chunks
            )
            counts["mention_chunks"] += sum("mentions" in chunk for chunk in chunks)
            # S23: words drawn inside figures, already in the page text, now tagged.
            counts["figure_text_spans"] += sum(
                len(cast(list[object], figure.get("text_spans") or [])) for figure in figures
            )
            counts["figure_text_chunks"] += sum(bool(chunk.get("figure_text")) for chunk in chunks)
    finally:
        runtime.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Add figure records to indexed papers.")
    parser.add_argument("--all", action="store_true", help="also refresh papers with figures")
    parser.add_argument("--crops", action="store_true", help="only render figure crops")
    parser.add_argument("--ocr", action="store_true", help="only OCR bitmap figures (S25)")
    args = parser.parse_args()
    counts = asyncio.run(backfill(args.all, args.crops, args.ocr))
    print(json.dumps(dict(sorted(counts.items()))))


if __name__ == "__main__":
    main()
