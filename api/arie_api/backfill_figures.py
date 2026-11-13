"""Add figure records to papers indexed before S22: ``make figures-backfill``.

Each INDEXED paper with a stored PDF and no ``figures`` key is parsed again in the PDF
worker pool; only the parse's figures are kept (ADR 0015). Captions are found again in the
stored page text and chunks gain ``figure``/``mentions`` tags. Pages, chunk text, ids and
offsets are unchanged, so vectors, guard scores and paper packs stay valid. ``--all``
also refreshes papers that already have figures, and adds S23's figure-text spans and tags.
Each parsed paper also gets its figure crops (S24). ``--crops`` (``make figure-crops``) only
renders crops for papers that already have figures, without parsing them again. With
``ingestion.figure_ocr`` on, each parsed paper's bitmap figures are read by OCR (S25), and
``--ocr`` (``make figure-ocr``) does only that for papers that already have figures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

from arie_core.pipeline.figures import attach_figures, ocr_chunks

from arie_api.runtime import ResearchRuntime
from arie_api.runtime.base import Document, context

ROOT = Path(__file__).resolve().parents[2]


async def refresh_ocr(runtime: ResearchRuntime, paper: Document) -> Counter[str]:
    """OCR the paper's bitmap figures again and replace its OCR chunks (S25)."""
    chunks = [c for c in cast(list[Document], paper.get("chunks") or []) if not c.get("ocr")]
    counts: Counter[str] = Counter(await runtime.store_figure_ocr(paper))
    added = ocr_chunks(str(paper["id"]), paper.get("figures") or [], len(chunks))
    paper["chunks"] = [*chunks, *added]
    counts["ocr_chunks"] += len(added)
    print(
        f"{paper['id']}: OCR {counts['ocr_texts']} of {counts['ocr_targets']} figures, "
        f"{counts['ocr_ms']} ms",
        file=sys.stderr,
        flush=True,
    )
    return counts


async def backfill(
    refresh: bool = False, crops_only: bool = False, ocr_only: bool = False
) -> Counter[str]:
    runtime = ResearchRuntime(ROOT)
    counts: Counter[str] = Counter()
    try:
        await runtime.initialize()
        ocr = runtime.config.ingestion.figure_ocr
        if ocr_only and not ocr:
            raise SystemExit("--ocr needs ingestion.figure_ocr: true in arie.yaml")
        rows = await runtime.db.fetch(
            context("figures"),
            "SELECT paper_id FROM arie_library WHERE status = 'INDEXED' ORDER BY paper_id",
        )
        for row in rows:
            paper = await runtime.get_paper(str(row["paper_id"]))
            if ocr_only:
                if paper is not None and paper.get("figures"):
                    counts.update(await refresh_ocr(runtime, paper))
                    await runtime.save_paper(paper)
                    counts["papers"] += 1
                continue
            if crops_only:
                if paper is not None and paper.get("figures"):
                    counts.update(await runtime.store_figure_crops(paper))
                    await runtime.save_paper(paper)
                    counts["papers"] += 1
                continue
            if paper is None or ("figures" in paper and not refresh):
                counts["skipped_done"] += paper is not None
                continue
            path = Path(str(paper.get("file_path") or ""))
            if paper.get("pdf_deleted") or not await asyncio.to_thread(path.is_file):
                counts["skipped_no_pdf"] += 1
                continue
            try:
                parsed = await runtime.pool.parse(str(path))
            except Exception as exc:
                counts["parse_failed"] += 1
                print(f"{paper['id']}: parse failed: {exc}")
                continue
            await runtime.remove_figure_crops(paper)
            missing = attach_figures(cast(dict[str, object], paper), parsed.figures)
            counts.update(await runtime.store_figure_crops(paper))
            if ocr:
                counts.update(await refresh_ocr(runtime, paper))
            await runtime.save_paper(paper)
            counts["papers"] += 1
            counts["papers_with_figures"] += bool(parsed.figures)
            counts["figures"] += len(parsed.figures)
            counts["boxes_uncaptioned"] += int(
                parsed.metadata.get("figure_boxes_uncaptioned", "0")
            )
            counts["captions_not_in_stored_text"] += missing
            counts["caption_chunks"] += sum(
                "figure" in chunk and not chunk.get("figure_text")
                for chunk in cast(list[Document], paper.get("chunks") or [])
            )
            counts["mention_chunks"] += sum(
                "mentions" in chunk for chunk in cast(list[Document], paper.get("chunks") or [])
            )
            # S23: words drawn inside figures, already in the page text, now tagged.
            counts["figure_text_spans_parsed"] += sum(len(f.text_spans) for f in parsed.figures)
            counts["figure_text_spans"] += sum(
                len(cast(list[object], figure.get("text_spans") or []))
                for figure in cast(list[Document], paper.get("figures") or [])
            )
            counts["figure_text_chunks"] += sum(
                bool(chunk.get("figure_text"))
                for chunk in cast(list[Document], paper.get("chunks") or [])
            )
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
