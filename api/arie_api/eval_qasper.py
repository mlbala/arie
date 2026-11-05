"""Evaluate answers on the first N papers of Qasper's dev split.

    uv run python -m arie_api.eval_qasper <papers>

Qasper's questions were written by NLP practitioners from a paper's title and abstract,
so they are an outside check next to ``eval_paper``'s self-generated questions. The dev
file is downloaded once into ``data/eval/qasper/``. For each of the first N papers (sorted
by id, so a run can be repeated) the command imports ``https://arxiv.org/pdf/<id>`` through
the normal import path, builds its vectors, saves the converted cases (see
``arie_core.harness.qasper``) and runs them in strict mode, exactly as chat answers.

Prints one JSON line per answer, then a last line with the paper and question counts and the
summary over every answer, shaped like ``eval_paper``'s. Papers that fail to import are
counted and skipped, not replaced, so the slice stays the same between runs. Exits 1 if no
paper was evaluated or a run did not complete. The papers join your library.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import msgspec
from arie_core.harness.paper_eval import PaperEvalResult, summarize
from arie_core.harness.qasper import (
    DEV_MEMBER,
    MAX_ARCHIVE_BYTES,
    ORIGIN,
    QasperPaper,
    arxiv_pdf_url,
    convert_questions,
    extract_dev,
    first_papers,
    parse_dev,
)

from arie_api.paper_eval import MODES
from arie_api.runtime import ResearchRuntime

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "eval" / "qasper"
QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
IMPORT_SECONDS = 900.0
VECTOR_ATTEMPTS = 3
VECTOR_RETRY_SECONDS = 20.0


def _download() -> bytes:
    body = bytearray()
    with httpx.stream("GET", QASPER_URL, timeout=60.0, follow_redirects=True) as response:
        response.raise_for_status()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > MAX_ARCHIVE_BYTES:
                raise ValueError("The Qasper archive is larger than expected")
    return bytes(body)


def _read_dev(cache: Path) -> bytes:
    """Blocking: the cached dev file, downloaded and unpacked (atomically) the first time."""
    path = cache / DEV_MEMBER
    if not path.exists():
        raw = extract_dev(_download())
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(raw)
        partial.replace(path)
    return path.read_bytes()


async def load_dev(cache: Path = CACHE) -> dict[str, QasperPaper]:
    """Read the dev split from the cache, downloading it the first time."""
    return parse_dev(await asyncio.to_thread(_read_dev, cache))


async def import_paper(runtime: ResearchRuntime, qasper_id: str) -> tuple[str, str]:
    """Import one paper; return its library id and an error message ("" when it is indexed)."""
    existing = await runtime.get_paper(qasper_id)
    if existing is not None and existing.get("status") == "INDEXED":
        return qasper_id, ""
    try:
        job = await runtime.start_import(arxiv_pdf_url(qasper_id))
    except (ValueError, RuntimeError) as exc:
        return qasper_id, str(exc)
    deadline = time.monotonic() + IMPORT_SECONDS
    while runtime.jobs[job["job_id"]]["status"] in {"queued", "running"}:
        if time.monotonic() > deadline:
            return qasper_id, "import timed out"
        await asyncio.sleep(1.0)
    job = runtime.jobs[job["job_id"]]
    paper_id = str(job.get("paper_id") or qasper_id)
    if job["status"] != "completed":
        return paper_id, str(job.get("error") or job.get("message") or job["status"])
    return paper_id, ""


async def build_vectors(runtime: ResearchRuntime, paper_id: str) -> bool:
    """Embed the paper, retrying a few times (a provider may rate-limit); False if it failed."""
    for attempt in range(VECTOR_ATTEMPTS):
        try:
            await runtime.ensure_paper_vectors(paper_id)
            return True
        except Exception as exc:
            print(f"  vectors failed: {type(exc).__name__}", file=sys.stderr, flush=True)
            if attempt + 1 < VECTOR_ATTEMPTS:
                await asyncio.sleep(VECTOR_RETRY_SECONDS * (attempt + 1))
    return False


async def evaluate(papers: int, mode: str = "strict") -> bool:
    dev = await load_dev()
    selected = first_papers(dev, papers)
    runtime = ResearchRuntime(ROOT)
    importer: asyncio.Task[None] | None = None
    counts: Counter[str] = Counter(papers=len(selected))
    conversion: Counter[str] = Counter()
    results: list[PaperEvalResult] = []
    settings: object = {}
    try:
        await runtime.initialize()
        # The first load may download models; answers are then timed as in production.
        await asyncio.wait_for(runtime.models.warm(), 600.0)
        if runtime.embedding is not None:
            await runtime.embedding.embed(["warm-up"], kill_after=600.0)
        importer = asyncio.create_task(runtime.run_imports())
        for number, qasper_id in enumerate(selected, 1):
            print(f"[{number}/{len(selected)}] {qasper_id}", file=sys.stderr, flush=True)
            paper_id, error = await import_paper(runtime, qasper_id)
            paper = None if error else await runtime.get_paper(paper_id)
            if paper is None or paper.get("status") != "INDEXED":
                counts["import_failed"] += 1
                print(f"  import failed: {error or 'not indexed'}", file=sys.stderr, flush=True)
                continue
            evaluations = runtime.evaluations
            if any(case.origin != ORIGIN for case in await evaluations.cases(paper_id)):
                counts["has_other_cases"] += 1
                print("  skipped: the paper has other saved cases", file=sys.stderr, flush=True)
                continue
            cases, outcome = convert_questions(
                paper_id, dev[qasper_id], evaluations.chunks(paper)
            )
            conversion.update(outcome)
            if not cases:
                counts["no_cases"] += 1
                continue
            if not await build_vectors(runtime, paper_id):
                # Keyword-only retrieval would make this paper's numbers incomparable.
                counts["vectors_failed"] += 1
                continue
            await evaluations.save_cases(paper_id, cases)
            run = await evaluations.run_to_completion(paper_id, mode=mode)
            settings = run["settings"]
            if run["status"] != "complete":
                counts["run_failed"] += 1
                print(f"  run failed: {run['message']}", file=sys.stderr, flush=True)
                continue
            counts["evaluated"] += 1
            for item in run["results"]:
                print(json.dumps({"paper_id": paper_id, **item}), flush=True)
                results.append(msgspec.convert(item, PaperEvalResult))
    finally:
        if importer is not None:
            importer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await importer
        runtime.close()
    ok = counts["evaluated"] > 0 and counts["run_failed"] == 0
    print(
        json.dumps(
            {
                "status": "complete" if ok else "incomplete",
                "message": f"Qasper dev, first {len(selected)} papers, {mode} mode",
                "settings": settings,
                "papers": dict(sorted(counts.items())),
                "conversion": dict(sorted(conversion.items())),
                "summary": {"configured": summarize(results)},
            }
        )
    )
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answers on a Qasper dev slice.")
    parser.add_argument("papers", type=int, help="how many papers, taken in id order")
    parser.add_argument("--mode", choices=MODES, default="strict", help="answer path to score")
    args = parser.parse_args(argv)
    if args.papers < 1:
        parser.exit(2, "error: papers must be at least 1\n")
    return args


def main() -> None:
    args = parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.papers, args.mode)) else 1)


if __name__ == "__main__":
    main()
