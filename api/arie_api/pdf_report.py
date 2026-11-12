"""Run the structural PDF check over every stored PDF and report: ``make pdf-report``.

Each file is inspected in its own worker process (60 s, 1.5 GB, like a parse). The report
prints how many files each rule would quarantine under the configured
``ingestion.pdf_check`` limits, and the largest object count, page count and filter-chain
depth seen, which the limits are set from (about 10 times each, S18). It changes no
paper and no file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from arie_core.pipeline import PdfRejected
from arie_core.registry.manager import ArieConfig, PdfCheckConfig
from arie_worker.pdf_check import ACTIVE_RULES, PdfFacts, inspect_pdf
from arie_worker.pool import BACKGROUND_NICE, SupervisedPool

from arie_api.providers import config_path

ROOT = Path(__file__).resolve().parents[2]
CHECK_SECONDS = 60.0
POOLS = 4
# The rule a value over its limit fails, per measure.
SIZE_RULES = {
    "objects": "too_many_objects",
    "pages": "too_many_pages",
    "filter_depth": "filter_depth",
}


@dataclass
class Report:
    limits: PdfCheckConfig
    checked: int = 0
    rules: Counter[str] = field(default_factory=Counter[str])
    rejected: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    most: dict[str, tuple[int, str]] = field(default_factory=dict[str, tuple[int, str]])
    slowest: tuple[float, str] = (0.0, "")

    def add(self, name: str, result: PdfFacts | str, seconds: float) -> None:
        self.checked += 1
        self.slowest = max(self.slowest, (seconds, name))
        if isinstance(result, str):
            failed = [result]
        else:
            failed = list(result.rules)
            for measure, value, limit in (
                ("objects", result.objects, self.limits.max_objects),
                ("pages", result.pages, self.limits.max_pages),
                ("filter_depth", result.filter_depth, self.limits.max_filter_depth),
            ):
                self.most[measure] = max(self.most.get(measure, (0, "")), (value, name))
                if value > limit:
                    failed.append(SIZE_RULES[measure])
        self.rules.update(failed)
        if failed:
            self.rejected.append((name, ", ".join(failed)))

    def lines(self, storage: Path) -> list[str]:
        limits = self.limits
        out = [
            f"Checked {self.checked} PDFs under {storage}",
            f"Limits: objects {limits.max_objects:,}, pages {limits.max_pages:,}, "
            f"filter depth {limits.max_filter_depth} (enabled: {limits.enabled})",
            f"Would quarantine: {len(self.rejected)}",
        ]
        for rule in (*ACTIVE_RULES, "encrypted", "unreadable", "timeout", "memory"):
            out.append(f"  {rule}: {self.rules[rule]}")
        for rule in SIZE_RULES.values():
            out.append(f"  {rule} (over the limit): {self.rules[rule]}")
        for measure in SIZE_RULES:
            value, name = self.most.get(measure, (0, "-"))
            out.append(f"Max {measure.replace('_', ' ')}: {value:,} ({name})")
        out.append(f"Slowest check: {self.slowest[0]:.2f} s ({self.slowest[1] or '-'})")
        out.extend(f"  rejected {name}: {rules}" for name, rules in self.rejected)
        return out


async def inspect_in_worker(pool: SupervisedPool, path: Path) -> PdfFacts | str:
    try:
        return await pool.submit(CHECK_SECONDS, inspect_pdf, str(path))
    except PdfRejected as exc:
        return str(exc).removeprefix("quarantined: ")
    except TimeoutError:
        return "timeout"
    except MemoryError:
        return "memory"


async def run(storage: Path, paths: list[Path], limits: PdfCheckConfig) -> Report:
    report = Report(limits)
    files = iter(paths)
    workers = max(1, min(POOLS, os.process_cpu_count() or 1))

    async def drain() -> None:
        # A fresh process for every file, so one bad PDF cannot affect the next.
        pool = SupervisedPool(1, "PDF report", tasks_per_child=1, nice=BACKGROUND_NICE)
        try:
            for path in files:
                started = time.monotonic()
                result = await inspect_in_worker(pool, path)
                report.add(str(path.relative_to(storage)), result, time.monotonic() - started)
        finally:
            pool.close()

    await asyncio.gather(*(drain() for _ in range(workers)))
    return report


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = ArieConfig.from_yaml(config_path(ROOT))
    files = cast(dict[str, object], config.plugins.get("file_storage", {}))
    storage = ROOT / str(files.get("root") or "data/papers")
    paths = sorted(storage.rglob("*.pdf"))
    report = asyncio.run(run(storage, paths, config.ingestion.pdf_check))
    print("\n".join(report.lines(storage)))


if __name__ == "__main__":
    main()
