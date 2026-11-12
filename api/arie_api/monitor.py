from __future__ import annotations

import asyncio
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from arie_api.runtime import Document, ResearchRuntime, context


def local_storage(db_path: Path | None, pdf_root: Path | None) -> Document:
    """Measure local files off the event loop; bound traversal and label partial measurements."""
    result: Document = {
        "database_bytes": None,
        "pdf_bytes": None,
        "pdf_count": None,
        "disk_free_bytes": None,
        "disk_total_bytes": None,
        "partial": False,
    }
    if db_path is not None:
        result["database_bytes"] = sum(
            path.stat().st_size
            for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))
            if path.exists()
        )
    if pdf_root is not None:
        usage = shutil.disk_usage(pdf_root)
        result.update(disk_free_bytes=usage.free, disk_total_bytes=usage.total)
        size = count = visited = 0
        started = time.monotonic()
        for directory, directories, files in os.walk(pdf_root, followlinks=False):
            directories[:] = [
                name for name in directories if not (Path(directory) / name).is_symlink()
            ]
            for name in files:
                visited += 1
                if visited > 50000 or time.monotonic() - started > 1.0:
                    result["partial"] = True
                    break
                path = Path(directory) / name
                if name.endswith(".pdf") and not path.is_symlink():
                    try:
                        size += path.stat().st_size
                        count += 1
                    except FileNotFoundError:
                        continue  # A concurrent atomic replacement can remove an old entry.
            if result["partial"]:
                break
        result.update(pdf_bytes=size, pdf_count=count)
    return result


class StorageMonitor:
    """Cache bounded storage probes; report unavailable metrics without claiming plugin health."""

    def __init__(self, runtime: ResearchRuntime) -> None:
        self.runtime = runtime
        self.cached: Document | None = None
        self.checked_at = 0.0
        self.lock = asyncio.Lock()

    async def snapshot(self) -> Document:
        async with self.lock:
            if self.cached is None or time.monotonic() - self.checked_at >= 15.0:
                self.cached = await self._probe()
                self.checked_at = time.monotonic()
            return {
                **self.cached,
                "active_jobs": [
                    dict(job)
                    for job in self.runtime.jobs.values()
                    if job["status"] in {"running", "queued"}
                ],
            }

    async def _probe(self) -> Document:
        runtime = self.runtime
        database = cast(dict[str, object], runtime.config.plugins.get("database", {}))
        files = cast(dict[str, object], runtime.config.plugins.get("file_storage", {}))
        db: Document = {
            "provider": database.get("use"),
            "status": "unavailable",
            "size_bytes": None,
            "papers": None,
            "evidence_chunks": None,
        }
        pdf: Document = {
            "provider": files.get("use"),
            "status": "unknown",
            "size_bytes": None,
            "files": None,
            "disk_free_bytes": None,
            "disk_total_bytes": None,
        }
        try:
            async with asyncio.timeout(3.0):
                counts = await runtime.db.fetch(
                    context("storage-health", 3.0),
                    "SELECT status, COUNT(*) AS count FROM arie_library GROUP BY status",
                )
                db.update(status="healthy", papers=sum(int(str(row["count"])) for row in counts))
                if database.get("use") == "sqlite":
                    chunks = await runtime.db.fetch(
                        context("evidence-count", 3.0),
                        "SELECT SUM(json_extract(summary_json, '$.chunk_count')) "
                        "AS count FROM arie_library WHERE status = 'INDEXED'",
                    )
                    db["evidence_chunks"] = int(str(chunks[0]["count"] or 0))
        except Exception:
            db["status"] = "unavailable"
        try:
            metrics = await asyncio.to_thread(
                local_storage,
                runtime.root / str(database.get("path") or "data/arie.db")
                if database.get("use") == "sqlite"
                else None,
                runtime.storage_root if files.get("use") == "local" else None,
            )
            db["size_bytes"] = metrics["database_bytes"]
            pdf.update(
                status=("partial" if metrics["partial"] else "healthy")
                if files.get("use") == "local"
                else "unknown",
                size_bytes=metrics["pdf_bytes"],
                files=metrics["pdf_count"],
                disk_free_bytes=metrics["disk_free_bytes"],
                disk_total_bytes=metrics["disk_total_bytes"],
            )
        except OSError:
            pdf["status"] = "unavailable"
        try:
            async with asyncio.timeout(3.0):
                vector = await runtime.vector_status()
        except Exception as exc:
            provider = cast(dict[str, object], runtime.config.plugins.get("vector_store", {}))
            vector = {
                "provider": provider.get("use"),
                "embedder": runtime.embedder_id,
                "status": "unavailable",
                "size_bytes": None,
                "vectors": None,
                "papers_indexed": None,
                "papers_total": None,
                "detail": f"Vector store check failed: {exc or type(exc).__name__}",
            }
        try:
            async with asyncio.timeout(3.0):
                guard = await runtime.guard_status()
        except Exception as exc:
            guard = {
                "plugin": runtime.guard_config.use,
                "status": "unavailable",
                "detail": f"Guard check failed: {exc or type(exc).__name__}",
            }
        return {
            "checked_at": datetime.now(UTC).isoformat(),
            "database": db,
            "file_storage": pdf,
            "vector_store": vector,
            **runtime.models.status(),
            "guard": guard,
        }
