from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import BinaryIO

from arie_core.plugin_api import plugin


@plugin(slot="file_storage", name="local")
class LocalFileStoragePlugin:
    """Atomic local file storage that verifies SHA-256 and size before commit."""

    CHUNK_SIZE = 64 * 1024

    def __init__(self, root: str | Path = "data/papers") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_target(self, key: str | Path) -> Path:
        candidate = Path(str(key))
        if not str(candidate).strip():
            raise ValueError("storage key must not be empty")
        if candidate.is_absolute():
            candidate = Path(*candidate.parts[1:])
        if ".." in candidate.parts:
            raise ValueError("storage key must not escape the storage root")
        target = self.root / candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _part_path(self, target: Path) -> Path:
        return target.with_name(f"{target.name}.part")

    async def read(self, ctx: object | None, path: str | Path) -> bytes:
        del ctx
        target = self._resolve_target(path)
        return target.read_bytes()

    async def write(self, ctx: object | None, path: str | Path, content: bytes) -> None:
        del ctx
        await self.put_stream(
            None,
            path,
            io.BytesIO(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_size=len(content),
        )

    async def put_stream(
        self,
        ctx: object | None,
        key: str | Path,
        stream: BinaryIO,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> dict[str, object]:
        del ctx
        target = self._resolve_target(key)
        part_path = self._part_path(target)
        hasher = hashlib.sha256()
        size = 0
        try:
            with part_path.open("wb") as handle:
                while True:
                    chunk = stream.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)

            actual_sha256 = hasher.hexdigest()
            if expected_size is not None and size != expected_size:
                raise ValueError(
                    f"size mismatch for {target}: expected {expected_size}, got {size}"
                )
            if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
                raise ValueError(
                    f"sha256 mismatch for {target}: expected {expected_sha256}, got {actual_sha256}"
                )

            os.replace(part_path, target)
            return {"key": str(key), "sha256": actual_sha256, "size": size}
        except Exception:
            if part_path.exists():
                part_path.unlink()
            raise


__all__ = ["LocalFileStoragePlugin"]
