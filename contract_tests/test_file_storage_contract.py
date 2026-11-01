from __future__ import annotations

import os
from pathlib import Path

import pytest


class AtomicFileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def write(self, ctx: object | None, path: str | Path, content: bytes) -> None:
        target = self.root / path if isinstance(path, str) else path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(target.suffix + ".part")
        temp_path.write_bytes(content)
        os.replace(temp_path, target)

    async def write_with_failure(
        self,
        ctx: object | None,
        path: str | Path,
        content: bytes,
    ) -> None:
        target = self.root / path if isinstance(path, str) else path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(target.suffix + ".part")
        try:
            temp_path.write_bytes(content)
            raise RuntimeError("simulate file write failure")
        finally:
            if temp_path.exists():
                temp_path.unlink()


@pytest.mark.asyncio
async def test_file_storage_contract_atomic_write_and_rollback(tmp_path: Path) -> None:
    storage = AtomicFileStorage(tmp_path)
    target = tmp_path / "papers" / "draft.txt"

    await storage.write(None, target, b"hello")
    assert target.read_bytes() == b"hello"

    with pytest.raises(RuntimeError, match="simulate file write failure"):
        await storage.write_with_failure(None, tmp_path / "papers" / "broken.txt", b"partial")

    assert not (tmp_path / "papers" / "broken.txt").exists()
    assert not (tmp_path / "papers" / "broken.txt.part").exists()
