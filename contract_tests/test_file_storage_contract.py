"""Slot contract for every discovered ``file_storage`` plugin (I4 atomic commits).

Writes commit through ``<name>.part`` and ``os.replace``; a failed or mismatched write
leaves neither the target nor the part file, and keys cannot escape the storage root.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Protocol, cast

import pytest
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
STORES = RegistryManager(project_root=ROOT).get_available_plugins("file_storage")


class StreamingStorage(Protocol):
    async def read(self, ctx: object | None, path: str) -> bytes: ...
    async def write(self, ctx: object | None, path: str, content: bytes) -> None: ...
    async def put_stream(
        self,
        ctx: object | None,
        key: str,
        stream: io.BytesIO,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> dict[str, object]: ...


@pytest.fixture(params=STORES)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> StreamingStorage:
    manager = RegistryManager(project_root=ROOT)
    return cast(
        StreamingStorage,
        manager.build_plugin("file_storage", str(request.param), root=tmp_path / "files"),
    )


def test_builtin_file_stores_are_discovered() -> None:
    assert {"local", "s3"} <= set(STORES)


@pytest.mark.asyncio
async def test_write_round_trips_and_leaves_no_part_file(
    store: StreamingStorage, tmp_path: Path
) -> None:
    await store.write(None, "2026/10/paper.pdf", b"%PDF-1.7 content")
    assert await store.read(None, "2026/10/paper.pdf") == b"%PDF-1.7 content"
    assert not list((tmp_path / "files").rglob("*.part"))


@pytest.mark.asyncio
async def test_stream_mismatch_commits_nothing(store: StreamingStorage, tmp_path: Path) -> None:
    content = b"%PDF-" + b"x" * (200 * 1024)
    receipt = await store.put_stream(
        None,
        "ok.pdf",
        io.BytesIO(content),
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size=len(content),
    )
    assert receipt["size"] == len(content)
    with pytest.raises(ValueError, match="size mismatch"):
        await store.put_stream(None, "bad.pdf", io.BytesIO(content), expected_size=1)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        await store.put_stream(None, "bad.pdf", io.BytesIO(content), expected_sha256="0" * 64)
    files = tmp_path / "files"
    assert not (files / "bad.pdf").exists() and not (files / "bad.pdf.part").exists()


@pytest.mark.parametrize("key", ["../escape.pdf", "a/../../escape.pdf", "  "])
@pytest.mark.asyncio
async def test_keys_cannot_escape_the_storage_root(store: StreamingStorage, key: str) -> None:
    with pytest.raises(ValueError):
        await store.write(None, key, b"%PDF-")
