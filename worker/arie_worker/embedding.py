from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import Embedder
from arie_core.registry.manager import RegistryManager

from arie_worker.pool import SupervisedPool, WorkerContext

# One embedder per worker process, so a model loads once per process rather than per batch.
_EMBEDDERS: dict[tuple[str, str], Embedder] = {}


def _embed(project_root: str, name: str, settings: str, texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with the configured plugin inside a disposable worker process."""
    os.environ["ARIE_EMBED_WORKER"] = "1"
    key = (name, settings)
    embedder = _EMBEDDERS.get(key)
    if embedder is None:
        manager = RegistryManager(project_root=project_root)
        embedder = cast(
            Embedder, manager.instantiate_plugin("embedder", name, **json.loads(settings))
        )
        _EMBEDDERS[key] = embedder
    ctx = WorkerContext(time.monotonic() + 60.0, f"embed:{name}", BudgetHandle())
    vectors = asyncio.run(embedder.embed(ctx, texts))
    if len(vectors) != len(texts):
        raise ValueError(f"Embedder {name!r} returned {len(vectors)} vectors for {len(texts)}")
    return [[float(value) for value in vector] for vector in vectors]


class EmbeddingPool(SupervisedPool):
    """Run the configured embedder in one recycled worker process, never in arie-api.

    Batches are serialized, so callers keep them small (16 texts) to let chat queries
    interleave with background indexing. Failure modes: model load or inference errors
    and timeouts propagate; the pool is rebuilt on the next call.
    """

    def __init__(self, project_root: Path, name: str, settings: dict[str, object]) -> None:
        super().__init__(1, "Embedding")
        self.root = project_root
        self.name = name
        self.settings = json.dumps(settings, sort_keys=True)

    async def embed(self, texts: list[str], kill_after: float = 60.0) -> list[list[float]]:
        if not texts:
            return []
        return await self.submit(
            kill_after, _embed, str(self.root), self.name, self.settings, texts
        )


__all__ = ["EmbeddingPool"]
