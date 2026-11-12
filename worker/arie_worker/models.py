"""One supervised worker pool for every local model plugin (embedder, entailment, reranker, guard).

A model plugin is any slot whose methods take ``(ctx, *args)`` and return plain data. The
pool loads each ``(slot, name, settings)`` plugin once per worker process and calls the
requested method there, so the API process never imports a model runtime. Several slots
may share one pool: their models then live in one process and pay its fixed cost once.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.registry.manager import RegistryManager

from arie_worker.pool import SupervisedPool, WorkerContext

# The environment flag each slot's plugins check to refuse running outside a worker.
# A slot not listed here uses ``ARIE_<SLOT>_WORKER``.
WORKER_FLAGS: Mapping[str, str] = {
    "embedder": "ARIE_EMBED_WORKER",
    "entailment": "ARIE_ENTAILMENT_WORKER",
    "reranker": "ARIE_RERANK_WORKER",
    "guard": "ARIE_GUARD_WORKER",
}

# Plugins loaded in this worker process, keyed by (slot, name, settings JSON).
_MODELS: dict[tuple[str, str, str], object] = {}


def _as_list(value: object) -> list[object] | None:
    return cast(list[object], value) if isinstance(value, list) else None


def worker_flag(slot: str) -> str:
    return WORKER_FLAGS.get(slot, f"ARIE_{slot.upper()}_WORKER")


def _call(
    project_root: str, slot: str, name: str, settings: str, method: str, args: tuple[object, ...]
) -> object:
    """Run ``plugin.method(ctx, *args)`` for the configured plugin inside a worker."""
    os.environ[worker_flag(slot)] = "1"
    key = (slot, name, settings)
    model = _MODELS.get(key)
    if model is None:
        manager = RegistryManager(project_root=project_root)
        model = manager.instantiate_plugin(slot, name, **json.loads(settings))
        _MODELS[key] = model
    ctx = WorkerContext(time.monotonic() + 60.0, f"{slot}:{name}", BudgetHandle())
    return asyncio.run(getattr(model, method)(ctx, *args))


class ModelPool(SupervisedPool):
    """Serve local model plugins from one recycled worker process, never from arie-api.

    Calls are serialized, so slots sharing a pool wait for each other; answer-path models
    (entailment, reranker) run at different stages of an answer, so they rarely contend.
    A caller that stops waiting leaves the call to finish, so the loaded models survive.
    Running past ``kill_after``, a crash or a memory breach kills the process and every
    model in it; the next call rebuilds the pool and reloads lazily. ``nice`` > 0 suits
    background work.
    """

    def __init__(self, project_root: Path, label: str, nice: int = 0) -> None:
        super().__init__(1, label, nice=nice, keep_on_cancel=True)
        self.root = project_root

    async def call(
        self,
        slot: str,
        name: str,
        settings: Mapping[str, object],
        method: str,
        *args: object,
        kill_after: float = 10.0,
    ) -> object:
        encoded = json.dumps(dict(settings), sort_keys=True)
        return await self.submit(
            kill_after, _call, str(self.root), slot, name, encoded, method, args
        )


class LocalModel:
    """One configured model plugin, called through a (possibly shared) ``ModelPool``.

    The typed helpers check what the plugin returned, because a wrong-length result
    would silently misalign scores or vectors with their inputs.
    """

    def __init__(
        self, pool: ModelPool, slot: str, name: str, settings: Mapping[str, object]
    ) -> None:
        self.pool = pool
        self.slot = slot
        self.name = name
        self.settings = dict(settings)

    async def call(self, method: str, *args: object, kill_after: float = 10.0) -> object:
        return await self.pool.call(
            self.slot, self.name, self.settings, method, *args, kill_after=kill_after
        )

    async def scores(
        self, method: str, *args: object, count: int, kill_after: float = 10.0
    ) -> list[float]:
        """Call a method returning one float per input; raise ``ValueError`` otherwise."""
        if count == 0:
            return []
        result = _as_list(await self.call(method, *args, kill_after=kill_after))
        if result is None or len(result) != count:
            size = "no list of" if result is None else len(result)
            raise ValueError(f"{self.slot} {self.name!r} returned {size} scores for {count}")
        return [float(cast(float, value)) for value in result]

    async def embed(self, texts: list[str], kill_after: float = 60.0) -> list[list[float]]:
        """Embed ``texts`` with an embedder plugin; raise ``ValueError`` on a count mismatch."""
        if not texts:
            return []
        result = _as_list(await self.call("embed", texts, kill_after=kill_after))
        if result is None or len(result) != len(texts):
            size = "no list of" if result is None else len(result)
            raise ValueError(f"Embedder {self.name!r} returned {size} vectors for {len(texts)}")
        return [[float(value) for value in cast(list[float], vector)] for vector in result]

    def close(self) -> None:
        """Stop the pool; only for a model that owns its pool (not a shared one)."""
        self.pool.close()


__all__ = ["WORKER_FLAGS", "LocalModel", "ModelPool", "worker_flag"]
