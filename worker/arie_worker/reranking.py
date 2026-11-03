from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import Reranker
from arie_core.registry.manager import RegistryManager

from arie_worker.pool import SupervisedPool, WorkerContext

# One model per worker process, so it loads once per process rather than per question.
_MODELS: dict[tuple[str, str], Reranker] = {}


def _score(
    project_root: str, name: str, settings: str, query: str, passages: list[str]
) -> list[float]:
    """Score passages against one question with the configured plugin in a worker process."""
    os.environ["ARIE_RERANK_WORKER"] = "1"
    key = (name, settings)
    model = _MODELS.get(key)
    if model is None:
        manager = RegistryManager(project_root=project_root)
        model = cast(Reranker, manager.instantiate_plugin("reranker", name, **json.loads(settings)))
        _MODELS[key] = model
    ctx = WorkerContext(time.monotonic() + 60.0, f"rerank:{name}", BudgetHandle())
    scores = asyncio.run(model.score(ctx, query, passages))
    if len(scores) != len(passages):
        raise ValueError(f"Reranker {name!r} returned {len(scores)} scores for {len(passages)}")
    return [float(score) for score in scores]


class RerankPool(SupervisedPool):
    """Run the cross-encoder reranker in one recycled worker process, never in arie-api.

    Uses the default worker limits (20 tasks, 1.5 GB RSS) and keeps normal priority,
    because every answer waits on it (ADR 0007). Calls are serialized; an answer sends
    one question and at most a few dozen passages. Failure modes: model load or
    inference errors and timeouts propagate (the caller keeps its own ranking), and the
    pool is rebuilt on the next call.
    """

    def __init__(self, project_root: Path, name: str, settings: dict[str, object]) -> None:
        super().__init__(1, "Reranker")
        self.root = project_root
        self.name = name
        self.settings = json.dumps(settings, sort_keys=True)

    async def score(self, query: str, passages: list[str], kill_after: float = 10.0) -> list[float]:
        if not passages:
            return []
        return await self.submit(
            kill_after, _score, str(self.root), self.name, self.settings, query, passages
        )


__all__ = ["RerankPool"]
