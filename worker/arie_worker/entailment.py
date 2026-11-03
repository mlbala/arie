from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import Entailment
from arie_core.registry.manager import RegistryManager

from arie_worker.pool import SupervisedPool, WorkerContext

# One model per worker process, so it loads once per process rather than per question.
_MODELS: dict[tuple[str, str], Entailment] = {}


def _score(
    project_root: str, name: str, settings: str, pairs: list[tuple[str, str]]
) -> list[float]:
    """Score (premise, hypothesis) pairs with the configured plugin in a worker process."""
    os.environ["ARIE_ENTAILMENT_WORKER"] = "1"
    key = (name, settings)
    model = _MODELS.get(key)
    if model is None:
        manager = RegistryManager(project_root=project_root)
        model = cast(
            Entailment, manager.instantiate_plugin("entailment", name, **json.loads(settings))
        )
        _MODELS[key] = model
    ctx = WorkerContext(time.monotonic() + 60.0, f"entail:{name}", BudgetHandle())
    scores = asyncio.run(model.score(ctx, pairs))
    if len(scores) != len(pairs):
        raise ValueError(
            f"Entailment model {name!r} returned {len(scores)} scores for {len(pairs)}"
        )
    return [float(score) for score in scores]


class EntailmentPool(SupervisedPool):
    """Run the Tier 3 entailment model in one recycled worker process, never in arie-api.

    Calls are serialized; an answer sends at most a handful of short pairs. The worker
    keeps normal priority because every answer waits on it. Failure
    modes: model load or inference errors and timeouts propagate (the caller abstains),
    and the pool is rebuilt on the next call.
    """

    def __init__(self, project_root: Path, name: str, settings: dict[str, object]) -> None:
        super().__init__(1, "Entailment")
        self.root = project_root
        self.name = name
        self.settings = json.dumps(settings, sort_keys=True)

    async def score(self, pairs: list[tuple[str, str]], kill_after: float = 10.0) -> list[float]:
        if not pairs:
            return []
        return await self.submit(
            kill_after, _score, str(self.root), self.name, self.settings, pairs
        )


__all__ = ["EntailmentPool"]
