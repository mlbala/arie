"""Answer-path model plugins (entailment, reranker), driven by one table of slots.

Each slot in ``ANSWER_SLOTS`` is read from ``plugins.<slot>`` in arie.yaml, warmed at
startup, retried while it fails to load, and reported in the storage and health pane.
By default every slot shares one worker process (``runtime_qos.shared_model_worker``),
so the Python and ONNX Runtime overhead is paid once. Adding a local model slot is one
``AnswerSlot`` entry here plus the harness hook that consumes its scores.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from arie_core.registry.manager import ArieConfig
from arie_worker.models import LocalModel, ModelPool

logger = logging.getLogger(__name__)

Document = dict[str, Any]


@dataclass(frozen=True)
class AnswerSlot:
    """How one answer-path model slot is configured, warmed and described."""

    slot: str
    method: str
    warmup: tuple[object, ...]
    # Keys under plugins.<slot> that ARIE reads itself instead of passing to the plugin.
    runtime_keys: tuple[str, ...]
    healthy: str
    disabled: str
    # What answers do while the model is loading or unavailable.
    fallback: str


ANSWER_SLOTS: tuple[AnswerSlot, ...] = (
    AnswerSlot(
        slot="entailment",
        method="score",
        warmup=([("ARIE warm-up premise.", "Warm-up.")],),
        runtime_keys=("min_score",),
        healthy="Every summary claim is checked for entailment by its own quote.",
        disabled="Tier 3 entailment is off; summaries rest on quote matching "
        "and the model reviewer.",
        fallback="answers use exact source excerpts",
    ),
    AnswerSlot(
        slot="reranker",
        method="score",
        warmup=("ARIE warm-up question", ["Warm-up passage."]),
        runtime_keys=(),
        healthy="Passages are reranked by relevance to each question.",
        disabled="Reranking is off; passages are chosen by keyword and vector search.",
        fallback="passages use keyword and vector order",
    ),
)

RETRY_SEC = 300.0


class AnswerModel:
    """One configured answer-path model: readiness, warm-up and status."""

    def __init__(self, spec: AnswerSlot, model: LocalModel, options: Mapping[str, object]):
        self.spec = spec
        self.model = model
        self.options = dict(options)
        self.ready = False
        self.error = ""

    async def warm(self) -> None:
        """Load the model so the first answer need not download it; retry every 5 minutes."""
        while not self.ready:
            try:
                await self.model.call(self.spec.method, *self.spec.warmup, kill_after=600.0)
                self.ready, self.error = True, ""
            except Exception as exc:
                logger.warning(
                    "%s plugin %s failed to load", self.spec.slot, self.model.name, exc_info=True
                )
                self.error = str(exc) or type(exc).__name__
                await asyncio.sleep(RETRY_SEC)

    async def scores(self, *args: object, count: int) -> list[float]:
        """One score per input; raises while the model is unavailable."""
        if not self.ready:
            raise RuntimeError(f"The {self.spec.slot} model is not ready")
        return await self.model.scores(self.spec.method, *args, count=count)

    def status(self) -> Document:
        status = "healthy" if self.ready else "degraded" if self.error else "loading"
        detail = {
            "healthy": self.spec.healthy,
            "loading": f"Loading {self.model.name}; {self.spec.fallback} until then.",
            "degraded": f"{self.model.name} unavailable ({self.error}); {self.spec.fallback}.",
        }[status]
        return {"plugin": self.model.name, "status": status, "detail": detail}


class AnswerModels:
    """Every configured answer-path model, sharing one worker unless configured apart."""

    def __init__(self, root: Path, config: ArieConfig) -> None:
        shared = config.runtime_qos.shared_model_worker
        shared_pool = ModelPool(root, "Answer models") if shared else None
        self.pools: list[ModelPool] = [shared_pool] if shared_pool else []
        self.models: dict[str, AnswerModel] = {}
        for spec in ANSWER_SLOTS:
            settings = dict(cast(dict[str, object], config.plugins.get(spec.slot, {})))
            name = str(settings.pop("use", "") or "none")
            if name == "none":
                continue
            options = {key: settings.pop(key) for key in spec.runtime_keys if key in settings}
            pool = shared_pool
            if pool is None:
                pool = ModelPool(root, spec.slot.capitalize())
                self.pools.append(pool)
            self.models[spec.slot] = AnswerModel(
                spec, LocalModel(pool, spec.slot, name, settings), options
            )

    def get(self, slot: str) -> AnswerModel | None:
        return self.models.get(slot)

    def option(self, slot: str, key: str, default: float) -> float:
        model = self.models.get(slot)
        return float(cast(float, model.options.get(key, default))) if model else default

    async def warm(self) -> None:
        """Warm every model concurrently; each keeps retrying until it loads."""
        await asyncio.gather(*(model.warm() for model in self.models.values()))

    def status(self) -> dict[str, Document]:
        """Per-slot status, including disabled slots, keyed by slot name."""
        return {
            spec.slot: (
                self.models[spec.slot].status()
                if spec.slot in self.models
                else {"plugin": "none", "status": "disabled", "detail": spec.disabled}
            )
            for spec in ANSWER_SLOTS
        }

    def close(self) -> None:
        for pool in self.pools:
            pool.close()


__all__ = ["ANSWER_SLOTS", "AnswerModel", "AnswerModels", "AnswerSlot"]
