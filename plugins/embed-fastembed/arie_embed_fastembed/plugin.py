from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from arie_core.plugin_api import Ctx, plugin

Vector = Sequence[float]


@plugin(slot="embedder", name="fastembed")
class FastEmbedPlugin:
    """Local ONNX sentence embeddings through FastEmbed, loaded only in arie-worker.

    The model downloads to ``FASTEMBED_CACHE_PATH`` (or ``cache_dir``) on first use and
    returns L2-normalized vectors. Failure modes: download or ONNX load errors propagate
    from the first ``embed`` call; calls outside a worker process raise ``RuntimeError``.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._encode: Callable[[list[str]], Iterable[Vector]] | None = None

    async def embed(self, ctx: Ctx, texts: Sequence[str]) -> list[list[float]]:
        del ctx
        if os.environ.get("ARIE_EMBED_WORKER") != "1":
            raise RuntimeError("FastEmbed must run inside arie-worker")
        if not texts:
            return []
        if self._encode is None:
            module = importlib.import_module("fastembed")
            engine = module.TextEmbedding(model_name=self.model, cache_dir=self.cache_dir)
            self._encode = cast(Callable[[list[str]], Iterable[Vector]], engine.embed)
        return [[float(value) for value in vector] for vector in self._encode(list(texts))]
