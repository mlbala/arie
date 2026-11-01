from __future__ import annotations

from collections.abc import Sequence

from arie_core.plugin_api import plugin


@plugin(slot="embedder", name="fastembed")
class FastEmbedPlugin:
    """Placeholder embedding adapter designed for worker-process execution."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = model

    async def embed(self, ctx: object | None, texts: Sequence[str]) -> list[list[float]]:
        del ctx
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for text in texts:
            normalized = text.strip()
            vector = [0.0] * 8
            for index, char in enumerate(normalized[:8]):
                vector[index] = float(ord(char) % 11) / 10.0
            if not normalized:
                vector = [0.0] * 8
            embeddings.append(vector)
        return embeddings
