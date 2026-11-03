from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from typing import Any, cast

import httpx
from arie_core.plugin_api import Ctx, configured_api_key, plugin

ENDPOINT = "https://api.openai.com/v1/embeddings"


def retry_delay(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait after HTTP 429: the server's ``Retry-After`` or 2, 4, 8… capped at 20."""
    try:
        return min(20.0, max(0.5, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return min(20.0, 2.0 * 2**attempt)


@plugin(slot="embedder", name="openai")
class OpenAIEmbedder:
    """OpenAI text embeddings over HTTPS, called from arie-worker under the caller's deadline.

    Uses ``OPENAI_API_KEY`` from the server environment; the key never enters arie.yaml.
    Rate limits (HTTP 429) are retried within the deadline. Failure modes: a missing key
    raises ``RuntimeError``; other HTTP errors and timeouts propagate,
    leaving the paper keyword-searchable until the vector indexer retries.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = configured_api_key(
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )

    async def embed(self, ctx: Ctx, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        remaining = ctx.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Embedding deadline has expired")
        async with asyncio.timeout(remaining):
            async with httpx.AsyncClient(timeout=remaining, follow_redirects=False) as client:
                attempt = 0
                while True:
                    response = await client.post(
                        ENDPOINT,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": self.model, "input": list(texts)},
                    )
                    if response.status_code != 429 or attempt == 5:
                        break
                    await asyncio.sleep(retry_delay(response, attempt))
                    attempt += 1
                response.raise_for_status()
                body = cast(dict[str, Any], response.json())
        rows = sorted(cast(list[dict[str, Any]], body["data"]), key=lambda row: int(row["index"]))
        return [[float(value) for value in row["embedding"]] for row in rows]
