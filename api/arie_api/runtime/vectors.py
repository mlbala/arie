"""The vector index: embedder choice, background indexing and semantic search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from arie_core.contracts import VectorRow
from arie_core.plugin_api import (
    VectorStore,
)
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import months_ago as months_ago
from arie_worker.models import LocalModel, ModelPool
from arie_worker.pool import BACKGROUND_NICE

from arie_api.providers import (
    AUTO_EMBEDDER,
    embedding_key_configured,
    resolve_embedder,
    save_plugin_selection,
)
from arie_api.runtime.base import (
    EMBED_BATCH,
    CallContext,
    Document,
    context,
    logger,
)
from arie_api.runtime.packs import PaperPacks


class VectorIndex(PaperPacks):
    """The vector index: embedder choice, background indexing and semantic search."""

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        super().__init__(root, config)
        vector_name = str(
            cast(dict[str, object], self.config.plugins.get("vector_store", {})).get("use")
            or "none"
        )
        if vector_name != "none" and resolve_embedder(self.config) is not None:
            self.vector_store = cast(VectorStore, self.instance("vector_store"))
            self.apply_embedder()

    def _embedder(self, name: str, settings: dict[str, object]) -> LocalModel:
        """An embedder in its own background-priority worker, so indexing yields the CPU."""
        pool = ModelPool(self.root, "Embedding", nice=BACKGROUND_NICE)
        return LocalModel(pool, "embedder", name, settings)

    def apply_embedder(self) -> None:
        """Adopt the embedder resolved from config; a new model re-embeds in the background.

        Until a paper is re-embedded, its old vectors no longer match ``embedder_id`` and
        semantic search returns nothing for it, so answers fall back to keyword retrieval.
        """
        choice = resolve_embedder(self.config)
        if self.vector_store is None or choice is None:
            return
        model = choice.settings.get("model")
        embedder_id = f"{choice.name}:{model}" if model else choice.name
        self.embedder_reason = choice.reason
        if embedder_id == self.embedder_id and self.embedding is not None:
            return
        previous = self.embedding
        self.embedding = self._embedder(choice.name, choice.settings)
        self.embedder_id, self.min_score = embedder_id, choice.min_score
        self.vector_errors.clear()
        if previous is not None:
            previous.close()
        self.vector_wakeup.set()

    async def check_embedder(self, name: str) -> None:
        """Embed one probe text with ``name`` in a disposable worker; raise ``ValueError``."""
        if name not in {AUTO_EMBEDDER, *self.manager.get_available_plugins("embedder")}:
            raise ValueError("Unknown embedder")
        candidate = self.config.model_copy(deep=True)
        candidate.plugins["embedder"] = {"use": name}
        choice = resolve_embedder(candidate)
        if choice is None:
            raise ValueError("Unknown embedder")
        profile = self.config.embedding_providers.get(choice.name)
        if (
            profile
            and profile.api_key_env
            and not embedding_key_configured(choice.name, self.config)
        ):
            raise ValueError(
                f"{profile.api_key_env} is not configured. Add it to .env and restart ARIE."
            )
        pool = self._embedder(choice.name, choice.settings)
        try:
            # First use may download a local model, so allow more than a chat-sized wait.
            vectors = await pool.embed(["ARIE embedder connection check"], kill_after=120.0)
        except Exception as exc:
            raise ValueError(f"{choice.name} embedding check failed: {exc}") from None
        finally:
            pool.close()
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError(f"{choice.name} returned no embedding")

    async def switch_embedder(self, name: str) -> Document:
        """Verify, save to arie.yaml, then adopt; the library re-embeds in the background."""
        if self.vector_store is None:
            raise ValueError("Vector search is off; set plugins.vector_store in arie.yaml")
        await self.check_embedder(name)
        selection: dict[str, object] = {"use": name}
        async with self.lock:
            await asyncio.to_thread(save_plugin_selection, self.root, "embedder", selection)
            self.config.plugins["embedder"] = selection
            self.apply_embedder()
        return {"slot": "embedder", "plugin": name, "embedder": self.embedder_id}

    async def _vector_backlog(self, ctx: CallContext) -> tuple[list[str], int]:
        """Return chat-ready papers lacking current vectors, and the chat-ready total."""
        rows = await self.db.fetch(
            ctx,
            """SELECT l.paper_id, l.summary_json, v.model AS vector_model,
            v.revision AS vector_revision FROM arie_library l
            LEFT JOIN arie_vector_index v ON v.paper_id = l.paper_id
            WHERE l.status = 'INDEXED' ORDER BY l.published_at DESC""",
        )
        pending: list[str] = []
        for row in rows:
            summary = cast(Document, json.loads(str(row["summary_json"])))
            revision = f"{summary.get('revision', '')}:{summary.get('chunk_count', 0)}"
            if (row["vector_model"], row["vector_revision"]) != (self.embedder_id, revision):
                pending.append(str(row["paper_id"]))
        return pending, len(rows)

    async def ensure_paper_vectors(self, paper_id: str) -> None:
        """Index one paper's vectors unless they are already current for this embedder."""
        pending, _ = await self._vector_backlog(context("vector-backlog", 10.0))
        if paper_id in pending:
            await self.index_paper_vectors(paper_id)

    async def index_paper_vectors(self, paper_id: str) -> None:
        """Replace one paper's vectors in batches, then record the embedder and revision.

        The marker is written last, so an interrupted run is retried by the next backlog scan.
        """
        # Capture the embedder: a switch mid-run must not tag old-model vectors as new.
        embedding, embedder_id = self.embedding, self.embedder_id
        if self.vector_store is None or embedding is None:
            return
        paper = await self.get_paper(paper_id)
        if paper is None or paper["status"] != "INDEXED":
            return
        chunks: list[Document] = [
            chunk
            for chunk in paper.get("chunks", [])
            if chunk.get("paper_id") == paper_id
            and chunk.get("chunk_id")
            and str(chunk.get("text", "")).strip()
        ]
        ctx = context(f"vectors:{paper_id}", 600.0)
        await self.vector_store.delete(ctx, filters={"paper_id": paper_id})
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = await embedding.embed([str(chunk["text"]) for chunk in batch])
            await self.vector_store.upsert(
                ctx,
                [
                    VectorRow(
                        id=str(chunk["chunk_id"]),
                        vector=vector,
                        metadata={
                            "paper_id": paper_id,
                            "model": embedder_id,
                            "page": str(chunk.get("page_start", "")),
                        },
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
        await self.db.execute(
            ctx,
            """INSERT INTO arie_vector_index (paper_id, model, revision, vectors, updated_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(paper_id) DO UPDATE SET model=excluded.model,
            revision=excluded.revision, vectors=excluded.vectors, updated_at=excluded.updated_at""",
            (
                paper_id,
                embedder_id,
                f"{paper.get('dedup_hash', '')}:{len(paper.get('chunks', []))}",
                len(chunks),
                datetime.now(UTC).isoformat(),
            ),
        )

    async def run_vector_index(self) -> None:
        """Keep vectors in step with the library: backfill, re-embed on model change, retry.

        Wakes at startup, whenever sync indexes a paper, and every 5 minutes while any paper
        failed. A failing paper stays keyword-searchable; its error is shown in the monitor.
        """
        if self.vector_store is None or self.embedding is None:
            return
        warmed = ""
        while True:
            self.vector_wakeup.clear()
            pending, _ = await self._vector_backlog(context("vector-backlog", 10.0))
            try:
                if pending and warmed != self.embedder_id:
                    # The first load may download the model, which can exceed a batch timeout.
                    await self.embedding.embed(["warm-up"], kill_after=600.0)
                    warmed = self.embedder_id
                    self.vector_errors.pop("*", None)
                for paper_id in pending:
                    try:
                        await self.index_paper_vectors(paper_id)
                        self.vector_errors.pop(paper_id, None)
                    except Exception as exc:
                        logger.warning("Vector indexing failed for %s", paper_id, exc_info=True)
                        self.vector_errors[paper_id] = str(exc) or type(exc).__name__
            except Exception as exc:
                logger.warning("Embedder %s failed to load", self.embedder_id, exc_info=True)
                self.vector_errors["*"] = str(exc) or type(exc).__name__
            try:
                timeout = 300.0 if self.vector_errors else None
                await asyncio.wait_for(self.vector_wakeup.wait(), timeout)
            except TimeoutError:
                pass

    async def semantic_search(self, paper_id: str, queries: Sequence[str]) -> list[str]:
        """Rank one paper's chunk IDs by best cosine similarity to any query, best first.

        Returns [] when vectors are off or the paper has no vectors for this embedder.
        Hits below ``min_score`` are dropped and every hit is re-checked against
        ``paper_id`` (I1). Embedder or store errors propagate to the caller.
        """
        embedding, embedder_id, floor = self.embedding, self.embedder_id, self.min_score
        if self.vector_store is None or embedding is None or not queries:
            return []
        ctx = context(f"semantic:{paper_id}", 6.0)
        marker = await self.db.fetch(
            ctx, "SELECT model FROM arie_vector_index WHERE paper_id = ?", (paper_id,)
        )
        if not marker or marker[0]["model"] != embedder_id:
            return []
        best: dict[str, float] = {}
        for vector in await embedding.embed(list(queries), kill_after=6.0):
            hits = await self.vector_store.search(
                ctx, vector, 8, filters={"paper_id": paper_id, "model": embedder_id}
            )
            for hit in hits:
                if hit.payload.get("paper_id") == paper_id and hit.score >= floor:
                    best[hit.id] = max(best.get(hit.id, hit.score), hit.score)
        return sorted(best, key=best.__getitem__, reverse=True)

    async def vector_status(self) -> Document:
        """Report measured index coverage for the configured embedder and vector store."""
        provider = cast(dict[str, object], self.config.plugins.get("vector_store", {})).get("use")
        status: Document = {
            "provider": provider,
            "embedder": self.embedder_id,
            "embedder_reason": self.embedder_reason,
            "status": "disabled",
            "size_bytes": None,
            "vectors": None,
            "papers_indexed": None,
            "papers_total": None,
            "detail": "Vector search is off; set plugins.embedder and plugins.vector_store "
            "in arie.yaml. Answers use keyword retrieval.",
        }
        if self.vector_store is None:
            return status
        ctx = context("vector-health", 3.0)
        stats = await self.vector_store.stats(ctx)
        pending, total = await self._vector_backlog(ctx)
        failed = [paper_id for paper_id in pending if paper_id in self.vector_errors]
        error = self.vector_errors.get("*") or next(
            (self.vector_errors[paper_id] for paper_id in failed), None
        )
        status.update(
            status="degraded" if error else "indexing" if pending else "healthy",
            size_bytes=stats.size_bytes,
            vectors=stats.vectors,
            papers_indexed=total - len(pending),
            papers_total=total,
            detail=f"{total - len(pending)} of {total} papers embedded with {self.embedder_id}"
            + (f"; {len(failed) or 'all'} failing: {error}" if error else ""),
        )
        return status
