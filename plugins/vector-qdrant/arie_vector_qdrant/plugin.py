from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from arie_core.contracts import Hit, VectorRow, VectorStats
from arie_core.plugin_api import Ctx, plugin

Payload = dict[str, str | int | float | bool | None]


@plugin(slot="vector_store", name="qdrant")
class QdrantVectorStorePlugin:
    """Local stand-in for a Qdrant adapter: exact cosine search persisted in SQLite.

    It does not connect to a Qdrant server; replace this file with a client-backed
    implementation of the same ``VectorStore`` contract to use a real deployment.

    Rows are loaded per (paper_id, model) and cached as a normalized matrix, so a search
    is one matrix-vector product. Failure modes: sqlite3 errors propagate; every upsert or
    delete batch is one transaction, so a failed write leaves the store unchanged.
    """

    def __init__(self, db_path: str | Path = "data/arie-qdrant.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str], tuple[list[Hit], np.ndarray]] = {}
        # Bumped by every write; a load that overlaps a write is served but not cached.
        self._generation = 0
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS vectors (
                vector_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, model TEXT NOT NULL,
                dim INTEGER NOT NULL, vector BLOB NOT NULL, metadata_json TEXT NOT NULL)"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS vectors_scope ON vectors(paper_id, model)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _load(self, paper_id: str, model: str) -> tuple[list[Hit], np.ndarray]:
        cached = self._cache.get((paper_id, model))
        if cached is not None:
            return cached
        generation = self._generation
        rows: list[Hit] = []
        vectors: list[np.ndarray] = []
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT vector_id, dim, vector, metadata_json FROM vectors "
                "WHERE paper_id = ? AND model = ? ORDER BY vector_id",
                (paper_id, model),
            ):
                vector = np.frombuffer(row["vector"], dtype=np.float32, count=int(row["dim"]))
                payload: Payload = dict(json.loads(row["metadata_json"]))
                rows.append(Hit(id=str(row["vector_id"]), score=0.0, payload=payload))
                vectors.append(vector)
        matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1.0, norms)
        if generation == self._generation:
            self._cache[(paper_id, model)] = (rows, matrix)
        return rows, matrix

    def _write(self, rows: Sequence[VectorRow]) -> None:
        records: list[tuple[str, str, str, int, bytes, str]] = []
        for row in rows:
            paper_id = row.metadata.get("paper_id", "")
            model = row.metadata.get("model", "")
            if not paper_id or not model:
                raise ValueError(f"Vector {row.id!r} needs paper_id and model metadata")
            vector = np.asarray(row.vector, dtype=np.float32)
            records.append(
                (row.id, paper_id, model, len(vector), vector.tobytes(), json.dumps(row.metadata))
            )
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO vectors (vector_id, paper_id, model, dim, vector, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(vector_id) DO UPDATE SET paper_id = excluded.paper_id,
                model = excluded.model, dim = excluded.dim, vector = excluded.vector,
                metadata_json = excluded.metadata_json""",
                records,
            )
        self._generation += 1
        self._cache.clear()

    def _delete(self, filters: Mapping[str, str]) -> None:
        paper_id = filters.get("paper_id")
        if not paper_id:
            raise ValueError("delete requires a paper_id filter")
        with self._connect() as conn:
            for vector_id, raw in conn.execute(
                "SELECT vector_id, metadata_json FROM vectors WHERE paper_id = ?", (paper_id,)
            ).fetchall():
                metadata = json.loads(raw)
                if all(metadata.get(key) == value for key, value in filters.items()):
                    conn.execute("DELETE FROM vectors WHERE vector_id = ?", (vector_id,))
        self._generation += 1
        self._cache.clear()

    def _search(self, query: Sequence[float], k: int, filters: Mapping[str, str]) -> list[Hit]:
        paper_id, model = filters.get("paper_id"), filters.get("model")
        if not paper_id or not model or k <= 0:
            return []
        rows, matrix = self._load(paper_id, model)
        vector = np.asarray(query, dtype=np.float32)
        if not matrix.size or matrix.shape[1] != vector.shape[0]:
            return []
        norm = float(np.linalg.norm(vector))
        scores = matrix @ (vector / (norm or 1.0))
        hits: list[Hit] = []
        for index in np.argsort(-scores).tolist():
            row = rows[int(index)]
            if all(row.payload.get(key) == value for key, value in filters.items()):
                hits.append(Hit(id=row.id, score=float(scores[index]), payload=dict(row.payload)))
                if len(hits) == k:
                    break
        return hits

    def _stats(self) -> VectorStats:
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        size = sum(
            path.stat().st_size
            for path in (self.db_path, Path(f"{self.db_path}-wal"))
            if path.exists()
        )
        return VectorStats(vectors=count, size_bytes=size)

    async def upsert(self, ctx: Ctx, rows: Sequence[VectorRow]) -> None:
        del ctx
        if rows:
            await asyncio.to_thread(self._write, rows)

    async def search(
        self,
        ctx: Ctx,
        query: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, str],
    ) -> list[Hit]:
        del ctx
        return await asyncio.to_thread(self._search, query, k, filters)

    async def delete(self, ctx: Ctx, *, filters: Mapping[str, str]) -> None:
        del ctx
        await asyncio.to_thread(self._delete, filters)

    async def stats(self, ctx: Ctx) -> VectorStats:
        del ctx
        return await asyncio.to_thread(self._stats)


__all__ = ["QdrantVectorStorePlugin"]
