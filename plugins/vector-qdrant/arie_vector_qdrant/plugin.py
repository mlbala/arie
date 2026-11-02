from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from arie_core.contracts import VectorRow
from arie_core.plugin_api import plugin


@plugin(slot="vector_store", name="qdrant")
class QdrantVectorStorePlugin:
    """Qdrant-compatible vector store adapter using SQLite as local persistence."""

    def __init__(self, db_path: str | Path = "data/arie-qdrant.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[list[VectorRow], np.ndarray]] = {}
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _load_for_paper(self, paper_id: str) -> tuple[list[VectorRow], np.ndarray]:
        cached = self._cache.get(paper_id)
        if cached is not None:
            return cached

        rows: list[VectorRow] = []
        vectors: list[np.ndarray] = []
        with self._connect() as conn:
            for row in conn.execute(
                """
                SELECT vector_id, paper_id, model, vector_json, metadata_json
                FROM vectors
                WHERE paper_id = ?
                ORDER BY created_at DESC
                """,
                (paper_id,),
            ):
                vector_values = json.loads(row["vector_json"])
                metadata = json.loads(row["metadata_json"] or "{}")
                vector_array = np.asarray(vector_values, dtype=np.float32)
                rows.append(
                    VectorRow(
                        id=str(row["vector_id"]),
                        vector=vector_array.astype(float).tolist(),
                        metadata={
                            "paper_id": str(row["paper_id"]),
                            **{str(key): str(value) for key, value in metadata.items()},
                        },
                    )
                )
                vectors.append(vector_array)

        if not vectors:
            empty = np.empty((0,), dtype=np.float32)
            self._cache[paper_id] = (rows, empty)
            return rows, empty

        matrix = np.stack(vectors, axis=0).astype(np.float32)
        self._cache[paper_id] = (rows, matrix)
        return rows, matrix

    async def upsert(self, ctx: object | None, rows: Sequence[object]) -> None:
        del ctx
        if not rows:
            return
        with self._connect() as conn:
            for row in rows:
                if not isinstance(row, VectorRow):
                    continue
                paper_id = str(row.metadata.get("paper_id") or "")
                model = str(row.metadata.get("model") or "default")
                metadata_json = json.dumps(
                    {str(key): str(value) for key, value in row.metadata.items()}
                )
                conn.execute(
                    """
                    INSERT INTO vectors (vector_id, paper_id, model, vector_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(vector_id) DO UPDATE SET
                        paper_id = excluded.paper_id,
                        model = excluded.model,
                        vector_json = excluded.vector_json,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        row.id,
                        paper_id,
                        model,
                        json.dumps([float(x) for x in row.vector]),
                        metadata_json,
                    ),
                )
            conn.commit()
        self._cache.clear()

    async def search(
        self,
        ctx: object | None,
        query: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[VectorRow]:
        del ctx
        if not query:
            return []
        paper_id = None if filters is None else str(filters.get("paper_id") or filters.get("paper"))
        if not paper_id:
            return []
        rows, matrix = self._load_for_paper(paper_id)
        if matrix.size == 0:
            return []
        query_array = np.asarray(query, dtype=np.float32)
        scores = matrix @ query_array
        ranked_indices = np.argsort(scores)[::-1][: max(0, int(k))]
        return [rows[int(index)] for index in ranked_indices.tolist()]


__all__ = ["QdrantVectorStorePlugin"]
