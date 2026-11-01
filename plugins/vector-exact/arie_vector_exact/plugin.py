from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from arie_core.contracts import VectorRow
from arie_core.plugin_api import plugin


@plugin(slot="vector_store", name="exact")
class ExactVectorStorePlugin:
    """Exact in-memory matrix-vector search scoped to a single paper_id."""

    def __init__(self, db_path: str | Path = "data/arie.sqlite3") -> None:
        self.db_path = Path(db_path)
        self._cache: dict[str, tuple[list[VectorRow], np.ndarray]] = {}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

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
                vector_id = str(row.id)
                payload = [float(x) for x in row.vector]
                metadata = row.metadata
                paper_id = str(metadata.get("paper_id") or "")
                model = str(metadata.get("model") or "default")
                metadata_json = json.dumps(
                    {str(key): str(value) for key, value in metadata.items()}
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
                    (vector_id, paper_id, model, json.dumps(payload), metadata_json),
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
        if scores.size == 0:
            return []
        ranked_indices = np.argsort(scores)[::-1][: max(0, int(k))]
        return [rows[int(index)] for index in ranked_indices.tolist()]


__all__ = ["ExactVectorStorePlugin"]
