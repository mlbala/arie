from __future__ import annotations

import pytest
from arie_core.contracts import VectorRow


class FakeVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, list[VectorRow]] = {}

    async def upsert(self, ctx: object | None, rows: list[VectorRow]) -> None:
        if any(row.metadata.get("paper_id") == "paper-3" for row in rows):
            raise RuntimeError("rollback vector write")
        for row in rows:
            bucket = self._rows.setdefault(row.id, [])
            bucket.append(row)

    async def search(
        self,
        ctx: object | None,
        query: list[float],
        k: int,
        *,
        filters: dict[str, str] | None = None,
    ) -> list[VectorRow]:
        if filters is None:
            return [row for rows in self._rows.values() for row in rows][:k]
        paper_id = filters.get("paper_id")
        if paper_id is None:
            return []
        results: list[VectorRow] = []
        for rows in self._rows.values():
            for row in rows:
                if row.metadata.get("paper_id") == paper_id:
                    results.append(row)
        return results[:k]


@pytest.mark.asyncio
async def test_vector_store_contract_filters_paper_ids() -> None:
    store = FakeVectorStore()
    await store.upsert(
        None,
        [
            VectorRow(
                id="vec-1",
                vector=[0.1, 0.2],
                metadata={"paper_id": "paper-1", "model": "mini"},
            ),
            VectorRow(
                id="vec-2",
                vector=[0.9, 0.8],
                metadata={"paper_id": "paper-2", "model": "mini"},
            ),
        ],
    )

    results = await store.search(None, [0.1, 0.2], 10, filters={"paper_id": "paper-1"})
    assert len(results) == 1
    assert results[0].metadata["paper_id"] == "paper-1"


@pytest.mark.asyncio
async def test_vector_store_contract_rollback_on_partial_failure() -> None:
    store = FakeVectorStore()

    async def failing_upsert() -> None:
        await store.upsert(
            None,
            [
                VectorRow(
                    id="vec-3",
                    vector=[1.0, 1.0],
                    metadata={"paper_id": "paper-3", "model": "mini"},
                )
            ],
        )

    with pytest.raises(RuntimeError, match="rollback vector write"):
        await failing_upsert()

    results = await store.search(None, [1.0, 1.0], 10, filters={"paper_id": "paper-3"})
    assert results == []
