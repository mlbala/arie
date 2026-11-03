"""VectorStore slot contract, run against every discovered ``vector_store`` plugin.

A bring-your-own store passes when it is constructible with no arguments (state under
the working directory, which these tests point at a temporary folder) or, for a server
store, from environment settings its tests provide.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from arie_core.budget import BudgetHandle
from arie_core.contracts import VectorRow
from arie_core.plugin_api import VectorStore
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
STORES = RegistryManager(project_root=ROOT).get_available_plugins("vector_store")


@dataclass(frozen=True)
class Context:
    deadline: float
    trace_id: str
    budget: BudgetHandle


def ctx() -> Context:
    return Context(time.monotonic() + 30.0, "vector-contract", BudgetHandle())


def row(vector_id: str, vector: list[float], paper_id: str, model: str = "m1") -> VectorRow:
    return VectorRow(vector_id, vector, {"paper_id": paper_id, "model": model, "page": "1"})


@pytest.fixture(params=STORES)
def store(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manager = RegistryManager(project_root=ROOT)
    return cast(VectorStore, manager.build_plugin("vector_store", str(request.param)))


def test_builtin_stores_are_discovered() -> None:
    assert "exact" in STORES


@pytest.mark.asyncio
async def test_search_is_scoped_ranked_and_echoes_metadata(store: VectorStore) -> None:
    await store.upsert(
        ctx(),
        [
            row("p1:0", [1.0, 0.0, 0.0], "p1"),
            row("p1:1", [0.6, 0.8, 0.0], "p1"),
            row("p1:2", [0.0, 0.0, 1.0], "p1"),
            row("p2:0", [1.0, 0.0, 0.0], "p2"),
            row("p1:old", [1.0, 0.0, 0.0], "p1", model="m0"),
        ],
    )
    hits = await store.search(ctx(), [2.0, 0.0, 0.0], 2, filters={"paper_id": "p1", "model": "m1"})
    assert [hit.id for hit in hits] == ["p1:0", "p1:1"]
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-4)
    assert math.isclose(hits[1].score, 0.6, abs_tol=1e-4)
    assert hits[0].payload == {"paper_id": "p1", "model": "m1", "page": "1"}
    everything = await store.search(
        ctx(), [1.0, 0.0, 0.0], 10, filters={"paper_id": "p1", "model": "m1"}
    )
    assert {hit.id for hit in everything} == {"p1:0", "p1:1", "p1:2"}
    assert (
        await store.search(ctx(), [1.0, 0.0, 0.0], 5, filters={"paper_id": "p3", "model": "m1"})
        == []
    )


@pytest.mark.asyncio
async def test_upsert_replaces_by_id(store: VectorStore) -> None:
    await store.upsert(ctx(), [row("p1:0", [1.0, 0.0], "p1")])
    await store.upsert(ctx(), [row("p1:0", [0.0, 1.0], "p1")])
    hits = await store.search(ctx(), [0.0, 1.0], 5, filters={"paper_id": "p1", "model": "m1"})
    assert [hit.id for hit in hits] == ["p1:0"]
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-4)
    assert (await store.stats(ctx())).vectors == 1


@pytest.mark.asyncio
async def test_rejected_batch_writes_nothing(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        await store.upsert(
            ctx(), [row("p1:0", [1.0, 0.0], "p1"), VectorRow("bad", [1.0, 0.0], {"model": "m1"})]
        )
    assert await store.search(ctx(), [1.0, 0.0], 5, filters={"paper_id": "p1", "model": "m1"}) == []
    assert (await store.stats(ctx())).vectors == 0


@pytest.mark.asyncio
async def test_delete_matches_every_filter(store: VectorStore) -> None:
    await store.upsert(
        ctx(),
        [
            row("p1:0", [1.0, 0.0], "p1"),
            row("p1:old", [1.0, 0.0], "p1", model="m0"),
            row("p2:0", [1.0, 0.0], "p2"),
        ],
    )
    await store.delete(ctx(), filters={"paper_id": "p1", "model": "m0"})
    assert (await store.stats(ctx())).vectors == 2
    await store.delete(ctx(), filters={"paper_id": "p1"})
    assert await store.search(ctx(), [1.0, 0.0], 5, filters={"paper_id": "p1", "model": "m1"}) == []
    remaining = await store.search(ctx(), [1.0, 0.0], 5, filters={"paper_id": "p2", "model": "m1"})
    assert [hit.id for hit in remaining] == ["p2:0"]
    stats = await store.stats(ctx())
    assert stats.vectors == 1
    assert stats.size_bytes is None or stats.size_bytes > 0
