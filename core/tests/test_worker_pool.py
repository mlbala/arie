from __future__ import annotations

import os

import pytest
from arie_worker.pool import SupervisedPool


@pytest.mark.asyncio
async def test_pool_recycles_a_worker_per_task_without_racing_the_watchdog() -> None:
    """One task per worker (as Docling needs) replaces processes while RSS is watched."""
    pool = SupervisedPool(1, "Test", tasks_per_child=1)
    try:
        pids = [await pool.submit(60.0, os.getpid) for _ in range(5)]
    finally:
        pool.close()
    assert len(set(pids)) == 5 and os.getpid() not in pids


@pytest.mark.asyncio
async def test_pool_reuses_workers_by_default() -> None:
    pool = SupervisedPool(1, "Test")
    try:
        pids = {await pool.submit(60.0, os.getpid) for _ in range(3)}
    finally:
        pool.close()
    assert len(pids) == 1
