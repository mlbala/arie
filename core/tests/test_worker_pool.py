from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from arie_worker.extraction import ExtractionLimits, ExtractionPool
from arie_worker.pool import BACKGROUND_NICE, DEFAULT_MAX_RSS_MB, SupervisedPool
from pydantic import ValidationError


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


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [True, False])
async def test_a_caller_giving_up_keeps_the_worker_only_when_asked(keep: bool) -> None:
    """A model pool keeps its loaded worker when the caller's own deadline expires."""
    pool = SupervisedPool(1, "Test", keep_on_cancel=keep)
    try:
        first = await pool.submit(30.0, os.getpid)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await pool.submit(30.0, time.sleep, 0.6)
        # The next call queues behind the abandoned one; its kill_after covers the wait.
        assert (await pool.submit(30.0, os.getpid) == first) is keep
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_a_pool_counts_the_calls_its_callers_gave_up_on_until_they_finish() -> None:
    pool = SupervisedPool(1, "Test", keep_on_cancel=True)
    try:
        await pool.submit(30.0, os.getpid)
        assert pool.abandoned_calls == 0
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await pool.submit(30.0, time.sleep, 0.6)
        assert pool.abandoned_calls == 1  # still running in the worker
        await pool.submit(30.0, os.getpid)  # queues behind it, so it has finished by now
        await asyncio.sleep(0.05)
        assert pool.abandoned_calls == 0
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_a_pool_that_kills_on_cancel_has_no_abandoned_calls() -> None:
    pool = SupervisedPool(1, "Test")
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await pool.submit(30.0, time.sleep, 5)
        assert pool.abandoned_calls == 0
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_a_call_past_kill_after_still_kills_a_keep_on_cancel_pool() -> None:
    pool = SupervisedPool(1, "Test", keep_on_cancel=True)
    try:
        first = await pool.submit(30.0, os.getpid)
        with pytest.raises(TimeoutError):
            await pool.submit(0.2, time.sleep, 5)
        assert await pool.submit(30.0, os.getpid) != first
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_pool_kills_a_worker_over_its_own_memory_limit() -> None:
    """Any interpreter exceeds 1 MB, so the watchdog must kill it and report the limit."""
    pool = SupervisedPool(1, "Test", max_rss_mb=1)
    try:
        with pytest.raises(MemoryError, match="1 MB memory limit"):
            await pool.submit(30.0, time.sleep, 10)
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_background_pool_workers_run_at_lower_cpu_priority() -> None:
    """``os.nice(0)`` reports the worker's niceness without changing it."""
    background, interactive = (
        SupervisedPool(1, "Background", nice=BACKGROUND_NICE),
        SupervisedPool(1, "Interactive"),
    )
    try:
        base = os.nice(0)
        assert await background.submit(30.0, os.nice, 0) == min(19, base + BACKGROUND_NICE)
        assert await interactive.submit(30.0, os.nice, 0) == base
    finally:
        background.close()
        interactive.close()


def test_extraction_pool_never_starts_more_workers_than_cpus() -> None:
    pool = ExtractionPool(Path("."), "pymupdf4llm", processes=64)
    assert pool.processes == (os.process_cpu_count() or 1)
    assert pool.nice == BACKGROUND_NICE


def test_extraction_limits_default_to_the_shared_caps() -> None:
    pool = ExtractionPool(Path("."), "pymupdf4llm")
    assert (pool.timeout_sec, pool.tasks_per_child, pool.max_rss_mb) == (
        60.0,
        20,
        DEFAULT_MAX_RSS_MB,
    )


def test_extraction_limits_leave_parser_options_for_the_constructor() -> None:
    config: dict[str, object] = {
        "timeout_sec": 300,
        "tasks_per_worker": 1,
        "max_rss_mb": 2560,
        "do_ocr": True,
    }
    limits = ExtractionLimits.pop_from(config)
    assert (limits.timeout_sec, limits.tasks_per_worker, limits.max_rss_mb) == (300, 1, 2560)
    assert config == {"do_ocr": True}


@pytest.mark.parametrize(
    "config",
    [{"max_rss_mb": 1024}, {"max_rss_mb": 9000}, {"tasks_per_worker": 0}, {"timeout_sec": 0}],
)
def test_extraction_limits_reject_values_outside_their_bounds(config: dict[str, object]) -> None:
    """The memory cap can only be raised, and only to a value a laptop can hold."""
    with pytest.raises(ValidationError):
        ExtractionLimits.pop_from(config)
