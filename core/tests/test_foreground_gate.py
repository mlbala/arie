"""The foreground gate: background work waits for questions in flight (S16, ADR 0012)."""

from __future__ import annotations

import asyncio

import pytest
from arie_core.registry.manager import RuntimeQoSConfig
from arie_core.sched import ForegroundGate


async def settle() -> None:
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_background_wait_returns_at_once_while_no_question_runs() -> None:
    gate = ForegroundGate()
    await asyncio.wait_for(gate.wait_quiet(), 1.0)
    assert gate.active == 0


@pytest.mark.asyncio
async def test_a_background_wait_is_delayed_while_a_question_runs_and_released_after() -> None:
    gate = ForegroundGate()
    started: list[str] = []

    async def background() -> None:
        await gate.wait_quiet()
        started.append("parse")

    with gate.hold():
        task = asyncio.create_task(background())
        await settle()
        assert started == [] and not task.done()
    await asyncio.wait_for(task, 1.0)
    assert started == ["parse"]


@pytest.mark.asyncio
async def test_the_gate_stays_closed_until_every_question_has_finished() -> None:
    gate = ForegroundGate()
    first, second = gate.hold(), gate.hold()
    first.__enter__()
    second.__enter__()
    task = asyncio.create_task(gate.wait_quiet())
    first.__exit__(None, None, None)
    await settle()
    assert gate.active == 1 and not task.done()
    second.__exit__(None, None, None)
    await asyncio.wait_for(task, 1.0)


@pytest.mark.asyncio
async def test_a_question_that_fails_still_releases_the_gate() -> None:
    gate = ForegroundGate()
    with pytest.raises(RuntimeError), gate.hold():
        raise RuntimeError("the answer failed")
    assert gate.active == 0
    await asyncio.wait_for(gate.wait_quiet(), 1.0)


@pytest.mark.asyncio
async def test_a_question_that_starts_before_a_waiter_resumes_holds_it_again() -> None:
    gate = ForegroundGate()
    with gate.hold():
        task = asyncio.create_task(gate.wait_quiet())
        await settle()
    # The gate opened, but the next question takes it before the waiter runs.
    with gate.hold():
        await settle()
        assert not task.done()
    await asyncio.wait_for(task, 1.0)


@pytest.mark.parametrize(
    ("setting", "cpus", "expected"),
    [(None, 2, False), (None, 1, False), (None, 3, True), (None, 8, True),
     (True, 2, True), (False, 8, False)],
)
def test_the_answer_models_share_a_worker_by_default_only_above_two_cpus(
    setting: bool | None, cpus: int, expected: bool
) -> None:
    assert RuntimeQoSConfig(shared_model_worker=setting).shares_model_worker(cpus) is expected


def test_without_a_cpu_count_the_host_counts_as_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("arie_core.registry.manager.os.process_cpu_count", lambda: None)
    assert RuntimeQoSConfig().shares_model_worker() is False
    monkeypatch.setattr("arie_core.registry.manager.os.process_cpu_count", lambda: 4)
    assert RuntimeQoSConfig().shares_model_worker() is True
