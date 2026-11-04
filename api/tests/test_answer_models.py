"""Answer-path model slots: one table drives config, shared workers, warm-up and status."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from arie_api.models import ANSWER_SLOTS, FAILURE_LIMIT, AnswerModel, AnswerModels
from arie_core.registry.manager import ArieConfig, RuntimeQoSConfig
from arie_worker.models import LocalModel, ModelPool

ROOT = Path(__file__).resolve().parents[2]
ENTAIL = {"use": "nli-onnx", "model": "cross-encoder/nli-deberta-v3-xsmall", "min_score": 0.7}
RERANK = {"use": "onnx-cross-encoder", "model": "cross-encoder/ms-marco-MiniLM-L6-v2"}


def config(shared: bool = True, **plugins: object) -> ArieConfig:
    return ArieConfig(
        runtime_qos=RuntimeQoSConfig(shared_model_worker=shared), plugins=dict(plugins)
    )


def test_slots_share_one_worker_by_default_and_split_when_configured() -> None:
    shared = AnswerModels(ROOT, config(entailment=ENTAIL, reranker=RERANK))
    split = AnswerModels(ROOT, config(False, entailment=ENTAIL, reranker=RERANK))
    try:
        assert len(shared.pools) == 1 and set(shared.models) == {"entailment", "reranker"}
        assert len(split.pools) == 2
        entailment = shared.get("entailment")
        assert entailment is not None
        # ARIE reads min_score itself; the plugin never receives it.
        assert shared.option("entailment", "min_score", 0.5) == 0.7
        assert (
            "min_score" not in entailment.model.settings and "use" not in entailment.model.settings
        )
    finally:
        shared.close()
        split.close()


def test_slots_set_to_none_or_missing_are_reported_disabled() -> None:
    models = AnswerModels(ROOT, config(entailment={"use": "none"}))
    assert models.models == {} and models.option("entailment", "min_score", 0.5) == 0.5
    status = models.status()
    assert set(status) == {spec.slot for spec in ANSWER_SLOTS}
    assert all(
        item["status"] == "disabled" and item["plugin"] == "none" for item in status.values()
    )


@pytest.mark.asyncio
async def test_scores_are_refused_until_ready_and_status_reports_load_errors() -> None:
    models = AnswerModels(ROOT, config(reranker=RERANK))
    reranker = models.get("reranker")
    assert reranker is not None
    with pytest.raises(RuntimeError, match="not ready"):
        await reranker.scores("q", ["p"], count=1)
    assert models.status()["reranker"]["status"] == "loading"
    reranker.error = "download failed"
    status = models.status()["reranker"]
    assert status["status"] == "degraded" and "download failed" in status["detail"]
    models.close()


class FakePool:
    """Returns ``result``, or raises it if it is an exception, after ``delay`` seconds."""

    def __init__(self, result: object, delay: float = 0.0) -> None:
        self.result = result
        self.delay = delay

    async def call(self, *args: object, **kwargs: object) -> object:
        await asyncio.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_wrong_length_results_are_rejected_rather_than_misaligned() -> None:
    short = LocalModel(FakePool([0.9]), "reranker", "fake", {})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="returned 1 scores for 2"):
        await short.scores("score", "q", ["a", "b"], count=2)
    wrong = LocalModel(FakePool({"a": 1}), "embedder", "fake", {})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="vectors for 1"):
        await wrong.embed(["a"])
    assert await short.scores("score", "q", [], count=0) == []


def back_reranker(models: AnswerModels, pool: FakePool) -> AnswerModel:
    """The reranker slot, marked loaded and backed by ``pool`` (call again to swap pools)."""
    reranker = models.get("reranker")
    assert reranker is not None
    reranker.model = LocalModel(pool, "reranker", "fake", {})  # pyright: ignore[reportArgumentType]
    reranker.ready = True
    return reranker


@pytest.mark.asyncio
async def test_three_timeouts_in_a_row_degrade_a_loaded_model_and_one_success_heals_it() -> None:
    models = AnswerModels(ROOT, config(reranker=RERANK))
    try:
        reranker = back_reranker(models, FakePool([0.5], delay=1.0))
        assert models.status()["reranker"]["status"] == "healthy"
        for attempt in range(1, FAILURE_LIMIT + 1):
            # The harness gives up first, as `asyncio.timeout` does in the answer path.
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.02):
                    await reranker.scores("q", ["p"], count=1)
            status = models.status()["reranker"]
            assert status["consecutive_failures"] == attempt
            assert status["status"] == ("degraded" if attempt == FAILURE_LIMIT else "healthy")
        status = models.status()["reranker"]
        assert "timed out after" in status["last_error"]
        assert status["last_error"] in status["detail"]
        assert 0.0 < status["last_latency_seconds"] < 1.0

        back_reranker(models, FakePool([0.9]))
        assert await reranker.scores("q", ["p"], count=1) == [0.9]
        status = models.status()["reranker"]
        assert status["status"] == "healthy" and status["consecutive_failures"] == 0
        assert status["last_error"] == "" and status["last_latency_seconds"] is not None
    finally:
        models.close()


@pytest.mark.asyncio
async def test_failures_count_but_a_success_between_them_resets_the_streak() -> None:
    models = AnswerModels(ROOT, config(reranker=RERANK))
    try:
        reranker = back_reranker(models, FakePool(RuntimeError("worker crashed")))
        for _ in range(FAILURE_LIMIT - 1):
            with pytest.raises(RuntimeError, match="worker crashed"):
                await reranker.scores("q", ["p"], count=1)
        back_reranker(models, FakePool([1.0]))
        await reranker.scores("q", ["p"], count=1)
        back_reranker(models, FakePool(RuntimeError("worker crashed")))
        with pytest.raises(RuntimeError):
            await reranker.scores("q", ["p"], count=1)
        assert models.status()["reranker"]["status"] == "healthy"
        for _ in range(FAILURE_LIMIT - 1):
            with pytest.raises(RuntimeError):
                await reranker.scores("q", ["p"], count=1)
        status = models.status()["reranker"]
        assert status["status"] == "degraded" and "worker crashed" in status["detail"]
    finally:
        models.close()


@pytest.mark.asyncio
async def test_a_pool_kill_after_timeout_is_reported_as_a_timeout() -> None:
    models = AnswerModels(ROOT, config(reranker=RERANK))
    try:
        reranker = back_reranker(models, FakePool(TimeoutError()))
        for _ in range(FAILURE_LIMIT):
            with pytest.raises(TimeoutError):
                await reranker.scores("q", ["p"], count=1)
        status = models.status()["reranker"]
        assert status["status"] == "degraded" and "timed out after" in status["detail"]
    finally:
        models.close()


@pytest.mark.asyncio
async def test_a_call_refused_because_the_model_is_not_ready_is_not_a_failure() -> None:
    models = AnswerModels(ROOT, config(reranker=RERANK))
    try:
        reranker = models.get("reranker")
        assert reranker is not None
        for _ in range(FAILURE_LIMIT + 1):
            with pytest.raises(RuntimeError, match="not ready"):
                await reranker.scores("q", ["p"], count=1)
        status = models.status()["reranker"]
        assert status["status"] == "loading" and status["consecutive_failures"] == 0
    finally:
        models.close()


@pytest.mark.asyncio
async def test_a_shared_pool_runs_both_real_models_in_one_process() -> None:
    hub = pytest.importorskip("huggingface_hub")
    for model in ("cross-encoder/nli-deberta-v3-xsmall", "cross-encoder/ms-marco-MiniLM-L6-v2"):
        if not isinstance(hub.try_to_load_from_cache(model, "onnx/model.onnx"), str):
            pytest.skip(f"{model} is not cached")
    models = AnswerModels(ROOT, config(entailment=ENTAIL, reranker=RERANK))
    try:
        entailment, reranker = models.get("entailment"), models.get("reranker")
        assert entailment is not None and reranker is not None
        entailment.ready = reranker.ready = True
        quote = "The encoder uses eight attention heads."
        [score] = await entailment.scores([(quote, "The encoder has 8 heads.")], count=1)
        ranks = await reranker.scores(
            "How many heads?", [quote, "We thank the reviewers."], count=2
        )
        assert score > 0.5 and ranks[0] > ranks[1]
        pool = models.pools[0]
        assert isinstance(pool, ModelPool)
        first = await pool.submit(10.0, os.getpid)
        assert first == await pool.submit(10.0, os.getpid) and first != os.getpid()
    finally:
        models.close()
