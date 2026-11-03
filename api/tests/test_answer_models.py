"""Answer-path model slots: one table drives config, shared workers, warm-up and status."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from arie_api.models import ANSWER_SLOTS, AnswerModels
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
    def __init__(self, result: object) -> None:
        self.result = result

    async def call(self, *args: object, **kwargs: object) -> object:
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
