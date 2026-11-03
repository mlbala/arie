"""Reranker plugin: worker-only, fails closed on the wrong model, and ranks real passages.

The real-model test runs only when the model is already in the Hugging Face cache (or
``ARIE_RERANK_CACHE`` points at one), so the default suite never downloads ~90 MB.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def reranker(**kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin(
        "reranker", "onnx-cross-encoder", **kwargs
    )


def test_reranker_plugin_is_discovered() -> None:
    manager = RegistryManager(project_root=ROOT)
    assert "onnx-cross-encoder" in manager.get_available_plugins("reranker")


@pytest.mark.asyncio
async def test_scoring_outside_a_worker_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARIE_RERANK_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="inside arie-worker"):
        await reranker().score(None, "question", ["passage"])


def fake_runtime(monkeypatch: pytest.MonkeyPatch, logits: list[float]) -> list[Any]:
    """Stand in for ONNX Runtime and tokenizers; return the sessions created."""
    sessions: list[Any] = []

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class InferenceSession:
        def __init__(self, path: str, sess_options: SessionOptions, providers: list[str]) -> None:
            self.options = sess_options
            sessions.append(self)

        def get_inputs(self) -> list[Any]:
            return [types.SimpleNamespace(name="input_ids")]

        def run(self, outputs: object, feed: dict[str, Any]) -> list[Any]:
            rows = [logits] * len(feed["input_ids"])
            return [types.SimpleNamespace(tolist=lambda: rows)]

    class Tokenizer:
        truncation: dict[str, object] = {}

        @staticmethod
        def from_file(path: str) -> Tokenizer:
            return Tokenizer()

        def enable_truncation(self, max_length: int, strategy: str) -> None:
            Tokenizer.truncation = {"max_length": max_length, "strategy": strategy}

        def enable_padding(self) -> None: ...

        def encode_batch(self, pairs: list[tuple[str, str]]) -> list[Any]:
            return [types.SimpleNamespace(ids=[1], attention_mask=[1], type_ids=[0]) for _ in pairs]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=InferenceSession),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setenv("ARIE_RERANK_WORKER", "1")
    sessions.append(Tokenizer)
    return sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("threads", [None, 1])
async def test_scores_are_probabilities_with_capped_threads(
    threads: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = fake_runtime(monkeypatch, [3.0])
    file = tmp_path / "model"
    file.write_text("{}")
    plugin = reranker(**({} if threads is None else {"threads": threads}))
    monkeypatch.setattr(plugin, "_download", lambda: {"tokenizer": file, "onnx": file})
    scores = await plugin.score(None, "question", ["a", "b", "c"])
    assert len(scores) == 3 and all(0.95 < score < 1.0 for score in scores)
    tokenizer, session = created[0], created[1]
    # Passages are truncated, never the question.
    assert tokenizer.truncation == {"max_length": 512, "strategy": "only_second"}
    assert session.options.intra_op_num_threads == (threads or 2)
    assert session.options.inter_op_num_threads == 1


@pytest.mark.asyncio
async def test_a_classifier_with_several_labels_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_runtime(monkeypatch, [0.1, 0.9])
    file = tmp_path / "model"
    file.write_text(json.dumps({}))
    plugin = reranker()
    monkeypatch.setattr(plugin, "_download", lambda: {"tokenizer": file, "onnx": file})
    with pytest.raises(ValueError, match="one relevance logit"):
        await plugin.score(None, "question", ["passage"])


def test_threads_below_one_are_rejected() -> None:
    with pytest.raises(ValueError, match="threads"):
        reranker(threads=0)


def cached_model() -> str | None:
    explicit = os.environ.get("ARIE_RERANK_CACHE")
    if explicit:
        return explicit
    hub = pytest.importorskip("huggingface_hub")
    found = hub.try_to_load_from_cache(MODEL, "onnx/model.onnx")
    return None if not isinstance(found, str) else str(Path(found).parents[4])


@pytest.mark.asyncio
async def test_real_model_in_a_worker_ranks_the_answering_passage_first() -> None:
    cache = cached_model()
    if cache is None:
        pytest.skip(f"{MODEL} is not cached; set ARIE_RERANK_CACHE to run")
    from arie_worker.reranking import RerankPool

    pool = RerankPool(ROOT, "onnx-cross-encoder", {"model": MODEL, "cache_dir": cache})
    passages = [
        "Retrieval models are widely used in question answering systems today.",
        "We propose Cedar, a retrieval model that uses a gated memory bank.",
        "Training uses four CPU workers and a local index of documents.",
        "Smith et al. 2020. Proceedings of the Retrieval Conference.",
    ]
    try:
        workers = await pool.score("How many CPU workers are used?", passages, 120.0)
        contribution = await pool.score("What does this paper propose?", passages, 120.0)
    finally:
        pool.close()
    assert max(range(4), key=workers.__getitem__) == 2
    assert max(range(4), key=contribution.__getitem__) == 1
