from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from arie_core.plugin_api import Ctx, plugin


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Model(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@plugin(slot="reranker", name="onnx-cross-encoder")
class OnnxCrossEncoderReranker:
    """Local ONNX cross-encoder that scores how well a passage answers a question.

    Loaded only in arie-worker. The default ``cross-encoder/ms-marco-MiniLM-L6-v2``
    (~90 MB) downloads from Hugging Face into ``HF_HOME`` (or ``cache_dir``) on first
    use. Passages are truncated at ``max_length`` tokens; the question never is. Scores
    are sigmoid probabilities in [0, 1]. ``threads`` caps ONNX Runtime intra-op threads
    (default 2) so one answer never pegs every core: measured on an 8-core Mac, the
    worst case (24 passages of ~500 tokens) takes 3.7 s on one thread and 1.9 s on two,
    inside the harness's 3 s reranking budget.
    ``memory_arena`` (default off) keeps ONNX Runtime's CPU memory arena, which holds
    on to its peak allocation: measured on Linux with this model and the reranker in
    one worker, turning it off cut worker memory from ~1.2 GB to ~0.7 GB at the same
    speed.
    Failure modes: download, ONNX load
    or output-shape errors propagate from the first ``score`` call; ``threads`` below 1
    raises ``ValueError``; calls outside a worker process raise ``RuntimeError``.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        onnx_file: str = "onnx/model.onnx",
        cache_dir: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        threads: int = 2,
        memory_arena: bool = False,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.model = model
        self.onnx_file = onnx_file
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.threads = threads
        self.memory_arena = memory_arena
        self._model: _Model | None = None

    async def score(self, ctx: Ctx, query: str, passages: Sequence[str]) -> list[float]:
        del ctx
        if os.environ.get("ARIE_RERANK_WORKER") != "1":
            raise RuntimeError("Reranking must run inside arie-worker")
        if not passages:
            return []
        if self._model is None:
            self._model = _OnnxCrossEncoder(
                self._download(), self.max_length, self.threads, self.memory_arena
            )
        pairs = [(str(query), str(passage)) for passage in passages]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            scores.extend(self._model.score(pairs[start : start + self.batch_size]))
        return scores

    def _download(self) -> dict[str, Path]:
        hub = importlib.import_module("huggingface_hub")
        fetch = cast(Callable[..., str], hub.hf_hub_download)
        return {
            name: Path(fetch(self.model, file, cache_dir=self.cache_dir))
            for name, file in (("tokenizer", "tokenizer.json"), ("onnx", self.onnx_file))
        }


class _OnnxCrossEncoder:
    """Tokenize (question, passage) pairs and turn the relevance logit into a probability."""

    def __init__(
        self, files: dict[str, Path], max_length: int, threads: int, memory_arena: bool = False
    ) -> None:
        tokenizers = importlib.import_module("tokenizers")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(files["tokenizer"]))
        self.tokenizer.enable_truncation(max_length, strategy="only_second")
        self.tokenizer.enable_padding()
        runtime = importlib.import_module("onnxruntime")
        options = runtime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = memory_arena
        self.session = runtime.InferenceSession(
            str(files["onnx"]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {str(item.name) for item in self.session.get_inputs()}
        self.numpy = importlib.import_module("numpy")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        encodings = cast(list[_Encoding], self.tokenizer.encode_batch(pairs))
        columns = {
            "input_ids": [item.ids for item in encodings],
            "attention_mask": [item.attention_mask for item in encodings],
            "token_type_ids": [item.type_ids for item in encodings],
        }
        feed = {
            name: self.numpy.array(rows, dtype=self.numpy.int64)
            for name, rows in columns.items()
            if name in self.inputs
        }
        logits = cast(list[list[float]], self.session.run(None, feed)[0].tolist())
        if len(logits) != len(pairs) or any(len(row) != 1 for row in logits):
            raise ValueError("Reranker must return one relevance logit per pair")
        return [1.0 / (1.0 + math.exp(-row[0])) for row in logits]
