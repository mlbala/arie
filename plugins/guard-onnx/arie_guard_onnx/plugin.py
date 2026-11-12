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
    def score(self, texts: list[str]) -> list[float]: ...


@plugin(slot="guard", name="onnx-classifier")
class OnnxInjectionClassifier:
    """Local ONNX sequence classifier that scores how likely a text is to instruct a model.

    Loaded only in arie-worker. The default ``protectai/deberta-v3-base-prompt-injection-v2``
    (Apache-2.0, ~740 MB) downloads from Hugging Face into ``HF_HOME`` (or ``cache_dir``) on
    first use. Each text is truncated at ``max_length`` tokens. The score is the softmax
    probability of class ``label_index`` (the model's ``INJECTION`` label, 1 by default).
    ``threads`` caps ONNX Runtime intra-op threads (default 2). Measured in S20 on an 8-core
    Mac with 2 threads: ~750 ms per paper chunk and ~1.1 GB peak RSS (ADR 0014), so ARIE
    sends it only the ~1% of chunks its prefilter passes. ``memory_arena`` is off by
    default, as for the reranker.
    Failure modes: download, ONNX load or output-shape errors propagate from the first
    ``score`` call; ``threads`` below 1 raises ``ValueError``; calls outside a worker process
    raise ``RuntimeError``.
    """

    def __init__(
        self,
        model: str = "protectai/deberta-v3-base-prompt-injection-v2",
        onnx_file: str = "onnx/model.onnx",
        tokenizer_file: str = "onnx/tokenizer.json",
        cache_dir: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        threads: int = 2,
        label_index: int = 1,
        memory_arena: bool = False,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        if label_index < 0:
            raise ValueError("label_index must not be negative")
        self.model = model
        self.onnx_file = onnx_file
        self.tokenizer_file = tokenizer_file
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.threads = threads
        self.label_index = label_index
        self.memory_arena = memory_arena
        self._model: _Model | None = None

    async def score(self, ctx: Ctx, texts: Sequence[str]) -> list[float]:
        del ctx
        if os.environ.get("ARIE_GUARD_WORKER") != "1":
            raise RuntimeError("Guard scoring must run inside arie-worker")
        if not texts:
            return []
        if self._model is None:
            self._model = _OnnxClassifier(
                self._download(),
                self.max_length,
                self.threads,
                self.label_index,
                self.memory_arena,
            )
        items = [str(text) for text in texts]
        # Batch texts of similar length so padding stays small; return scores in input order.
        order = sorted(range(len(items)), key=lambda index: len(items[index]))
        scores: dict[int, float] = {}
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            values = self._model.score([items[index] for index in batch])
            scores.update(zip(batch, values, strict=True))
        return [scores[index] for index in range(len(items))]

    def _download(self) -> dict[str, Path]:
        hub = importlib.import_module("huggingface_hub")
        fetch = cast(Callable[..., str], hub.hf_hub_download)
        return {
            name: Path(fetch(self.model, file, cache_dir=self.cache_dir))
            for name, file in (("tokenizer", self.tokenizer_file), ("onnx", self.onnx_file))
        }


class _OnnxClassifier:
    """Tokenize texts and turn the classifier's logits into the injection probability."""

    def __init__(
        self,
        files: dict[str, Path],
        max_length: int,
        threads: int,
        label_index: int,
        memory_arena: bool,
    ) -> None:
        tokenizers = importlib.import_module("tokenizers")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(files["tokenizer"]))
        self.tokenizer.enable_truncation(max_length)
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
        self.label_index = label_index
        self.numpy = importlib.import_module("numpy")

    def score(self, texts: list[str]) -> list[float]:
        encodings = cast(list[_Encoding], self.tokenizer.encode_batch(texts))
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
        if len(logits) != len(texts) or any(len(row) <= self.label_index for row in logits):
            raise ValueError("Guard model must return a logit per class for each text")
        return [_softmax(row)[self.label_index] for row in logits]


def _softmax(row: list[float]) -> list[float]:
    top = max(row)
    exps = [math.exp(value - top) for value in row]
    total = sum(exps)
    return [value / total for value in exps]
