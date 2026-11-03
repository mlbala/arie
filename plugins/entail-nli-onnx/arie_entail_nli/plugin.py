from __future__ import annotations

import importlib
import json
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


@plugin(slot="entailment", name="nli-onnx")
class NLIEntailmentPlugin:
    """Local ONNX natural-language-inference cross-encoder, loaded only in arie-worker.

    Scores how strongly an exact source quote (premise) entails a drafted claim
    (hypothesis). Model files download from Hugging Face into ``HF_HOME`` (or
    ``cache_dir``) on first use. Premises are truncated first at ``max_length`` tokens,
    so a claim is never judged against a hypothesis cut short. ``threads`` caps ONNX
    Runtime's intra-op threads (default 1): its own default of one thread per core pegs
    every core on each answer and, measured on 10 claims, spends ~1.8x the CPU time of a
    single thread for ~2x less latency. Failure modes: download, ONNX load or label-map
    errors propagate from the first ``score`` call; ``threads`` below 1 raises
    ``ValueError``; calls outside a worker process raise ``RuntimeError``.
    """

    def __init__(
        self,
        model: str = "cross-encoder/nli-deberta-v3-xsmall",
        onnx_file: str = "onnx/model.onnx",
        cache_dir: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        threads: int = 1,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.model = model
        self.onnx_file = onnx_file
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.threads = threads
        self._model: _Model | None = None

    async def score(self, ctx: Ctx, pairs: Sequence[tuple[str, str]]) -> list[float]:
        del ctx
        if os.environ.get("ARIE_ENTAILMENT_WORKER") != "1":
            raise RuntimeError("NLI entailment must run inside arie-worker")
        if not pairs:
            return []
        if self._model is None:
            self._model = _OnnxNLI(self._download(), self.max_length, self.threads)
        scores: list[float] = []
        items = [(str(premise), str(hypothesis)) for premise, hypothesis in pairs]
        for start in range(0, len(items), self.batch_size):
            scores.extend(self._model.score(items[start : start + self.batch_size]))
        return scores

    def _download(self) -> dict[str, Path]:
        hub = importlib.import_module("huggingface_hub")
        fetch = cast(Callable[..., str], hub.hf_hub_download)
        return {
            name: Path(fetch(self.model, file, cache_dir=self.cache_dir))
            for name, file in (
                ("config", "config.json"),
                ("tokenizer", "tokenizer.json"),
                ("onnx", self.onnx_file),
            )
        }


class _OnnxNLI:
    """Tokenize pairs, run the cross-encoder, and softmax logits into P(entailment)."""

    def __init__(self, files: dict[str, Path], max_length: int, threads: int) -> None:
        config = cast(dict[str, object], json.loads(files["config"].read_text()))
        labels = cast(dict[str, str], config.get("id2label") or {})
        matches = [int(key) for key, label in labels.items() if label.lower() == "entailment"]
        if len(matches) != 1:
            raise ValueError("NLI model config has no single 'entailment' label")
        self.entailment = matches[0]
        tokenizers = importlib.import_module("tokenizers")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(files["tokenizer"]))
        self.tokenizer.enable_truncation(max_length, strategy="only_first")
        self.tokenizer.enable_padding()
        runtime = importlib.import_module("onnxruntime")
        options = runtime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
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
        if len(logits) != len(pairs):
            raise ValueError("NLI model returned the wrong number of rows")
        return [_softmax(row)[self.entailment] for row in logits]


def _softmax(row: list[float]) -> list[float]:
    peak = max(row)
    weights = [math.exp(value - peak) for value in row]
    total = sum(weights)
    return [weight / total for weight in weights]
