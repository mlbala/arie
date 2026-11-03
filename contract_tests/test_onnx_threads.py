"""Local ONNX plugins cap their CPU threads instead of using every core.

ONNX Runtime defaults to one intra-op thread per core, so an unconfigured embedder or
NLI model pegs the whole machine on each answer. Fake modules stand in for FastEmbed and
ONNX Runtime so these tests never download a model.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]


def build(slot: str, name: str, **kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin(slot, name, **kwargs)


@pytest.mark.parametrize(("slot", "name"), [("embedder", "fastembed"), ("entailment", "nli-onnx")])
def test_threads_below_one_are_rejected(slot: str, name: str) -> None:
    with pytest.raises(ValueError, match="threads"):
        build(slot, name, threads=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("threads", [None, 3])
async def test_fastembed_passes_the_thread_cap(
    threads: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    class TextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding)
    )
    monkeypatch.setenv("ARIE_EMBED_WORKER", "1")
    kwargs = {} if threads is None else {"threads": threads}
    assert await build("embedder", "fastembed", **kwargs).embed(None, ["text"]) == [[1.0, 0.0]]
    assert seen["threads"] == (threads or 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("threads", [None, 2])
async def test_nli_session_uses_the_thread_cap(
    threads: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            return [types.SimpleNamespace(tolist=lambda: [[0.0, 5.0]] * len(feed["input_ids"]))]

    class Tokenizer:
        @staticmethod
        def from_file(path: str) -> Tokenizer:
            return Tokenizer()

        def enable_truncation(self, *args: object, **kwargs: object) -> None: ...
        def enable_padding(self) -> None: ...

        def encode_batch(self, pairs: list[tuple[str, str]]) -> list[Any]:
            return [types.SimpleNamespace(ids=[1], attention_mask=[1], type_ids=[0]) for _ in pairs]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=InferenceSession),
    )
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"id2label": {"0": "contradiction", "1": "entailment"}}))
    plugin = build("entailment", "nli-onnx", **({} if threads is None else {"threads": threads}))
    monkeypatch.setattr(
        plugin, "_download", lambda: {"config": config, "tokenizer": config, "onnx": config}
    )
    monkeypatch.setenv("ARIE_ENTAILMENT_WORKER", "1")
    [score] = await plugin.score(None, [("premise", "hypothesis")])
    assert score > 0.99
    assert sessions[0].options.intra_op_num_threads == (threads or 1)
    assert sessions[0].options.inter_op_num_threads == 1
