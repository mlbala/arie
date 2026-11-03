from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import httpx
import pytest
import yaml
from arie_api.providers import (
    DEFAULT_LLM,
    LLMProviders,
    resolve_embedder,
    save_plugin_selection,
)
from arie_api.runtime import ResearchRuntime, context
from arie_core.plugin_api import Embedder
from arie_core.registry.manager import ArieConfig, RegistryManager

ROOT = Path(__file__).resolve().parents[2]


def config_with(llm: str, embedder: str) -> ArieConfig:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins, llm={"use": llm}, embedder={"use": embedder})
    return config.model_copy(update={"plugins": plugins})


@pytest.mark.parametrize(
    ("llm", "keys", "expected", "reason"),
    [
        ("openai", {"OPENAI_API_KEY": "k"}, "openai", "Following the OpenAI"),
        ("gemini", {"GEMINI_API_KEY": "k"}, "gemini", "Following the Gemini"),
        ("gemini", {"GOOGLE_API_KEY": "k"}, "gemini", "Following the Gemini"),
        ("gemini", {}, "fastembed", "GEMINI_API_KEY is not set"),
        ("deepseek", {"DEEPSEEK_API_KEY": "k"}, "fastembed", "no embeddings API"),
        ("openai-compatible", {"OPENAI_API_KEY": "k"}, "fastembed", "no embeddings API"),
    ],
)
def test_auto_embedder_follows_provider_only_when_its_key_is_set(
    llm: str, keys: dict[str, str], expected: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    for env, value in keys.items():
        monkeypatch.setenv(env, value)
    choice = resolve_embedder(config_with(llm, "auto"))
    assert choice is not None and choice.name == expected and reason in choice.reason
    profile = config_with(llm, "auto").embedding_providers[expected]
    assert choice.settings == {"model": profile.model} and choice.min_score == profile.min_score
    assert not any("key" in str(key) for key in choice.settings)


def test_named_embedder_overrides_provider_and_none_disables_vectors() -> None:
    choice = resolve_embedder(config_with("openai", "fastembed"))
    assert choice is not None and choice.name == "fastembed"
    assert choice.settings["model"] == "BAAI/bge-small-en-v1.5"
    assert resolve_embedder(config_with("openai", "none")) is None


def test_saving_one_slot_keeps_comments_and_other_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (ROOT / "arie.yaml").read_text(encoding="utf-8")
    path = tmp_path / "arie.yaml"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("ARIE_CONFIG", str(path))
    save_plugin_selection(tmp_path, "embedder", {"use": "openai"})
    save_plugin_selection(tmp_path, "llm", {"use": "openai", "model": "gpt-5.4-mini"})
    changed = path.read_text(encoding="utf-8")
    kept = [line for line in original.splitlines() if line.lstrip().startswith("#")]
    assert all(line in changed for line in kept)
    plugins = cast(dict[str, object], yaml.safe_load(changed)["plugins"])
    assert plugins["embedder"] == {"use": "openai"}
    assert plugins["llm"] == {"use": "openai", "model": "gpt-5.4-mini"}
    assert plugins["vector_store"] == yaml.safe_load(original)["plugins"]["vector_store"]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.asyncio
async def test_cloud_embedders_send_server_key_and_keep_input_order(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "secret" not in str(request.url)
        if provider == "openai":
            assert request.url.host == "api.openai.com"
            assert request.headers["authorization"] == "Bearer server-secret"
            assert body == {"model": "text-embedding-3-small", "input": ["a", "b"]}
            # Out-of-order rows must be returned in input order.
            data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
            return httpx.Response(200, json={"data": data})
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.headers["x-goog-api-key"] == "server-secret"
        assert [item["content"]["parts"][0]["text"] for item in body["requests"]] == ["a", "b"]
        assert {item["outputDimensionality"] for item in body["requests"]} == {768}
        embeddings = [{"values": [1.0, 0.0]}, {"values": [0.0, 1.0]}]
        return httpx.Response(200, json={"embeddings": embeddings})

    original = httpx.AsyncClient

    def client(*, timeout: float, follow_redirects: bool = False) -> httpx.AsyncClient:
        assert follow_redirects is False
        return original(timeout=timeout, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    registry = RegistryManager(project_root=ROOT)
    assert {"fastembed", "openai", "gemini"} <= set(registry.get_available_plugins("embedder"))
    embedder = cast(Embedder, registry.build_plugin("embedder", provider, api_key="server-secret"))
    assert await embedder.embed(context("embed", 5.0), ["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_switching_model_provider_moves_auto_embedder_to_a_new_vector_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    config = config_with("deepseek", "auto")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        vector_store={"use": "exact", "path": str(tmp_path / "vectors.db")},
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))
    try:
        assert runtime.embedder_id == "fastembed:BAAI/bge-small-en-v1.5"
        before = runtime.embedding
        runtime.config.plugins["llm"] = {"use": "openai"}
        runtime.apply_embedder()
        assert runtime.embedder_id == "openai:text-embedding-3-small"
        assert runtime.min_score == 0.25 and runtime.embedding is not before
        assert runtime.vector_wakeup.is_set()
    finally:
        runtime.close()


@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.asyncio
async def test_cloud_embedders_wait_out_rate_limits(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        if provider == "openai":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        return httpx.Response(200, json={"embeddings": [{"values": [1.0]}]})

    original = httpx.AsyncClient

    def client(*, timeout: float, follow_redirects: bool = False) -> httpx.AsyncClient:
        return original(timeout=timeout, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    registry = RegistryManager(project_root=ROOT)
    embedder = cast(Embedder, registry.build_plugin("embedder", provider, api_key="k"))
    assert await embedder.embed(context("retry", 5.0), ["a"]) == [[1.0]]
    assert len(calls) == 2


def test_deepseek_is_the_default_and_other_providers_need_their_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    assert DEFAULT_LLM == "deepseek" and config.plugins["llm"] == {"use": "deepseek"}
    providers = LLMProviders(ROOT, config, RegistryManager(project_root=ROOT), asyncio.Lock())
    options = {str(option["name"]): option for option in providers.options()}
    assert [name for name, option in options.items() if option["default"]] == ["deepseek"]
    assert options["deepseek"]["key_configured"] is True
    assert options["openai"]["status"] == options["gemini"]["status"] == "missing_key"
