from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from arie_api.routes import create_app
from arie_api.runtime import ResearchRuntime, context
from arie_core.plugin_api import LLM, Ctx, LLMRequest, LLMResponse, TokenUsage
from arie_core.registry.manager import ArieConfig, RegistryManager

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("provider", ["openai", "gemini", "deepseek"])
@pytest.mark.asyncio
async def test_adapters_send_real_bounded_json_requests_and_server_credentials(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        assert "secret" not in str(request.url)
        if provider == "gemini":
            assert request.headers["x-goog-api-key"] == "server-secret"
            assert body["generationConfig"]["maxOutputTokens"] == 32
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 5,
                        "thoughtsTokenCount": 2,
                    },
                },
            )
        assert request.headers["authorization"] == "Bearer server-secret"
        assert body["model"] != "gemini-3.5-flash-lite"
        assert body["max_completion_tokens" if provider == "openai" else "max_tokens"] == 32
        if provider == "deepseek":
            assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    original = httpx.AsyncClient

    def client(*, timeout: float, follow_redirects: bool = False) -> httpx.AsyncClient:
        return original(
            timeout=timeout,
            follow_redirects=follow_redirects,
            transport=httpx.MockTransport(respond),
        )

    monkeypatch.setattr(httpx, "AsyncClient", client)
    registry = RegistryManager(project_root=ROOT)
    assert {"openai", "gemini", "deepseek"} <= set(registry.get_available_plugins("llm"))
    model = cast(LLM, registry.build_plugin("llm", provider, api_key="server-secret"))
    response = await model.complete(
        context("probe", 5.0),
        LLMRequest(
            prompt="Return JSON.",
            system_prompt="JSON only.",
            max_output_tokens=32,
        ),
    )
    assert response.text == '{"ok":true}' and seen
    assert response.usage.completion == (7 if provider == "gemini" else 5)
    assert response.text != "Return JSON."


@pytest.mark.parametrize(
    ("provider", "key_env"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
    ],
)
@pytest.mark.asyncio
async def test_empty_or_placeholder_keys_fail_before_network(
    provider: str,
    key_env: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(key_env, "your-api-key-here")
    registry = RegistryManager(project_root=ROOT)
    model = cast(LLM, registry.build_plugin("llm", provider, api_key=""))
    assert getattr(model, "api_key", None) is None
    with pytest.raises((ValueError, RuntimeError), match="not configured"):
        await model.complete(context("missing", 5), LLMRequest(prompt="Test"))


async def runtime_at(tmp_path: Path) -> ResearchRuntime:
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    plugins = dict(config.plugins)
    plugins.update(
        database={"use": "sqlite", "path": str(tmp_path / "arie.db")},
        file_storage={"use": "local", "root": str(tmp_path / "papers")},
        embedder={"use": "none"},
        entailment={"use": "none"},
    )
    runtime = ResearchRuntime(ROOT, config.model_copy(update={"plugins": plugins}))
    runtime.providers.root = tmp_path
    (tmp_path / "arie.yaml").write_text((ROOT / "arie.yaml").read_text())
    await runtime.initialize()
    return runtime


@pytest.mark.asyncio
async def test_missing_key_is_visible_and_cannot_replace_active_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    runtime = await runtime_at(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)), base_url="http://test"
        ) as client:
            data = (await client.get("/v1/plugins")).json()
            llm = next(slot for slot in data["plugins"] if slot["slot"] == "llm")
            option = next(item for item in llm["providers"] if item["name"] == "openai")
            assert option["status"] == "missing_key" and not option["key_configured"]
            previous = dict(cast(dict[str, object], runtime.config.plugins["llm"]))
            before = (tmp_path / "arie.yaml").read_text()
            response = await client.put("/v1/plugins/llm", json={"plugin": "openai"})
            assert response.status_code == 400 and "OPENAI_API_KEY" in response.json()["detail"]
            assert runtime.config.plugins["llm"] == previous
            assert (tmp_path / "arie.yaml").read_text() == before
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_connection_probe_is_isolated_and_successful_selection_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    runtime = await runtime_at(tmp_path)
    calls: list[LLMRequest] = []
    active = object()
    runtime.manager.activate_plugin("llm", "deepseek", active)

    class Candidate:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx
            calls.append(request)
            return LLMResponse('{"ok":true}', TokenUsage(10, 5))

    candidate = Candidate()

    def build(slot: str, name: str, **kwargs: object) -> object:
        assert (slot, name) == ("llm", "openai")
        assert kwargs["model"] == "gpt-5.4-mini"
        assert kwargs["api_key"] == "private-test-key"
        return candidate

    monkeypatch.setattr(runtime.manager, "build_plugin", build)
    try:
        await runtime.providers.check("openai")
        assert runtime.manager.active_plugins["llm"] is active
        assert runtime.config.plugins["llm"] != {"use": "openai"}
        result = await runtime.providers.switch("openai", {"model": "gpt-5.4-mini"})
        assert len(calls) == 1  # Saving reuses the successful check for this model and key.
        assert runtime.manager.active_plugins["llm"] is candidate
        saved = ArieConfig.from_yaml(tmp_path / "arie.yaml")
        assert saved.plugins["llm"] == {"use": "openai", "model": "gpt-5.4-mini"}
        assert set(saved.llm_providers) >= {"openai", "gemini", "deepseek"}
        assert "private-test-key" not in json.dumps(result)
        assert "private-test-key" not in json.dumps(runtime.providers.options())
        assert "private-test-key" not in (tmp_path / "arie.yaml").read_text()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_rejected_credentials_are_sanitized_and_preserve_active_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "private-invalid-key")
    runtime = await runtime_at(tmp_path)
    active = object()
    runtime.manager.activate_plugin("llm", "deepseek", active)

    class Rejected:
        async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
            del ctx, request
            response = httpx.Response(
                401,
                request=httpx.Request("POST", "https://provider.test"),
                text="private-invalid-key",
            )
            response.raise_for_status()
            raise AssertionError("Unreachable")

    def build(*args: Any, **kwargs: Any) -> object:
        return Rejected()

    monkeypatch.setattr(runtime.manager, "build_plugin", build)
    before = (tmp_path / "arie.yaml").read_text()
    try:
        with pytest.raises(ValueError, match="HTTP 401") as error:
            await runtime.providers.switch("gemini", {})
        assert "private-invalid-key" not in str(error.value)
        assert runtime.providers.describe("gemini")["status"] == "error"
        assert runtime.manager.active_plugins["llm"] is active
        assert (tmp_path / "arie.yaml").read_text() == before
    finally:
        runtime.close()
