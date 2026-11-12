from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from arie_core.budget import BudgetHandle
from arie_core.registry.manager import (
    DEPRECATED_RUNTIME_QOS_KEYS,
    ArieConfig,
    PluginDefinition,
    RegistryManager,
)
from arie_core.sched import EventLoopLagMonitor, supervised


@dataclass
class Ctx:
    deadline: float
    trace_id: str = "trace"
    budget: BudgetHandle | None = None
    paper_id: str | None = None


class OldPlugin:
    def __init__(self) -> None:
        self.closed = False

    async def fetch(self, ctx: Ctx, *, paper_id: str | None = None) -> list[dict[str, object]]:
        await asyncio.sleep(0.05)
        return [{"paper_id": paper_id or ctx.paper_id, "source": "old"}]

    def close(self) -> None:
        self.closed = True


class NewPlugin:
    def __init__(self) -> None:
        self.closed = False

    async def fetch(self, ctx: Ctx, *, paper_id: str | None = None) -> list[dict[str, object]]:
        return [{"paper_id": paper_id or ctx.paper_id, "source": "new"}]

    def close(self) -> None:
        self.closed = True


def test_budget_enforces_tool_call_and_token_caps() -> None:
    budget = BudgetHandle(max_tool_calls=2, max_tokens=100, max_wall_sec=2.0)
    assert budget.record_tool_call("lookup") == "lookup"
    assert budget.record_tool_call("search") == "search"
    with pytest.raises(ValueError):
        budget.record_tool_call("lookup")
    with pytest.raises(RuntimeError):
        budget.record_tool_call("rerank")
    budget = BudgetHandle(max_tokens=10)
    budget.consume(10)
    with pytest.raises(RuntimeError):
        budget.consume(1)
    assert "BATS:" in budget.status()


@pytest.mark.asyncio
async def test_registry_hot_switch_keeps_inflight_requests() -> None:
    manager = RegistryManager(project_root=".")
    manager._plugin_defs["old_source"] = PluginDefinition(  # pyright: ignore[reportPrivateUsage]
        name="old_source",
        slot="source",
        module="tests.old",
        object_name="OldPlugin",
        factory=OldPlugin,
    )
    manager._plugin_defs["new_source"] = PluginDefinition(  # pyright: ignore[reportPrivateUsage]
        name="new_source",
        slot="source",
        module="tests.new",
        object_name="NewPlugin",
        factory=NewPlugin,
    )
    manager._slot_plugins["source"].update(  # pyright: ignore[reportPrivateUsage]
        {
            name: manager._plugin_defs[name]  # pyright: ignore[reportPrivateUsage]
            for name in ("old_source", "new_source")
        }
    )
    manager._instances["source"] = OldPlugin()  # pyright: ignore[reportPrivateUsage]
    manager._slot_to_name["source"] = "old_source"  # pyright: ignore[reportPrivateUsage]

    old_ctx = Ctx(deadline=10.0, budget=BudgetHandle(), paper_id="paper-1")
    in_flight = asyncio.create_task(
        manager.call_plugin("source", "fetch", old_ctx, paper_id="paper-1")
    )
    await asyncio.sleep(0.02)
    await manager.switch_plugin("source", "new_source")
    result = await in_flight
    old_items = result  # pyright: ignore[reportUnknownVariableType]
    assert old_items[0]["source"] == "old"  # pyright: ignore[reportIndexIssue]
    assert manager._instances["source"].__class__ is NewPlugin  # pyright: ignore[reportPrivateUsage]

    fresh_ctx = Ctx(deadline=10.0, budget=BudgetHandle(), paper_id="paper-2")
    next_result = await manager.call_plugin("source", "fetch", fresh_ctx, paper_id="paper-2")
    assert next_result[0]["source"] == "new"  # pyright: ignore[reportIndexIssue]
    assert next_result[0]["paper_id"] == "paper-2"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_lag_monitor_runs_under_a_task_group_and_reports_blocking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    monitor = EventLoopLagMonitor(threshold_ms=5.0, interval_ms=10.0)

    class Stop(Exception):
        pass

    def block_loop() -> None:
        time.sleep(0.05)  # Deliberately block the event loop past the threshold.

    async def block_then_stop() -> None:
        await asyncio.sleep(0.02)
        block_loop()
        await asyncio.sleep(0.03)
        raise Stop

    tasks: list[asyncio.Task[None]] = []
    with caplog.at_level(logging.WARNING, logger="arie_core.sched"):
        try:
            async with asyncio.TaskGroup() as group:
                tasks.append(group.create_task(supervised(monitor.run)()))
                group.create_task(supervised(block_then_stop)())
        except* Stop:
            pass
    assert tasks[0].cancelled()
    assert any("Event loop lag" in record.message for record in caplog.records)


def test_arie_yaml_with_retired_runtime_qos_keys_loads_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "arie.yaml"
    path.write_text(
        "runtime_qos:\n"
        "  use_uvloop: true\n"
        "  user_initiated_override_concurrency: 2\n"
        "  background_pdf_stream_concurrency: 4\n"
        "  arxiv_rate_limit_delay_sec: 3.0\n"
        "  utility_worker_processes: 2\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="arie_core.registry.manager"):
        config = ArieConfig.from_yaml(path)
    assert config.runtime_qos.utility_worker_processes == 2
    warnings = [r.getMessage() for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(warnings) == 1
    assert all(key in warnings[0] for key in DEPRECATED_RUNTIME_QOS_KEYS)
    assert not hasattr(config.runtime_qos, "use_uvloop")


def test_unknown_runtime_qos_key_is_still_rejected(tmp_path: Path) -> None:
    path = tmp_path / "arie.yaml"
    path.write_text("runtime_qos:\n  worker_count: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker_count"):
        ArieConfig.from_yaml(path)


def test_pdf_check_config_defaults_on_and_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "arie.yaml"
    path.write_text("ingestion:\n  pdf_check:\n    max_objects: 5\n", encoding="utf-8")
    check = ArieConfig.from_yaml(path).ingestion.pdf_check
    assert check.enabled and check.max_objects == 5
    path.write_text("ingestion:\n  pdf_check:\n    max_streams: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_streams"):
        ArieConfig.from_yaml(path)


def test_repo_arie_yaml_has_no_retired_runtime_qos_keys(caplog: pytest.LogCaptureFixture) -> None:
    root = Path(__file__).resolve().parents[2]
    with caplog.at_level(logging.WARNING, logger="arie_core.registry.manager"):
        ArieConfig.from_yaml(root / "arie.yaml")
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]


def test_context_tokens_default_per_plugin_and_explicit_override(tmp_path: Path) -> None:
    config = ArieConfig()
    assert config.explain.max_pack_share == 0.6
    assert config.context_tokens_for("deepseek") == 64_000
    assert config.context_tokens_for("openai") == 128_000
    assert config.context_tokens_for("gemini") == 128_000
    assert config.context_tokens_for("some-plugin") == 32_000

    path = tmp_path / "arie.yaml"
    path.write_text(
        "llm_providers:\n"
        "  deepseek: {model: m, api_key_env: DEEPSEEK_API_KEY, context_tokens: 100000}\n"
        "  other: {model: m, api_key_env: OTHER_KEY}\n"
        "explain:\n  max_pack_share: 0.4\n",
        encoding="utf-8",
    )
    loaded = ArieConfig.from_yaml(path)
    assert loaded.context_tokens_for("deepseek") == 100_000
    assert loaded.context_tokens_for("other") == 32_000  # configured without a window
    assert loaded.explain.max_pack_share == 0.4


def test_a_custom_endpoint_without_a_window_gets_the_fallback(tmp_path: Path) -> None:
    path = tmp_path / "arie.yaml"
    path.write_text(
        "llm_providers:\n"
        "  openai: {model: m, api_key_env: OPENAI_API_KEY, base_url: 'http://localhost:11434/v1'}\n"
        "  gemini: {model: m, api_key_env: GEMINI_API_KEY, base_url: 'http://proxy/v1',"
        " context_tokens: 200000}\n"
        "  deepseek: {model: m, api_key_env: DEEPSEEK_API_KEY}\n",
        encoding="utf-8",
    )
    loaded = ArieConfig.from_yaml(path)
    assert loaded.context_tokens_for("openai") == 32_000  # the endpoint may serve any model
    assert loaded.context_tokens_for("gemini") == 200_000  # an explicit window always wins
    assert loaded.context_tokens_for("deepseek") == 64_000  # no base_url: the plugin default


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("explain:\n  max_pack_share: 0\n", "max_pack_share"),
        ("explain:\n  max_pack_share: 1.5\n", "max_pack_share"),
        ("explain:\n  budget: 3\n", "budget"),
        ("answer:\n  default_mode: fast\n", "default_mode"),
        ("answer:\n  mode: explain\n", "mode"),
        ("judge:\n  chat: sometimes\n", "chat"),
        ("judge:\n  chat: 1\n", "chat"),
        ("judge:\n  eval: all\n", "eval"),
        ("eval:\n  judge_provider: nowhere\n", "judge_provider"),
        ("eval:\n  judge_provider: ''\n", "judge_provider"),
        (
            "llm_providers:\n  x: {model: m, api_key_env: X_KEY, context_tokens: 100}\n",
            "context_tokens",
        ),
    ],
)
def test_bad_explain_and_context_settings_are_rejected(
    tmp_path: Path, text: str, match: str
) -> None:
    path = tmp_path / "arie.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ArieConfig.from_yaml(path)


def test_repo_arie_yaml_states_the_context_windows_and_pack_share() -> None:
    root = Path(__file__).resolve().parents[2]
    config = ArieConfig.from_yaml(root / "arie.yaml")
    assert {name: p.context_tokens for name, p in config.llm_providers.items()} == {
        # The windows the providers' docs state for deepseek-flash, gpt-5.4-mini and
        # gemini-3.5-flash-lite (sources in docs/explain-mode/PROGRESS.md).
        "deepseek": 1_000_000,
        "openai": 400_000,
        "gemini": 1_048_576,
    }
    assert config.explain.max_pack_share == 0.6


def test_the_default_answer_mode_is_strict_and_a_config_can_change_it(tmp_path: Path) -> None:
    assert ArieConfig().answer.default_mode == "strict"
    root = Path(__file__).resolve().parents[2]
    assert ArieConfig.from_yaml(root / "arie.yaml").answer.default_mode == "strict"
    path = tmp_path / "arie.yaml"
    path.write_text("answer:\n  default_mode: explain\n", encoding="utf-8")
    assert ArieConfig.from_yaml(path).answer.default_mode == "explain"


def test_the_judge_is_on_in_chat_by_default_and_a_config_can_turn_it_off(tmp_path: Path) -> None:
    assert ArieConfig().judge.chat == "all" and ArieConfig().eval.judge_provider is None
    root = Path(__file__).resolve().parents[2]
    assert ArieConfig.from_yaml(root / "arie.yaml").judge.chat == "all"
    path = tmp_path / "arie.yaml"
    path.write_text("judge:\n  chat: off\neval:\n  judge_provider: openai\n", encoding="utf-8")
    loaded = ArieConfig.from_yaml(path)
    assert loaded.judge.chat == "off" and loaded.eval.judge_provider == "openai"
    # YAML reads the unquoted words as booleans; they still mean what they say.
    path.write_text("judge:\n  chat: off\n", encoding="utf-8")
    assert ArieConfig.from_yaml(path).judge.chat == "off"
    path.write_text("judge:\n  chat: on\n", encoding="utf-8")
    assert ArieConfig.from_yaml(path).judge.chat == "all"
