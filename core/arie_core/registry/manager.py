from __future__ import annotations

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
import asyncio
import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypeVar, cast

import yaml
from dotenv import load_dotenv
from opentelemetry import metrics, trace
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arie_core.budget import BudgetHandle
from arie_core.events import EventBus
from arie_core.registry.sync import SyncConfig

logger = logging.getLogger(__name__)

load_dotenv()

_T = TypeVar("_T")


class RetryablePluginError(RuntimeError):
    pass


@dataclass(frozen=True)
class PluginDefinition:
    name: str
    slot: str
    module: str
    object_name: str
    factory: Callable[..., object] | None = None


# Keys that were validated but never read. They are still accepted so existing arie.yaml
# copies load; ``ArieConfig.from_yaml`` warns when a file has them.
DEPRECATED_RUNTIME_QOS_KEYS = (
    "use_uvloop",
    "user_initiated_override_concurrency",
    "background_pdf_stream_concurrency",
    "arxiv_rate_limit_delay_sec",
)


class RuntimeQoSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slow_callback_threshold_ms: float = 15.0
    utility_worker_processes: int = 4
    # Entailment and reranker models share one worker process; false gives each its own.
    shared_model_worker: bool = True

    @model_validator(mode="before")
    @classmethod
    def drop_deprecated_keys(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        known = cast(dict[str, object], data)
        return {
            key: value for key, value in known.items() if key not in DEPRECATED_RUNTIME_QOS_KEYS
        }

    @field_validator("slow_callback_threshold_ms")
    @classmethod
    def ensure_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("values must be non-negative")
        return value


class PluginConfigEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    use: str
    model: str | None = None
    path: str | None = None
    root: str | None = None
    inline: str | None = None


# Context windows in tokens, chosen to be conservative but not checked against the providers'
# documented limits. A provider that sets ``context_tokens`` in ``llm_providers`` overrides
# these; an unknown plugin, or a provider with a ``base_url`` and no ``context_tokens`` (the
# endpoint may serve any model), gets the fallback. Explain mode sizes the paper pack against
# this number (ADR 0010).
DEFAULT_CONTEXT_TOKENS = {"deepseek": 64_000, "openai": 128_000, "gemini": 128_000}
FALLBACK_CONTEXT_TOKENS = 32_000


class LLMProviderConfig(BaseModel):
    """Server-side provider defaults; credentials are resolved from environment variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    base_url: str | None = None
    # None means the plugin's default in ``DEFAULT_CONTEXT_TOKENS``.
    context_tokens: int | None = Field(default=None, ge=4096)


def default_llm_providers() -> dict[str, LLMProviderConfig]:
    return {
        "deepseek": LLMProviderConfig(model="deepseek-flash", api_key_env="DEEPSEEK_API_KEY"),
        "openai": LLMProviderConfig(model="gpt-5.4-mini", api_key_env="OPENAI_API_KEY"),
        "gemini": LLMProviderConfig(model="gemini-3.5-flash-lite", api_key_env="GEMINI_API_KEY"),
    }


class EmbeddingProviderConfig(BaseModel):
    """Embedder defaults; ``min_score`` is the cosine floor calibrated for that model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    min_score: float = Field(ge=-1.0, le=1.0)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")


def default_embedding_providers() -> dict[str, EmbeddingProviderConfig]:
    return {
        "fastembed": EmbeddingProviderConfig(model="BAAI/bge-small-en-v1.5", min_score=0.6),
        "openai": EmbeddingProviderConfig(
            model="text-embedding-3-small", min_score=0.25, api_key_env="OPENAI_API_KEY"
        ),
        "gemini": EmbeddingProviderConfig(
            model="gemini-embedding-001", min_score=0.55, api_key_env="GEMINI_API_KEY"
        ),
    }


class ExplainConfig(BaseModel):
    """Explain-mode settings (ADR 0010)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    # The whole paper pack is used only if its estimated tokens fit within this share of the
    # provider's context window; otherwise the paged pack is used.
    max_pack_share: float = Field(default=0.6, gt=0.0, le=1.0)


class ArieConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = "hybrid"
    runtime_qos: RuntimeQoSConfig = Field(default_factory=RuntimeQoSConfig)
    plugins: dict[str, object] = Field(default_factory=dict)
    llm_providers: dict[str, LLMProviderConfig] = Field(default_factory=default_llm_providers)
    embedding_providers: dict[str, EmbeddingProviderConfig] = Field(
        default_factory=default_embedding_providers
    )
    sync: SyncConfig = Field(default_factory=lambda: SyncConfig())
    explain: ExplainConfig = Field(default_factory=ExplainConfig)

    def context_tokens_for(self, provider: str) -> int:
        """The context window to plan for with this LLM plugin."""
        profile = self.llm_providers.get(provider)
        if profile is not None and profile.context_tokens is not None:
            return profile.context_tokens
        if profile is not None and profile.base_url:
            return FALLBACK_CONTEXT_TOKENS
        return DEFAULT_CONTEXT_TOKENS.get(provider, FALLBACK_CONTEXT_TOKENS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArieConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = cast(dict[str, object], yaml.safe_load(handle) or {})
        runtime_qos = loaded.get("runtime_qos")
        if isinstance(runtime_qos, dict):
            present = [
                key
                for key in DEPRECATED_RUNTIME_QOS_KEYS
                if key in cast(dict[str, object], runtime_qos)
            ]
            if present:
                logger.warning(
                    "%s: runtime_qos keys %s are deprecated and ignored; remove them",
                    path,
                    ", ".join(present),
                )
        return cls.model_validate(loaded)


class HealthCheckable(Protocol):
    def ping(self) -> bool | object: ...


class Closable(Protocol):
    def close(self) -> object | None: ...


class CircuitBreaker:
    """A simple half-open circuit breaker guarding cached plugin endpoints."""

    def __init__(self, *, failure_threshold: int = 3, cool_down_sec: float = 5.0) -> None:
        self.failure_threshold = failure_threshold
        self.cool_down_sec = cool_down_sec
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        return (time.monotonic() - self._opened_at) >= self.cool_down_sec

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class RegistryManager:
    """Discovers available plugins and mediates plugin execution with QoS + safety wrappers."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        config: ArieConfig | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.config = config or ArieConfig()
        self._plugin_defs: dict[str, PluginDefinition] = {}
        self._slot_to_name: dict[str, str] = {}
        self._slot_plugins: dict[str, dict[str, PluginDefinition]] = defaultdict(dict)
        self._instances: dict[str, object] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)
        self._flight: dict[str, int] = defaultdict(int)
        self._flight_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._event_bus = EventBus()
        self._meter = metrics.get_meter("arie_core.registry")
        self._counter = self._meter.create_counter("plugin.calls")
        self._tracer = trace.get_tracer("arie_core.registry")
        self._discover_plugins()

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def active_plugins(self) -> dict[str, object]:
        return dict(self._instances)

    def _discover_plugins(self) -> None:
        self._register_builtin_plugins()
        self._register_entry_point_plugins()
        self._register_workspace_plugins()

    def _register_builtin_plugins(self) -> None:
        module = importlib.import_module("arie_core")
        del module

    def _register_entry_point_plugins(self) -> None:
        try:
            discovered = list(importlib.metadata.entry_points(group="arie.plugins"))
        except TypeError:
            groups = importlib.metadata.entry_points()
            discovered = list(groups.get("arie.plugins", []))
        for entry_point in discovered:
            if not hasattr(entry_point, "load"):
                continue
            name = getattr(entry_point, "name", None)
            if not isinstance(name, str):
                continue
            try:
                plugin_module = entry_point.load()
            except Exception:
                logger.warning("Skipped plugin entry point %r: import failed", name, exc_info=True)
                continue
            self._index_module(plugin_module, name)

    def _register_workspace_plugins(self) -> None:
        plugin_root = self.project_root / "plugins"
        if not plugin_root.exists():
            return
        for candidate in sorted(plugin_root.iterdir()):
            if candidate.is_dir():
                pkg_root = candidate / candidate.name
                candidate_paths = [candidate]
                if pkg_root.exists():
                    candidate_paths.append(pkg_root)
                for item in candidate_paths:
                    for file_path in sorted(item.rglob("*.py")):
                        if file_path.name.startswith("_"):
                            continue
                        self._import_python_file(file_path)
            elif candidate.suffix == ".py":
                self._import_python_file(candidate)

    def _import_python_file(self, file_path: Path) -> None:
        module_name = f"_arie_plugin_{file_path.stem}_{abs(hash(file_path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # A plugin that fails to import is unavailable; say why instead of hiding it.
            sys.modules.pop(module_name, None)
            logger.warning("Skipped plugin file %s: import failed", file_path, exc_info=True)
            return
        self._index_module(module, file_path.stem)

    def _index_module(self, module: ModuleType, fallback_name: str) -> None:
        for _, attribute in vars(module).items():
            if not inspect.isclass(attribute):
                continue
            slot = getattr(attribute, "__arie_plugin_slot__", None)
            name = getattr(attribute, "__arie_plugin_name__", None)
            if not slot or not name:
                continue
            definition = PluginDefinition(
                name=str(name),
                slot=str(slot),
                module=module.__name__,
                object_name=getattr(attribute, "__name__", str(attribute)),
                factory=attribute,
            )
            self._plugin_defs[definition.name] = definition
            self._slot_plugins[definition.slot][definition.name] = definition
            self._slot_to_name.setdefault(definition.slot, definition.name)

    def load_yaml_config(self, path: str | Path) -> ArieConfig:
        self.config = ArieConfig.from_yaml(path)
        return self.config

    def get_available_plugins(self, slot: str) -> list[str]:
        definitions = self._slot_plugins.get(slot, {})
        return sorted(definitions)

    def _normalize_plugin_config(self, slot: str, value: object) -> tuple[str, dict[str, object]]:
        if isinstance(value, str):
            return value, {}
        if isinstance(value, list):
            if not value:
                return "", {}
            return str(value[0]), {}
        if isinstance(value, dict):
            config = dict(value)
            plugin_name = str(config.pop("use", ""))
            params = {str(key): val for key, val in config.items()}
            return plugin_name, params
        return "", {}

    def instantiate_plugin(
        self,
        slot: str,
        plugin_name: str | None = None,
        **kwargs: object,
    ) -> object:
        if plugin_name is None:
            configured = self.config.plugins.get(slot, "")
            if isinstance(configured, dict):
                plugin_name = str(configured.get("use", ""))
            elif isinstance(configured, list) and configured:
                plugin_name = str(configured[0])
            else:
                plugin_name = str(configured)
        if not plugin_name:
            raise ValueError(f"No plugin configured for slot: {slot}")
        instance = self.build_plugin(slot, plugin_name, **kwargs)
        self.activate_plugin(slot, plugin_name, instance)
        return instance

    def activate_plugin(self, slot: str, plugin_name: str, instance: object) -> None:
        """Activate an already validated instance without reconstructing or retesting it."""
        self._instances[slot] = instance
        self._slot_to_name[slot] = plugin_name

    def build_plugin(self, slot: str, plugin_name: str, **kwargs: object) -> object:
        """Construct a candidate without changing the active slot."""
        definition = self._slot_plugins.get(slot, {}).get(plugin_name)
        if definition is None:
            raise KeyError(f"Plugin {plugin_name!r} not found for slot {slot!r}")
        return definition.factory(**kwargs) if definition.factory is not None else definition

    async def health_check(self, instance: object) -> bool:
        health_handler = getattr(instance, "healthcheck", None)
        if callable(health_handler):
            result = health_handler()
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        ping = getattr(instance, "ping", None)
        if callable(ping):
            result = ping()
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        return True

    async def switch_plugin(self, slot: str, new_name: str) -> str:
        if new_name not in self._plugin_defs:
            raise KeyError(f"Plugin {new_name!r} not found")
        old_instance = self._instances.get(slot)
        config = self.config.plugins.get(slot, {})
        params: dict[str, object] = {}
        if isinstance(config, dict):
            params = {str(key): value for key, value in config.items() if key != "use"}
        new_instance = self.instantiate_plugin(slot, new_name, **params)
        if not await self.health_check(new_instance):
            raise RuntimeError(f"Plugin {new_name!r} failed health check")
        self._instances[slot] = new_instance
        self._slot_to_name[slot] = new_name
        if old_instance is not None and old_instance is not new_instance:
            event = self._flight_events[slot]
            while self._flight.get(slot, 0) > 0:
                await event.wait()
                event.clear()
            shutdown = getattr(old_instance, "close", None)
            if callable(shutdown):
                close_result = shutdown()
                if asyncio.iscoroutine(close_result):
                    await close_result
        return new_name

    def _post_filter_result(self, result: object, *, paper_id: str | None) -> object:
        if paper_id is None:
            return result

        def matches(item: object) -> bool:
            if isinstance(item, dict):
                return str(item.get("paper_id")) == paper_id
            return str(getattr(item, "paper_id", None)) == paper_id

        if isinstance(result, list):
            return [item for item in result if matches(item)]
        if isinstance(result, tuple):
            return tuple(item for item in result if matches(item))
        return result

    async def _call_with_safety(
        self,
        slot: str,
        method_name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        instance = self._instances.get(slot)
        if instance is None:
            raise KeyError(f"No active plugin for slot: {slot}")
        method = getattr(instance, method_name, None)
        if method is None:
            raise AttributeError(f"Plugin {slot!r} has no method {method_name!r}")

        breaker = self._circuit_breakers[slot]
        if not breaker.allow():
            raise RetryablePluginError(f"Circuit open for plugin {slot!r}")

        ctx = None
        if args:
            first = args[0]
            if hasattr(first, "deadline") or hasattr(first, "paper_id"):
                ctx = first
        if ctx is None:
            ctx = kwargs.get("ctx")

        absolute_deadline: float | None = None
        if ctx is not None:
            deadline_value = getattr(ctx, "deadline", None)
            if isinstance(deadline_value, (int, float)):
                deadline_float = float(deadline_value)
                if deadline_float > 1_000_000.0:
                    absolute_deadline = deadline_float
                else:
                    absolute_deadline = time.monotonic() + max(0.0, deadline_float)

        budget = getattr(ctx, "budget", None) if ctx is not None else None
        if isinstance(budget, BudgetHandle):
            budget.reserve(1.0)

        paper_id: str | None = None
        if ctx is not None:
            value = getattr(ctx, "paper_id", None)
            if isinstance(value, str):
                paper_id = value
        if paper_id is None:
            paper_value = kwargs.get("paper_id")
            if isinstance(paper_value, str):
                paper_id = paper_value

        delay = 0.05
        for attempt in range(1, 6):
            self._flight[slot] += 1
            try:
                with self._tracer.start_as_current_span(f"plugin.{slot}.{method_name}") as span:
                    self._counter.add(1, {"plugin.slot": slot, "plugin.method": method_name})
                    span.set_attribute("plugin.slot", slot)
                    span.set_attribute("plugin.method", method_name)
                    if absolute_deadline is not None:
                        timeout_seconds = max(0.0, absolute_deadline - time.monotonic())
                        async with asyncio.timeout(timeout_seconds):
                            result = method(*args, **kwargs)
                            if asyncio.iscoroutine(result):
                                result = await result
                    else:
                        result = method(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
                    breaker.record_success()
                    return self._post_filter_result(result, paper_id=paper_id)
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                breaker.record_failure()
                if attempt >= 5 or not isinstance(exc, (TimeoutError, ConnectionError, OSError)):
                    raise
                backoff = delay + random.uniform(0.0, 0.25)
                await asyncio.sleep(backoff)
                delay *= 2
            except Exception:
                breaker.record_failure()
                raise
            finally:
                next_count = max(0, self._flight.get(slot, 0) - 1)
                self._flight[slot] = next_count
                if next_count == 0:
                    self._flight_events[slot].set()

        raise RuntimeError(f"Plugin call {slot}.{method_name} failed after retries")

    async def call_plugin(
        self,
        slot: str,
        method_name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        return await self._call_with_safety(slot, method_name, *args, **kwargs)


__all__ = [
    "ArieConfig",
    "CircuitBreaker",
    "PluginConfigEntry",
    "PluginDefinition",
    "RegistryManager",
    "RetryablePluginError",
    "RuntimeQoSConfig",
]
