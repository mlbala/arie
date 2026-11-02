from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
import yaml
from arie_core.budget import BudgetHandle
from arie_core.plugin_api import LLM, LLMRequest, configured_api_key
from arie_core.registry.manager import ArieConfig, LLMProviderConfig, RegistryManager

_LABELS = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "openai-compatible": "OpenAI-compatible",
}


@dataclass(frozen=True)
class ProbeContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


@dataclass(frozen=True)
class ConnectionProbe:
    checked_at: float
    status: str
    message: str
    instance: object | None = None


class LLMProviders:
    """Check server credentials and save verified provider choices without disclosing keys."""

    def __init__(
        self,
        root: Path,
        config: ArieConfig,
        manager: RegistryManager,
        lock: asyncio.Lock,
    ) -> None:
        self.root, self.config, self.manager, self.lock = root, config, manager, lock
        self.checks: dict[tuple[str, str, str, str], ConnectionProbe] = {}

    def profile(self, name: str) -> LLMProviderConfig:
        if name == "openai-compatible" and name not in self.config.llm_providers:
            return LLMProviderConfig(model="gpt-4.1-mini", api_key_env="OPENAI_API_KEY")
        if name not in self.config.llm_providers:
            raise ValueError(f"No LLM defaults are configured for {name!r}")
        return self.config.llm_providers[name]

    def settings(self, name: str, changes: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve only this provider's model, endpoint and server credential."""
        if name not in self.manager.get_available_plugins("llm"):
            raise ValueError("Unknown LLM provider")
        profile = self.profile(name)
        active = cast(dict[str, object], self.config.plugins.get("llm", {}))
        overrides = active if active.get("use") == name else {}
        supplied = changes or {}
        if set(supplied) - {"model"}:
            raise ValueError(
                "Only the model can be changed here; configure endpoints and keys on the server"
            )
        model = supplied.get("model", overrides.get("model", profile.model))
        if not isinstance(model, str) or not model.strip() or len(model) > 128:
            raise ValueError("Model must be a nonempty name of at most 128 characters")
        result: dict[str, object] = {"model": model.strip()}
        base_url = overrides.get("base_url") or profile.base_url
        if base_url:
            result["base_url"] = base_url
        raw_key = overrides.get("api_key")
        key = configured_api_key(str(raw_key) if raw_key else os.getenv(profile.api_key_env))
        if not key and name == "gemini" and profile.api_key_env == "GEMINI_API_KEY":
            key = configured_api_key(os.getenv("GOOGLE_API_KEY"))
        result["api_key"] = key or ""
        return result

    def _signature(self, name: str, settings: dict[str, object]) -> tuple[str, str, str, str]:
        return (
            name,
            str(settings["model"]),
            str(settings.get("base_url", "")),
            hashlib.sha256(str(settings["api_key"]).encode()).hexdigest(),
        )

    def describe(self, name: str) -> dict[str, object]:
        settings = self.settings(name)
        key_env = self.profile(name).api_key_env
        probe = self.checks.get(self._signature(name, settings))
        configured = bool(settings["api_key"])
        if not configured:
            status, message = "missing_key", f"{key_env} is not configured on the server."
        elif probe and time.monotonic() - probe.checked_at < 300:
            status, message = probe.status, probe.message
        else:
            status, message = "not_tested", "API key configured. Connection has not been tested."
        return {
            "name": name,
            "label": _LABELS.get(name, name),
            "model": settings["model"],
            "key_env": key_env,
            "key_configured": configured,
            "status": status,
            "message": message,
        }

    def options(self) -> list[dict[str, object]]:
        return [
            self.describe(name)
            for name in self.manager.get_available_plugins("llm")
            if name in self.config.llm_providers or name == "openai-compatible"
        ]

    async def check(
        self,
        name: str,
        changes: dict[str, object] | None = None,
        *,
        cached: bool = False,
    ) -> object:
        """Probe a candidate without activating it; reject missing or invalid keys."""
        settings = self.settings(name, changes)
        key_env = self.profile(name).api_key_env
        if not settings["api_key"]:
            raise ValueError(f"{key_env} is not configured. Add it to .env and restart ARIE.")
        signature = self._signature(name, settings)
        previous = self.checks.get(signature)
        if (
            cached
            and previous
            and previous.status == "verified"
            and time.monotonic() - previous.checked_at < 300
        ):
            return previous.instance
        label = _LABELS.get(name, name)
        try:
            instance = self.manager.build_plugin("llm", name, **settings)
            ctx = ProbeContext(
                time.monotonic() + 12.0, uuid.uuid4().hex, BudgetHandle(max_wall_sec=12.0)
            )
            async with asyncio.timeout(12.0):
                response = await cast(LLM, instance).complete(
                    ctx,
                    LLMRequest(
                        prompt='Return exactly this JSON object: {"ok":true}',
                        system_prompt="This is a connection check. Return JSON only.",
                        tools=[],
                        max_output_tokens=128,
                    ),
                )
            if json.loads(response.text) != {"ok": True}:
                raise ValueError("The provider did not return a valid JSON connection-check reply")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                message = f"{label} rejected the API key (HTTP {status}). Check {key_env}."
            elif status == 429:
                message = f"{label} rate limit or quota exceeded (HTTP 429)."
            else:
                message = (
                    f"{label} rejected the request (HTTP {status}). Check the model and {key_env}."
                )
            self.checks[signature] = ConnectionProbe(time.monotonic(), "error", message)
            raise ValueError(message) from None
        except TimeoutError, httpx.TimeoutException:
            message = f"{label} connection timed out."
            self.checks[signature] = ConnectionProbe(time.monotonic(), "error", message)
            raise ValueError(message) from None
        except Exception:
            message = f"{label} connection check failed. Check the model and server connection."
            self.checks[signature] = ConnectionProbe(time.monotonic(), "error", message)
            raise ValueError(message) from None
        self.checks[signature] = ConnectionProbe(
            time.monotonic(), "verified", "Connection verified.", instance
        )
        return instance

    async def switch(self, name: str, changes: dict[str, object]) -> dict[str, object]:
        """Verify before saving; preserve the active provider and YAML on failure."""
        instance = await self.check(name, changes, cached=True)
        settings = self.settings(name, changes)
        selection = {"use": name, "model": settings["model"]}

        def save() -> None:
            path = self.root / "arie.yaml"
            original = path.read_text(encoding="utf-8")
            loaded = cast(dict[str, object], yaml.safe_load(original))
            plugins = dict(cast(dict[str, object], loaded.get("plugins", {})))
            plugins["llm"] = selection
            block = yaml.safe_dump({"plugins": plugins}, sort_keys=False)
            changed, count = re.subn(
                r"(?m)^plugins:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*",
                lambda _: block,
                original,
                count=1,
            )
            if not count:
                changed = original.rstrip() + "\n" + block
            ArieConfig.model_validate(yaml.safe_load(changed))
            descriptor, temporary = tempfile.mkstemp(
                prefix=".arie-", suffix=".yaml", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(changed)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)

        async with self.lock:
            await asyncio.to_thread(save)
            self.manager.activate_plugin("llm", name, instance)
            self.config.plugins["llm"] = selection
        return {"slot": "llm", "plugin": name, "config": selection}
