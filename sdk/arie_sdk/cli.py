from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_ROOT = REPO_ROOT / "plugins"
SLOT_PREFIXES = {
    "database": "db",
    "db": "db",
    "source": "source",
    "vector": "vector",
    "embed": "embed",
    "files": "files",
    "file": "files",
    "parser": "parser",
    "llm": "llm",
    "reranker": "reranker",
    "storage": "files",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return "plugin"
    return slug


def _camel(value: str) -> str:
    parts = [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]
    if not parts:
        return "Plugin"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _make_plugin_template(slot: str, name: str) -> str:
    slug = _slug(name)
    plugin_class = f"{_camel(slug)}{slot.title()}Plugin"
    slot_name = slot.strip().lower()
    if slot_name == "source":
        return (
            "from __future__ import annotations\n\n"
            "from collections.abc import Mapping\n"
            "from typing import Any\n\n"
            "from arie_core.plugin_api import (\n"
            "    ScrapedPaperInput,\n"
            "    normalize_scraper_output,\n"
            "    plugin,\n"
            ")\n\n"
            f"@plugin(slot={slot_name!r}, name={slug!r})\n"
            f"class {plugin_class}:\n"
            "    \"\"\"Scaffolded ARIE source plugin.\"\"\"\n\n"
            "    async def fetch(\n"
            "        self, ctx: object | None, source_id: str\n"
            "    ) -> list[ScrapedPaperInput]:\n"
            "        del ctx\n"
            "        if source_id != \"" + slug + "\":\n"
            "            return []\n"
            "        return []\n\n\n__all__ = [\"" + plugin_class + "\"]\n"
        )
    if slot_name == "database":
        return (
            "from __future__ import annotations\n\n"
            "from collections.abc import Sequence\n"
            "from typing import Any\n\n"
            "from arie_core.plugin_api import plugin\n\n"
            f"@plugin(slot={slot_name!r}, name={slug!r})\n"
            f"class {plugin_class}:\n"
            "    \"\"\"Scaffolded ARIE database plugin.\"\"\"\n\n"
            "    async def startup(self, ctx: object | None = None) -> None:\n"
            "        del ctx\n\n"
            "    async def query(\n"
            "        self, *, sql: str, params: Sequence[Any] | None = None\n"
            "    ) -> list[dict[str, Any]]:\n"
            "        del sql, params\n"
            "        return []\n\n\n__all__ = [\"" + plugin_class + "\"]\n"
        )
    return (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from arie_core.plugin_api import plugin\n\n"
        f"@plugin(slot={slot_name!r}, name={slug!r})\n"
        f"class {plugin_class}:\n"
        "    \"\"\"Scaffolded ARIE plugin.\"\"\"\n\n"
        "    async def run(self, ctx: object | None = None, **kwargs: Any) -> Any:\n"
        "        del ctx, kwargs\n"
        "        return None\n\n\n__all__ = [\"" + plugin_class + "\"]\n"
    )


def _scaffold_plugin(slot: str, name: str) -> Path:
    slot_key = slot.strip().lower()
    slot_prefix = SLOT_PREFIXES.get(slot_key, slot_key)
    slug = _slug(name)
    package_dir = PLUGINS_ROOT / f"{slot_prefix}-{slug}"
    if package_dir.exists():
        raise FileExistsError(f"Plugin scaffold already exists at {package_dir}")

    package_name = f"arie_{slot_prefix}_{slug.replace('-', '_')}"
    package_path = package_dir / package_name
    package_path.mkdir(parents=True)

    pyproject = package_dir / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        f"name = \"arie-plugin-{slot_prefix}-{slug}\"\n"
        "version = \"0.1.0\"\n"
        "requires-python = \">=3.14\"\n"
        "dependencies = [\n"
        "    \"arie-core\",\n"
        "]\n\n"
        "[build-system]\n"
        "requires = [\"hatchling\"]\n"
        "build-backend = \"hatchling.build\"\n\n"
        "[tool.uv.sources]\n"
        "arie-core = { workspace = true }\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        f"packages = [\"{package_name}\"]\n",
        encoding="utf-8",
    )

    package_init = package_path / "__init__.py"
    package_init.write_text(
        "from .plugin import *\n",
        encoding="utf-8",
    )

    plugin_file = package_path / "plugin.py"
    plugin_file.write_text(_make_plugin_template(slot_key, slug), encoding="utf-8")
    return package_dir


def _resolve_plugin_target(target: str | None) -> str | None:
    if not target:
        return None
    raw = str(target).strip()
    if raw.startswith("/") or raw.startswith("."):
        return raw
    if raw.startswith("plugins/"):
        return str((REPO_ROOT / raw).resolve())

    candidate = (REPO_ROOT / "plugins" / raw).resolve()
    if candidate.exists():
        return str(candidate)

    if raw.startswith("source-") or raw.startswith("db-") or raw.startswith("llm-"):
        return str((REPO_ROOT / "plugins" / raw).resolve())
    return raw


def _run_plugin_test(plugin_target: str | None) -> int:
    command = [shutil.which("uv") or "uv", "run", "pytest", "contract_tests/", "-q", "--tb=short"]
    env = os.environ.copy()
    plugin_paths: list[str] = []
    if plugin_target:
        plugin_path = Path(plugin_target).resolve()
        if plugin_path.exists():
            plugin_paths.append(str(plugin_path))
        else:
            plugin_path = (REPO_ROOT / "plugins" / plugin_target).resolve()
            if plugin_path.exists():
                plugin_paths.append(str(plugin_path))
    if plugin_paths:
        current_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [*plugin_paths, current_path] if current_path else plugin_paths
        )

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def _group_plugin_new(args: argparse.Namespace) -> int:
    scaffold_dir = _scaffold_plugin(args.slot, args.name)
    print(f"Scaffolded plugin at {scaffold_dir}")
    return 0


def _group_plugin_test(args: argparse.Namespace) -> int:
    target = _resolve_plugin_target(args.plugin or args.target)
    return _run_plugin_test(target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plugin_group = subparsers.add_parser("plugin", help="Create or validate ARIE plugins")
    plugin_sub = plugin_group.add_subparsers(dest="plugin_command", required=True)

    new_parser = plugin_sub.add_parser("new", help="Scaffold a new plugin package")
    new_parser.add_argument("--slot", required=True, help="Plugin slot such as source or database")
    new_parser.add_argument("--name", required=True, help="Plugin name such as deepmind")
    new_parser.set_defaults(func=_group_plugin_new)

    test_parser = plugin_sub.add_parser("test", help="Run contract tests against a target plugin")
    test_parser.add_argument("target", nargs="?", help="Target plugin path or directory name")
    test_parser.add_argument("--plugin", dest="plugin", help="Target plugin path or directory name")
    test_parser.set_defaults(func=_group_plugin_test)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
