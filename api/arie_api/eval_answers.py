from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import msgspec
from arie_core.harness.evaluation import score_answer, source_truth_cases
from arie_core.harness.orchestrator import answer_question
from arie_core.plugin_api import LLM
from arie_core.registry.manager import ArieConfig, RegistryManager

from arie_api.providers import LLMProviders
from arie_api.runtime import context

ROOT = Path(__file__).resolve().parents[2]


async def evaluate() -> bool:
    """Evaluate the configured model against fixed, synthetic source facts."""
    config = ArieConfig.from_yaml(ROOT / "arie.yaml")
    manager = RegistryManager(project_root=ROOT, config=config)
    settings = cast(dict[str, object], config.plugins.get("llm", {}))
    providers = LLMProviders(ROOT, config, manager, asyncio.Lock())
    name = str(settings.get("use") or "deepseek")
    llm = cast(LLM, manager.build_plugin("llm", name, **providers.settings(name)))
    if not getattr(llm, "api_key", None):
        print(json.dumps({"status": "not_run", "reason": "Configured LLM has no API key"}))
        return False
    passed = True
    for case in source_truth_cases():
        async with asyncio.timeout(25.0):
            ctx = context(case.name, 25.0)
            run = await answer_question(
                ctx,
                "evaluation",
                case.question,
                case.rows,
                llm,
                project_root=ROOT,
                history=case.history,
            )
        result = run.answer
        score = score_answer(
            case,
            result,
            tool_calls=len(run.reading.actions),
            task=run.plan.task,
            planner=run.planner_status,
            guardrail=run.guardrail.code,
        )
        print(msgspec.json.encode(score).decode(), flush=True)
        if result.fallback_reason in {"provider_http_401", "provider_http_403"}:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": result.fallback_reason,
                        "message": "Provider rejected the configured credentials",
                    }
                )
            )
            return False
        passed = passed and score.passed
    print(
        json.dumps({"status": "passed" if passed else "failed", "cases": len(source_truth_cases())})
    )
    return passed


def main() -> None:
    raise SystemExit(0 if asyncio.run(evaluate()) else 1)


if __name__ == "__main__":
    main()
