from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import msgspec
from arie_core.harness.evaluation import score_answer, source_truth_cases
from arie_core.harness.orchestrator import answer_question
from arie_core.harness.rerank import RerankCheck
from arie_core.harness.verifier import MIN_ENTAILMENT, EntailmentCheck
from arie_core.plugin_api import LLM
from arie_core.registry.manager import ArieConfig, RegistryManager

from arie_api.models import AnswerModels
from arie_api.providers import DEFAULT_LLM, LLMProviders, config_path
from arie_api.runtime import context

ROOT = Path(__file__).resolve().parents[2]


async def evaluate() -> bool:
    """Evaluate the configured model against fixed, synthetic source facts."""
    config = ArieConfig.from_yaml(config_path(ROOT))
    manager = RegistryManager(project_root=ROOT, config=config)
    settings = cast(dict[str, object], config.plugins.get("llm", {}))
    providers = LLMProviders(ROOT, config, manager, asyncio.Lock())
    name = str(settings.get("use") or DEFAULT_LLM)
    llm = cast(LLM, manager.build_plugin("llm", name, **providers.settings(name)))
    if not getattr(llm, "api_key", None):
        print(json.dumps({"status": "not_run", "reason": "Configured LLM has no API key"}))
        return False
    models = AnswerModels(ROOT, config)
    floor = models.option("entailment", "min_score", MIN_ENTAILMENT)
    entailment, reranker = models.get("entailment"), models.get("reranker")
    for model in (entailment, reranker):
        if model is not None:
            # The first load may download the model; answers are then timed as in production.
            await asyncio.wait_for(model.warm(), 600.0)

    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        assert entailment is not None
        return await entailment.scores(list(pairs), count=len(pairs))

    async def rerank(query: str, passages: Sequence[str]) -> list[float]:
        assert reranker is not None
        return await reranker.scores(query, list(passages), count=len(passages))

    try:
        return await _run_cases(
            llm,
            entail if entailment is not None else None,
            floor,
            rerank if reranker is not None else None,
        )
    finally:
        models.close()


async def _run_cases(
    llm: LLM, entail: EntailmentCheck | None, floor: float, rerank: RerankCheck | None
) -> bool:
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
                entail=entail,
                entail_floor=floor,
                rerank=rerank,
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
