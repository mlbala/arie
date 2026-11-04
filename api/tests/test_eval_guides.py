"""Gate 3's comparison: guided and unguided overview answers, judged pairwise."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest
from arie_api import eval_guides
from arie_api.runtime import CallContext, Document
from arie_core.harness.judge import PairwiseJudgement
from arie_core.harness.orchestrator import ResearchAnswer


class FakeRuntime:
    def __init__(self) -> None:
        self.asked: list[tuple[str, bool]] = []
        self.judged: list[tuple[str, str, list[str]]] = []

    async def get_paper(self, paper_id: str) -> Document | None:
        abstract = "Cedar gates memory writes. It improves accuracy by 6 points. We release code."
        return {"id": paper_id, "abstract": abstract} if paper_id == "paper" else None

    async def answer_paper_question(
        self, ctx: CallContext, paper: Document, question: str, *, guides: bool = True
    ) -> ResearchAnswer:
        del ctx, paper
        self.asked.append((question, guides))
        return cast(ResearchAnswer, SimpleNamespace(answer=f"{question}|{guides}"))

    async def judge_explain_vs_strict(
        self,
        trace_id: str,
        question: str,
        key_facts: Sequence[str],
        explain_text: str,
        strict_text: str,
    ) -> PairwiseJudgement:
        del trace_id
        self.judged.append((explain_text, strict_text, list(key_facts)))
        if question.startswith("What limitations"):
            return PairwiseJudgement(status="failed", fallback_reason="malformed_output")
        winner = "explain" if "method" in question else "tie"
        return PairwiseJudgement(
            status="complete", winner=winner, score=1.0 if winner == "explain" else 0.5
        )


@pytest.mark.asyncio
async def test_each_overview_question_is_answered_both_ways_and_judged_guided_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def text(answer: object) -> str:
        return cast(str, answer)

    monkeypatch.setattr(eval_guides, "answer_text", text)
    runtime = FakeRuntime()
    results = await eval_guides.compare_paper(runtime, "paper")

    assert [q for q, _ in runtime.asked[::2]] == list(eval_guides.OVERVIEW_QUESTIONS)
    assert [guided for _, guided in runtime.asked] == [True, False] * 4
    guided, unguided, facts = runtime.judged[0]
    assert guided.endswith("|True") and unguided.endswith("|False")
    assert facts == [
        "Cedar gates memory writes.",
        "It improves accuracy by 6 points.",
        "We release code.",
    ]
    assert [item["winner"] for item in results] == ["tie", "guided", "tie", ""]
    summary = eval_guides.summarize(results)
    assert summary == {
        "cases": 4,
        "judged": 3,
        "wins": 1,
        "ties": 2,
        "losses": 0,
        "win_rate": 0.667,
    }
    assert await eval_guides.compare_paper(runtime, "missing") == []
