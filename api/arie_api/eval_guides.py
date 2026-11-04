"""Gate 3 (S14): do overview answers improve when the strict planner sees the paper guide?

For each paper (building its guide first if it has none) every overview question is answered
in strict mode twice, with and without the guide's section summaries, and the two answers are
judged pairwise, once each way round (``judge_pairwise``). The reference facts are the
abstract's sentences. Prints one JSON line per case, then one summary line:

    uv run python -m arie_api.eval_guides <paper_id> ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import msgspec
from arie_core.harness.citations import sentence_spans
from arie_core.harness.judge import COMPLETE, NO_ANSWER, PairwiseJudgement
from arie_core.harness.orchestrator import ResearchAnswer
from arie_core.harness.paper_eval import answer_text

from arie_api.runtime import CallContext, Document, ResearchRuntime, context

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ANSWER_SECONDS = 25.0
OVERVIEW_QUESTIONS = (
    "What is this paper about?",
    "Summarize the method of this paper.",
    "What are the main results of this paper?",
    "What limitations does this paper discuss?",
)
_FACTS = 6


class GuideComparisonRuntime(Protocol):
    """The parts of ``ResearchRuntime`` the comparison uses (a double in tests)."""

    async def get_paper(self, paper_id: str) -> Document | None: ...

    async def answer_paper_question(
        self, ctx: CallContext, paper: Document, question: str, *, guides: bool = True
    ) -> ResearchAnswer: ...

    async def judge_explain_vs_strict(
        self,
        trace_id: str,
        question: str,
        key_facts: Sequence[str],
        explain_text: str,
        strict_text: str,
    ) -> PairwiseJudgement: ...


def abstract_facts(abstract: str) -> list[str]:
    """The first sentences of the abstract: what a good overview of the paper states."""
    text = " ".join(abstract.split())
    return [text[a:b] for a, b in sentence_spans(text)][:_FACTS]


def summarize(results: Sequence[Document]) -> Document:
    """``win_rate`` of the guided answers over the judged cases (1 win, 0.5 tie, 0 loss)."""
    scores = [float(item["score"]) for item in results if item.get("score") is not None]
    return {
        "cases": len(results),
        "judged": len(scores),
        "wins": sum(score == 1.0 for score in scores),
        "ties": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "win_rate": round(sum(scores) / len(scores), 3) if scores else None,
    }


async def compare_paper(runtime: GuideComparisonRuntime, paper_id: str) -> list[Document]:
    """Answer each overview question with and without the guide and judge the pair."""
    paper = await runtime.get_paper(paper_id)
    if paper is None:
        return []
    facts = abstract_facts(str(paper.get("abstract", "")))
    results: list[Document] = []
    for number, question in enumerate(OVERVIEW_QUESTIONS):
        trace = f"eval-guide:{paper_id}:{number}"
        texts: dict[bool, str] = {}
        try:
            for guided in (True, False):
                async with asyncio.timeout(ANSWER_SECONDS):
                    run = await runtime.answer_paper_question(
                        context(f"{trace}:{guided}", ANSWER_SECONDS),
                        paper,
                        question,
                        guides=guided,
                    )
                texts[guided] = answer_text(run.answer) or NO_ANSWER
            judged = await runtime.judge_explain_vs_strict(
                trace, question, facts, texts[True], texts[False]
            )
        except Exception as exc:
            logger.warning("Guide comparison failed for %s", trace, exc_info=True)
            reason = "answer_timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
            judged = PairwiseJudgement(status="failed", fallback_reason=reason)
        # ``explain`` in the judgement is the guided answer, ``strict`` the unguided one.
        results.append(
            {
                "paper_id": paper_id,
                "question": question,
                "status": judged.status,
                "winner": {"explain": "guided", "strict": "unguided"}.get(
                    judged.winner, judged.winner
                ),
                "score": judged.score if judged.status == COMPLETE else None,
                "reasons": list(judged.reasons),
                "fallback_reason": judged.fallback_reason,
                "same_answer": texts.get(True) == texts.get(False),
            }
        )
    return results


async def run(paper_ids: Sequence[str]) -> Document:
    runtime = ResearchRuntime(ROOT)
    results: list[Document] = []
    try:
        await runtime.initialize()
        await asyncio.wait_for(runtime.models.warm(), 600.0)
        for paper_id in paper_ids:
            guide = await runtime.get_guide(paper_id)
            if guide is None:
                built = await runtime.build_guide(paper_id)
                status = built.status if built is not None else "not_indexed"
                print(json.dumps({"paper_id": paper_id, "guide": status}), flush=True)
                if built is None or built.guide is None:
                    continue
            for item in await compare_paper(runtime, paper_id):
                results.append(item)
                print(json.dumps(item), flush=True)
    finally:
        runtime.close()
    summary = summarize(results)
    summary["papers"] = len({item["paper_id"] for item in results})
    summary["same_answer"] = sum(bool(item["same_answer"]) for item in results)
    print(json.dumps({"summary": msgspec.to_builtins(summary)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Overview answers with and without guides.")
    parser.add_argument("paper_ids", nargs="+")
    args = parser.parse_args()
    asyncio.run(run(list(args.paper_ids)))


if __name__ == "__main__":
    main()
