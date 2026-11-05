"""The offline prompt optimizer (S17), run on real GEPA with a scripted model."""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from arie_api import optimize_prompts
from arie_api.optimize_prompts import (
    COMPONENT,
    ExplainAdapter,
    ExplainRollouts,
    Material,
    Report,
    load_items,
    optimize_prompt,
    plan,
)
from arie_api.runtime import ResearchRuntime
from arie_core.contracts import LLMDelta, LLMRequest, LLMResponse, TokenUsage
from arie_core.harness.explain import load_explain_prompts
from arie_core.harness.optimize import BudgetExhausted, CallBudget, OptCase, split_cases
from arie_core.harness.paper_eval import PaperEvalCase
from arie_core.harness.reader import EvidenceChunk
from arie_core.pipeline.pack import build_pack
from arie_core.plugin_api import LLM, Ctx

ROOT = Path(__file__).resolve().parents[2]
PAPER = "2611.00042"
FACT = "Cedar gates memory writes with a learned score"
BETTER = "BETTER-MARK: state the mechanism first, then cite it with a marker such as [c3]."
CONTEXT = 1_000_000


def chunk(number: int, text: str, page: int) -> EvidenceChunk:
    start = number * 1000
    return EvidenceChunk(PAPER, f"{PAPER}:{number}", text, page, start, start + len(text))


ROWS = [
    chunk(0, "## Abstract\nCedar is a retrieval model with a gated memory bank.", 1),
    chunk(1, "## Introduction\nRetrieval models forget context. Cedar keeps a memory bank.", 1),
    chunk(2, "## Methods\nCedar gates memory writes with a learned score.", 2),
    chunk(3, "The gate threshold is applied to every run.", 2),
]
MATERIAL = Material(build_pack(PAPER, "Cedar", "We study gated memory.", "hash:4", ROWS), ROWS)


def case(origin: str, number: int) -> OptCase:
    return OptCase(
        PAPER,
        PaperEvalCase(
            case_id=f"{origin}-{number}",
            question=f"How does Cedar decide what to store in setting {origin} {number}?",
            kind="fact",
            origin=origin,
            key_facts=[FACT],
        ),
    )


ITEMS = [case("generated", n) for n in range(4)] + [case("qasper", n) for n in range(2)]
TRAIN, HELD_OUT = split_cases(ITEMS, seed=17)


class ScriptedRoles:
    """One model for the writer, the judge and GEPA's reflection, told apart by their prompts.

    The writer states the key fact only when the rules it was given carry ``BETTER-MARK``.
    """

    def __init__(self, reflection: str) -> None:
        self.reflection = reflection
        self.requests: list[LLMRequest] = []

    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse:
        raise AssertionError("Explain mode streams")

    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]:
        del ctx
        self.requests.append(request)
        prompt = request.prompt
        if prompt.startswith("I gave a model these writer rules"):
            text = self.reflection
        elif "Role: support judge" in prompt:
            count = len(re.findall(r"^\d+\. ", prompt.split("Sentences to judge:")[1], re.M))
            text = "\n".join(f"{n} | supported | The chunk says it." for n in range(1, count + 1))
        elif "BETTER-MARK" in prompt:
            text = f"{FACT} [c2]."
        else:
            text = "Cedar is a retrieval model [c0]."
        yield LLMDelta(text)
        yield LLMDelta(finish=True, usage=TokenUsage(10, 10))

    def reflections(self) -> list[LLMRequest]:
        return [r for r in self.requests if r.prompt.startswith("I gave a model these writer")]


def rollouts(model: ScriptedRoles, cap: int) -> ExplainRollouts:
    return ExplainRollouts(
        load_explain_prompts(ROOT),
        {PAPER: MATERIAL},
        cast(LLM, model),
        CallBudget(cap),
        context_tokens=CONTEXT,
        pack_share=0.6,
    )


def sandbox(tmp_path: Path) -> Path:
    (tmp_path / "prompts").mkdir()
    shutil.copy(ROOT / "prompts" / "explain.v1.yaml", tmp_path / "prompts" / "explain.v1.yaml")
    return tmp_path


def better_rules() -> str:
    return load_explain_prompts(ROOT).writer_rules + "\n- " + BETTER


async def search(model: ScriptedRoles, root: Path, cap: int = 70) -> tuple[Report, ExplainRollouts]:
    rolled = rollouts(model, cap)
    report = await optimize_prompt(rolled, TRAIN, HELD_OUT, root=root, context_tokens=CONTEXT)
    return report, rolled


@pytest.mark.asyncio
async def test_every_call_is_counted_under_the_cap_and_the_held_out_cases_stay_out_of_the_search(
    tmp_path: Path,
) -> None:
    model = ScriptedRoles(f"```\n{better_rules()}\n```")
    report, rolled = await search(model, sandbox(tmp_path))

    assert 0 < report.calls_used <= report.cap == 70
    assert report.calls_used == rolled.budget.used == len(model.requests)
    assert len(TRAIN) == 4 and len(HELD_OUT) == 2
    assert {x.key for x in TRAIN}.isdisjoint({x.key for x in HELD_OUT})
    assert (report.train_cases, report.held_out_cases) == (4, 2)
    assert report.held_out_old["cases"] == report.held_out_new["cases"] == 2

    train_questions = [x.case.question for x in TRAIN]
    held_questions = [x.case.question for x in HELD_OUT]
    prompts = [r.prompt for r in model.requests]
    held = [i for i, text in enumerate(prompts) if any(q in text for q in held_questions)]
    searched = [i for i, text in enumerate(prompts) if any(q in text for q in train_questions)]
    assert held and searched and max(searched) < min(held)
    # What GEPA's reflection read comes from the train cases only, and it did read some.
    reflected = [r.prompt for r in model.reflections()]
    assert reflected and all(q not in text for text in reflected for q in held_questions)
    assert any(q in text for text in reflected for q in train_questions)
    # The held-out cases cost exactly the calls the run kept back for them: a writer and a
    # judge call for each of the 2 cases under each of the 2 prompts.
    assert len(held) == 4
    assert len(prompts) - min(held) == report.calls_reserved == 8


@pytest.mark.asyncio
async def test_the_candidate_is_written_beside_the_released_prompt_and_nothing_loads_it(
    tmp_path: Path,
) -> None:
    root = sandbox(tmp_path)
    released = (root / "prompts" / "explain.v1.yaml").read_bytes()
    report, _ = await search(ScriptedRoles(f"```\n{better_rules()}\n```"), root)

    assert report.candidate == "explain.v2" and report.candidate_path == "prompts/explain.v2.yaml"
    assert report.train_new > report.train_old
    assert report.held_out_new["score"] > report.held_out_old["score"]
    assert report.held_out_old["fact_recall_prose"] == 0.0
    assert report.held_out_new["fact_recall_prose"] == 1.0
    written = load_explain_prompts(root, 2)
    assert BETTER in written.writer_rules
    assert written.shared_header == load_explain_prompts(root).shared_header
    # The released file is unchanged and is still what every role loads.
    assert (root / "prompts" / "explain.v1.yaml").read_bytes() == released
    assert load_explain_prompts(root).version == "explain.v1"
    assert sorted(p.name for p in (root / "prompts").iterdir()) == [
        "explain.v1.yaml",
        "explain.v2.yaml",
    ]


@pytest.mark.asyncio
async def test_rules_that_cannot_be_used_cost_no_model_call_and_write_no_file(
    tmp_path: Path,
) -> None:
    root = sandbox(tmp_path)
    model = ScriptedRoles("```\nBe brief.\n```")
    report, _ = await search(model, root)

    assert report.candidate == "" and "nothing better" in report.note
    assert [p.name for p in (root / "prompts").iterdir()] == ["explain.v1.yaml"]
    assert report.held_out_new == report.held_out_old
    assert not any("Be brief." in r.prompt for r in model.requests if "Question:" in r.prompt)
    assert model.reflections()


@pytest.mark.asyncio
async def test_a_call_past_the_cap_ends_the_batch_instead_of_scoring_it() -> None:
    model = ScriptedRoles("")
    rolled = rollouts(model, 1)
    adapter = ExplainAdapter(rolled, asyncio.get_running_loop(), [])
    seed = load_explain_prompts(ROOT).writer_rules

    with pytest.raises(BudgetExhausted):
        await asyncio.to_thread(adapter.evaluate, TRAIN[:2], {COMPONENT: seed})

    # The writer's call was the one allowed; the judge's was refused before the model.
    assert len(model.requests) == 1 and rolled.budget.used == 1 and rolled.budget.refused


def test_a_cap_too_low_for_the_search_is_refused_before_any_call() -> None:
    assert plan(TRAIN, HELD_OUT, 70) == 8
    with pytest.raises(ValueError, match="--max-calls 20 is too low"):
        plan(TRAIN, HELD_OUT, 20)
    with pytest.raises(ValueError, match="at least 2"):
        plan(TRAIN, [], 500)
    with pytest.raises(SystemExit):
        optimize_prompts.parse_args(["explain", "--max-calls", "0"])
    with pytest.raises(SystemExit):
        optimize_prompts.parse_args(["explain", "--max-calls", "9", "--cases", "1"])


class FakeEvaluations:
    def chunks(self, paper: dict[str, Any]) -> list[EvidenceChunk]:
        return ROWS if paper["id"] == "p1" else []

    async def cases(self, paper_id: str) -> list[PaperEvalCase]:
        def make(case_id: str, kind: str, facts: list[str]) -> PaperEvalCase:
            return PaperEvalCase(
                case_id=case_id, question="Q?", kind=kind, origin="generated", key_facts=facts
            )

        return [make("a", "fact", ["x"]), make("b", "abstain", []), make("c", "fact", [])]

    async def feedback_cases(
        self, paper_id: str, rows: Sequence[EvidenceChunk]
    ) -> list[tuple[PaperEvalCase, tuple[str, ...]]]:
        del rows
        scored = PaperEvalCase(
            case_id="feedback-1", question="Q?", kind="fact", origin="feedback", key_facts=["y"]
        )
        review = PaperEvalCase(
            case_id="feedback-2", question="Q?", kind="feedback", origin="feedback"
        )
        return [(scored, ("wrong_facts",)), (review, ())]


class FakeRuntime:
    def __init__(self) -> None:
        self.evaluations = FakeEvaluations()
        self.db = SimpleNamespace(fetch=self.fetch)

    async def fetch(self, ctx: object, sql: str) -> list[dict[str, str]]:
        del ctx, sql
        return [{"paper_id": "p1"}, {"paper_id": "gone"}, {"paper_id": "pending"}]

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        status = {"p1": "INDEXED", "pending": "PARSING"}.get(paper_id)
        return {"id": paper_id, "status": status} if status else None


@pytest.mark.asyncio
async def test_only_scored_fact_cases_of_indexed_papers_load_with_the_readers_reasons() -> None:
    items = await load_items(cast(ResearchRuntime, FakeRuntime()))
    assert [(x.paper_id, x.case.case_id, x.reasons) for x in items] == [
        ("p1", "a", ()),
        ("p1", "feedback-1", ("wrong_facts",)),
    ]
