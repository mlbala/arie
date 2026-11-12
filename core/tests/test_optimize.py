"""The offline prompt optimizer's pure parts (S17): call cap, split, metric, feedback, files."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import msgspec
import pytest
import yaml
from arie_api.runtime.base import CallContext
from arie_core.budget import BudgetHandle
from arie_core.contracts import LLMRequest
from arie_core.harness.citations import check_answer
from arie_core.harness.explain import load_explain_prompts
from arie_core.harness.judge import SentenceVerdict, SupportJudgement
from arie_core.harness.optimize import (
    EXCERPT_PENALTY,
    BudgetExhausted,
    CallBudget,
    CountingLLM,
    OptCase,
    Rollout,
    check_rules,
    feedback_text,
    metric,
    next_version,
    pick_cases,
    render_candidate,
    split_cases,
    summarize_rollouts,
)
from arie_core.harness.paper_eval import PaperEvalCase, PaperEvalResult
from arie_core.plugin_api import LLM
from conftest import ScriptedModel

ROOT = Path(__file__).resolve().parents[2]


def ctx() -> CallContext:
    return CallContext(time.monotonic() + 60, "optimize-test", BudgetHandle(max_wall_sec=60))


def item(paper: str, number: int, origin: str, reasons: tuple[str, ...] = ()) -> OptCase:
    case = PaperEvalCase(
        case_id=f"{paper}-{origin}-{number}",
        question=f"Question {number} of {paper}?",
        kind="fact",
        origin=origin,
        key_facts=[f"fact number {number} of {paper}"],
    )
    return OptCase(paper, case, reasons)


def result(**changes: object) -> PaperEvalResult:
    fields: dict[str, object] = {
        "case_id": "c",
        "question": "q",
        "kind": "fact",
        "passed": True,
        "grounded": True,
        "evidence_hit": True,
        "fact_recall": 0.0,
        "abstained": False,
        "mode": "explain",
        "claims": 1,
        "model_calls": 1,
        "seconds": 1.0,
        "answer_written": True,
    }
    return msgspec.convert({**fields, **changes}, PaperEvalResult)


@pytest.mark.asyncio
async def test_the_cap_refuses_the_call_past_it_before_the_model_is_reached() -> None:
    budget = CallBudget(2)
    model = ScriptedModel([{"a": 1}, {"a": 2}], streams=[["one"], ["two"]])
    llm = CountingLLM(cast(LLM, model), budget)
    request = LLMRequest(prompt="p")

    await llm.complete(ctx(), request)
    assert [delta.text async for delta in llm.stream(ctx(), request)][:1] == ["one"]
    assert budget.used == 2 and budget.left == 0 and not budget.refused

    with pytest.raises(BudgetExhausted):
        await llm.complete(ctx(), request)
    with pytest.raises(BudgetExhausted):
        _ = [delta async for delta in llm.stream(ctx(), request)]
    assert len(model.requests) == 2
    assert budget.used == 2 and budget.refused
    with pytest.raises(ValueError, match="at least 1"):
        CallBudget(0)


def test_the_split_is_fixed_by_the_seed_disjoint_and_keeps_every_origin() -> None:
    items = [item("p1", n, "generated") for n in range(6)]
    items += [item("p2", n, "qasper") for n in range(5)] + [item("p1", 9, "feedback")]
    train, held_out = split_cases(items, seed=17)
    again = split_cases(list(reversed(items)), seed=17)

    assert (train, held_out) == again
    assert {x.key for x in train}.isdisjoint({x.key for x in held_out})
    assert len(train) + len(held_out) == len(items)
    assert {x.case.origin for x in held_out} == {"generated", "qasper"}
    assert {x.case.origin for x in train} == {"generated", "qasper", "feedback"}
    assert split_cases(items, seed=18) != (train, held_out)
    assert split_cases([], seed=1) == ([], [])


def test_picking_cases_takes_each_origin_in_turn_and_the_same_ones_per_seed() -> None:
    items = [item("p", n, "generated") for n in range(5)]
    items += [item("q", n, "qasper") for n in range(5)]
    picked = pick_cases(items, 4, seed=3)
    assert [x.case.origin for x in picked] == ["generated", "qasper", "generated", "qasper"]
    assert pick_cases(items, 4, seed=3) == picked
    assert len(pick_cases(items, None, seed=3)) == 10 and len(pick_cases(items, 99, seed=3)) == 10


def test_the_metric_is_recall_plus_support_minus_the_excerpt_penalty() -> None:
    assert metric(result(fact_recall_prose=1.0, citation_support=0.5)) == 1.5
    # The judge failed: the deterministic valid rate stands in for its figure.
    assert metric(result(fact_recall_prose=0.5, citation_valid_rate=0.8)) == 1.3
    # Cites nothing at all.
    assert metric(result(fact_recall_prose=0.5)) == 0.5
    # No written answer: the reader would have seen excerpts.
    assert metric(result(answer_written=False)) == -EXCERPT_PENALTY


def test_feedback_names_the_missed_fact_the_flag_the_judge_and_the_readers_reasons() -> None:
    text = "Cedar gates writes with a score of 0.9 [c1]. It also helps [c2]."
    checks = check_answer(text, {1: "Cedar gates writes with a learned score.", 2: "It helps."})
    second = checks.sentences[1]
    judgement = SupportJudgement(
        status="complete",
        verdicts=(SentenceVerdict(2, second.start, second.end, "partial", "Only half is said."),),
    )
    case = OptCase(
        "paper-9",
        PaperEvalCase(
            case_id="c",
            question="How?",
            kind="fact",
            origin="feedback",
            key_facts=["fact number 1 of paper-9"],
            note="It missed the point.",
        ),
        ("wrong_facts", "too_long"),
    )
    note = feedback_text(case, result(), text, checks, judgement)

    assert "did not state: fact number 1 of paper-9" in note
    assert "number_not_in_source" in note and "0.9" in note
    assert "found it partial" in note and "Only half is said." in note
    assert "wrong_facts, too_long" in note and "It missed the point." in note
    assert len(note) <= 1500
    gone = result(answer_written=False, fallback_reason="first_token_timeout")
    unwritten = feedback_text(case, gone, "", checks, judgement)
    assert unwritten.startswith("No written answer (first_token_timeout)")


def test_rules_must_keep_the_marker_example_stay_short_and_not_quote_the_answer() -> None:
    good = "Role: writer.\n- Put a marker such as [c3] after every sentence."
    assert check_rules(good) == ""
    assert check_rules("  ") == "empty"
    assert check_rules(good + "x" * 4000) == "too_long"
    assert check_rules("Be brief and cite the paper.") == "no_marker_example"
    quoted = good + " The gate threshold is 0.4."
    assert check_rules(quoted, ["the gate  threshold is 0.4"]) == "quotes_ground_truth"
    # Short key facts are too common in prose to guard.
    assert check_rules(good + " Use 0.4.", ["0.4"]) == ""


def test_summaries_average_the_cases_and_count_the_excerpt_fallbacks() -> None:
    def rollout(score: float, **changes: object) -> Rollout:
        return Rollout(item("p", 1, "generated"), "", result(**changes), score, "")

    rollouts = [
        rollout(1.5, fact_recall_prose=1.0, citation_support=0.5),
        rollout(-1.0, answer_written=False),
    ]
    assert summarize_rollouts(rollouts) == {
        "cases": 2,
        "score": 0.25,
        "fact_recall_prose": 0.5,
        "citation_support": 0.5,
        "excerpt_fallbacks": 1,
    }
    assert summarize_rollouts([]) == {"cases": 0}


def test_a_candidate_file_is_the_seed_with_new_rules_and_nothing_loads_it_by_default(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    original = (ROOT / "prompts" / "explain.v2.yaml").read_text()
    (prompts / "explain.v2.yaml").write_text(original)
    seed = load_explain_prompts(tmp_path)
    assert next_version(prompts, "explain") == 3

    rules = "Role: writer.\n- Cite as [c3]: always.\n\n  - Indented # not a comment"
    (prompts / "explain.v3.yaml").write_text(render_candidate(seed, rules, 3, "Train 1 -> 2.\nok"))

    candidate = load_explain_prompts(tmp_path, 3)
    assert candidate.writer_rules == rules and candidate.version == "explain.v3"
    assert candidate.shared_header == seed.shared_header
    assert candidate.depth_words == seed.depth_words
    assert "Train 1 -> 2. ok" in (prompts / "explain.v3.yaml").read_text()
    # The released prompt is untouched and is still the one every role loads.
    assert (prompts / "explain.v2.yaml").read_text() == original
    assert load_explain_prompts(tmp_path).version == "explain.v2"
    assert next_version(prompts, "explain") == 4


def test_a_prompt_file_whose_version_disagrees_with_its_name_fails_closed(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "explain.v2.yaml").write_text((ROOT / "prompts" / "explain.v1.yaml").read_text())
    with pytest.raises(ValueError, match="says version 1"):
        load_explain_prompts(tmp_path, 2)
    assert yaml.safe_load((prompts / "explain.v2.yaml").read_text())["version"] == 1
