"""Reasons, implicit negatives, saved-answer facts and judge agreement (S15)."""

from __future__ import annotations

import pytest
from arie_core.harness.feedback import (
    ASKED_AGAIN,
    GATE_RATED,
    REASONS,
    REPEATED,
    SAID_WRONG,
    Rated,
    agreement,
    answer_facts,
    implicit_negative,
    judge_opinion,
)

PREVIOUS = "What accuracy does the method reach on the test set?"


def test_the_reasons_are_the_six_the_panel_offers() -> None:
    assert REASONS == (
        "wrong",
        "not_in_paper",
        "too_shallow",
        "too_long",
        "missing_citation",
        "did_not_answer",
    )


@pytest.mark.parametrize(
    "question",
    [
        "No, that's wrong.",
        "Wrong. The paper reports 92 percent.",
        "That is incorrect",
        "That’s not right, check page 4",
        "You are wrong about the dataset",
        "That doesn't answer my question",
        "You didn't answer what I asked",
        "This is not what I asked",
        "Nope, that isn't it",
    ],
)
def test_a_follow_up_that_says_the_answer_was_wrong_is_negative(question: str) -> None:
    assert implicit_negative(PREVIOUS, question) == SAID_WRONG


@pytest.mark.parametrize(
    "question",
    [
        "Try again",
        "Please try again.",
        "Can you explain the method again?",
        "Answer that again",
        "Again, what is the accuracy?",
        "Could you rephrase your answer",
        "One more time please",
    ],
)
def test_a_follow_up_that_asks_for_the_answer_again_is_negative(question: str) -> None:
    assert implicit_negative(PREVIOUS, question) == ASKED_AGAIN


@pytest.mark.parametrize(
    "question",
    [
        PREVIOUS,
        "What accuracy does the method reach on the test set",
        "what accuracy does this method reach on the test set?",
    ],
)
def test_repeating_the_question_with_high_word_overlap_is_negative(question: str) -> None:
    assert implicit_negative(PREVIOUS, question) == REPEATED


@pytest.mark.parametrize(
    "question",
    [
        "",
        "How was the test set built?",
        "Why is the baseline wrong?",
        "Is that incorrect for small models?",
        "What accuracy does the method reach on the test set and how does it compare to the "
        "baseline on the validation set?",
        "Does the model try again after a failed check?",
        "Nothing else to add, thanks",
        "Why?",
    ],
)
def test_other_follow_ups_are_not_negative(question: str) -> None:
    assert implicit_negative(PREVIOUS, question) is None


def test_a_short_repeated_question_is_not_enough_words_to_call() -> None:
    assert implicit_negative("Why?", "Why?") is None


def _answer(**data: object) -> dict[str, object]:
    return {"event": "answer", "data": data}


def test_answer_facts_come_from_the_answer_event() -> None:
    strict = answer_facts(
        [
            {"event": "progress", "data": {}},
            _answer(
                trace_id="abc",
                evaluation={"task": "overview", "trace": {"prompt_version": "qa.v1"}},
            ),
        ]
    )
    assert (strict.mode, strict.task, strict.prompt_version, strict.trace_id) == (
        "strict",
        "overview",
        "qa.v1",
        "abc",
    )
    explain = answer_facts(
        [_answer(prompt_version="explain.v1", trace_id="explain:1", evaluation={"task": "explain"})]
    )
    assert (explain.mode, explain.task, explain.prompt_version, explain.trace_id) == (
        "explain",
        "explain",
        "explain.v1",
        "explain:1",
    )


def test_answer_facts_are_empty_for_an_old_or_missing_answer() -> None:
    assert answer_facts([]).mode == ""
    old = answer_facts([_answer(evaluation={"task": "fact"})])
    assert (old.mode, old.task, old.prompt_version, old.trace_id) == ("strict", "fact", "", "")


def _judge(status: str, unsupported: object) -> dict[str, object]:
    return {"event": "judge", "data": {"status": status, "unsupported": unsupported}}


def test_the_judge_opinion_is_down_only_for_an_unsupported_sentence() -> None:
    assert judge_opinion([_judge("complete", 0)]) == "up"
    assert judge_opinion([_judge("complete", 2)]) == "down"
    assert judge_opinion([_judge("failed", 0)]) is None
    assert judge_opinion([_judge("complete", None)]) is None
    # Strict answers have no judge event.
    assert judge_opinion([_answer(evaluation={"task": "fact"})]) is None


def test_agreement_counts_only_judged_answers_overall_and_per_task() -> None:
    rows = [
        Rated("up", "explain", "up"),
        Rated("down", "explain", "down"),
        Rated("down", "explain", "up"),
        Rated("up", "explain", None),
        Rated("down", "fact", None),
    ]
    result = agreement(rows)
    assert (result.rated, result.judged, result.agreed) == (5, 3, 2)
    assert result.rate == 0.667 and not result.ready
    by_task = {row.task: row for row in result.by_task}
    assert (by_task["explain"].rated, by_task["explain"].judged, by_task["explain"].agreed) == (
        4,
        3,
        2,
    )
    assert by_task["fact"].rate is None and by_task["fact"].judged == 0


def test_agreement_with_nothing_judged_has_no_rate_and_the_gate_needs_thirty_ratings() -> None:
    empty = agreement([])
    assert (empty.rated, empty.rate, empty.ready, empty.by_task) == (0, None, False, ())
    enough = agreement(Rated("up", "", None) for _ in range(GATE_RATED))
    assert enough.ready and enough.rate is None and enough.by_task[0].task == "unknown"
