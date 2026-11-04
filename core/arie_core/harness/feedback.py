"""What a reader's ratings tell us: reasons, implicit negatives and judge agreement (S15).

A thumbs-down finds a failure; it is not a quality score. This module holds the pure parts of
the loop: the reasons a reader can give, the rule that reads a follow-up question as a
negative rating of the answer before it, the facts a saved answer records about itself, and
the agreement between the judge's verdict and the reader's rating. Storage is in
``arie_api.paper_eval`` and ``arie_api.runtime.chat``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import msgspec

# Why a reader marked an answer down. The order is the order the web shows them in.
REASONS = (
    "wrong",
    "not_in_paper",
    "too_shallow",
    "too_long",
    "missing_citation",
    "did_not_answer",
)

# What an implicit negative can be: the follow-up said the answer was wrong, asked for it
# again, or asked the previous question again.
SAID_WRONG = "said_wrong"
ASKED_AGAIN = "asked_again"
REPEATED = "repeated_question"

# Gate 4 (docs/explain-mode/README.md) is read once this many answers are rated.
GATE_RATED = 30

# Share of words two questions must have in common (union) to count as the same question.
_REPEAT_OVERLAP = 0.7
_REPEAT_MIN_WORDS = 3

# A reader who opens with a question word is asking about the paper, not judging the answer.
_ASKING = re.compile(
    r"^(why|how|what|when|where|which|who|is|are|does|do|did|can|could|should|would)\b"
)
_WRONG = re.compile(
    r"^(wrong|incorrect)\b"
    r"|^(no|nope|nah)\s*[,.!]"
    r"|^(no|nope|nah)\s+(that|this|it|you)\b"
    r"|\b(that|this|it|the answer|your answer)(?:'s| is| was| seems| looks)"
    r" (?:not (?:right|correct|true)|wrong|incorrect)\b"
    r"|\byou(?:'re| are) (?:wrong|mistaken|incorrect)\b"
    r"|\b(?:that|this|it|you|answer)\s+(?:doesn't|does not|didn't|did not|isn't|is not)\s+"
    r"(?:answer|address|help|what i)\b"
    r"|\bnot what i (?:asked|meant|wanted)\b"
)
_AGAIN = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would) you\s+)?"
    r"(?:(?:try|answer|ask|say|explain|describe|summari[sz]e|tell me)\b(?:\W+\w+){0,3}?\W+again\b"
    r"|redo (?:that|it|this)\b|retry\b|rephrase (?:that|it|your answer|the answer)\b"
    r"|once more\b|one more time\b)"
    r"|^again\b"
)


def _normal(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normal(text)))


def implicit_negative(previous: str, question: str) -> str | None:
    """Name the signal that ``question`` gives about the answer to ``previous``, or ``None``.

    ``said_wrong``: the follow-up says the answer was wrong or did not answer. ``asked_again``:
    it asks for the answer again. ``repeated_question``: it repeats the previous question
    (word overlap of at least 0.7 over the union, both with 3 or more words). A follow-up that
    only extends the question is not a negative. The rules are narrow on purpose: a missed
    negative costs a signal, a false one pollutes the feedback list.
    """
    text = _normal(question)
    if not text:
        return None
    if _WRONG.search(text) and not _ASKING.match(text):
        return SAID_WRONG
    if _AGAIN.search(text):
        return ASKED_AGAIN
    before, after = _words(previous), _words(question)
    enough = min(len(before), len(after)) >= _REPEAT_MIN_WORDS
    if enough and len(before & after) / len(before | after) >= _REPEAT_OVERLAP:
        return REPEATED
    return None


class AnswerFacts(msgspec.Struct, frozen=True):
    """What a saved answer says about how it was made, read from its saved events.

    ``mode`` is the answer path (``strict`` or ``explain``), ``task`` the planner's task for a
    strict answer and ``explain`` for an explanation. Each is empty when the answer predates
    the field: ``trace_id`` was added in S15.
    """

    mode: str = ""
    task: str = ""
    prompt_version: str = ""
    trace_id: str = ""


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def answer_facts(events: Sequence[Mapping[str, object]]) -> AnswerFacts:
    """Read the mode, task, prompt version and trace id from a turn's ``answer`` event."""
    for item in events:
        if item.get("event") != "answer":
            continue
        data = _mapping(item.get("data"))
        evaluation = _mapping(data.get("evaluation"))
        task = _text(evaluation.get("task"))
        trace = _mapping(evaluation.get("trace"))
        return AnswerFacts(
            mode=("explain" if task == "explain" else "strict") if task else "",
            task=task,
            prompt_version=_text(data.get("prompt_version")) or _text(trace.get("prompt_version")),
            trace_id=_text(data.get("trace_id")),
        )
    return AnswerFacts()


def judge_opinion(events: Sequence[Mapping[str, object]]) -> str | None:
    """What the support judge thought of a saved answer: ``up``, ``down`` or ``None``.

    ``down`` when it judged the answer (status ``complete``) and found at least one cited
    sentence ``unsupported``; ``up`` when it found none (``partial`` counts as support, as in
    ``citation_support``). ``None`` when no judgement ran or it failed, which is every strict
    answer: only explain answers have a ``judge`` event.
    """
    for item in reversed(events):
        if item.get("event") != "judge":
            continue
        data = _mapping(item.get("data"))
        unsupported = data.get("unsupported")
        if data.get("status") != "complete" or not isinstance(unsupported, int):
            return None
        return "down" if unsupported else "up"
    return None


class Rated(msgspec.Struct, frozen=True):
    """One explicitly rated answer: the reader's rating, its task and the judge's opinion."""

    rating: str
    task: str
    opinion: str | None


class TaskAgreement(msgspec.Struct, frozen=True):
    task: str
    rated: int
    judged: int
    agreed: int
    rate: float | None


class Agreement(msgspec.Struct, frozen=True):
    """How often the judge's verdict matched the reader's rating.

    ``rated`` counts every explicit rating, ``judged`` those the judge also gave a verdict on,
    ``agreed`` those where the two match. ``rate`` is ``agreed / judged`` (``None`` with no
    judged answer). ``ready`` is true once ``GATE_RATED`` answers are rated.
    """

    rated: int
    judged: int
    agreed: int
    rate: float | None
    ready: bool
    by_task: tuple[TaskAgreement, ...]


def _rate(agreed: int, judged: int) -> float | None:
    return round(agreed / judged, 3) if judged else None


def agreement(items: Iterable[Rated]) -> Agreement:
    """Compare the judge's opinion with each reader rating, overall and per task."""
    totals: dict[str, list[int]] = {}
    for item in items:
        counts = totals.setdefault(item.task or "unknown", [0, 0, 0])
        counts[0] += 1
        if item.opinion is not None:
            counts[1] += 1
            counts[2] += int(item.opinion == item.rating)
    by_task = tuple(
        TaskAgreement(task, rated, judged, agreed, _rate(agreed, judged))
        for task, (rated, judged, agreed) in sorted(totals.items())
    )
    rated = sum(row.rated for row in by_task)
    judged = sum(row.judged for row in by_task)
    agreed = sum(row.agreed for row in by_task)
    return Agreement(rated, judged, agreed, _rate(agreed, judged), rated >= GATE_RATED, by_task)


__all__ = [
    "ASKED_AGAIN",
    "GATE_RATED",
    "REASONS",
    "REPEATED",
    "SAID_WRONG",
    "Agreement",
    "AnswerFacts",
    "Rated",
    "TaskAgreement",
    "agreement",
    "answer_facts",
    "implicit_negative",
    "judge_opinion",
]
