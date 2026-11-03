from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Literal

import msgspec

from arie_core.harness.reader import is_overview, query_terms


class DialogueTurn(msgspec.Struct, frozen=True):
    """Untrusted conversation pointers; previous answer prose is never source evidence."""

    question: str
    source_ids: tuple[str, ...] = ()


class AnswerPlan(msgspec.Struct, frozen=True):
    question: str
    depth: Literal["brief", "standard", "detailed"]
    overview: bool
    follow_up: bool = False
    previous_source_ids: tuple[str, ...] = ()
    topic_question: str = ""
    task: str = "answer"
    queries: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    user_question: str = ""
    # True when the user's words ask for depth; ARIE's own upgrade to "detailed" is not.
    detail_requested: bool = False

    @property
    def max_claims(self) -> int:
        return {"brief": 2, "standard": 4, "detailed": 8}[self.depth]

    @property
    def extractive_bytes(self) -> int:
        # Byte counts conservatively bound output tokens when no tokenizer/model is available.
        return {"brief": 240, "standard": 420, "detailed": 700}[self.depth]

    @property
    def target_words(self) -> int:
        return {"brief": 40, "standard": 100, "detailed": 220}[self.depth]


def _follow_up(question: str) -> bool:
    if is_overview(question):
        return False
    return not query_terms(question) or bool(
        re.search(r"\b(?:continue|go on|tell me more|expand on (?:it|that|this))\b", question, re.I)
    )


def plan_answer(question: str, history: Sequence[DialogueTurn] = ()) -> AnswerPlan:
    """Resolve follow-ups from prior questions, never from assistant-generated facts."""
    clean = " ".join(html.unescape(question).split())
    follow_up = _follow_up(clean)
    previous = ""
    if follow_up:
        for turn in reversed(history[-4:]):
            if not _follow_up(turn.question):
                previous = turn.question[:1200]
                break
    depth: Literal["brief", "standard", "detailed"] = "standard"
    if re.search(r"\b(?:brief\w*|concise|short|one sentence|tl;?dr)\b", clean, re.I):
        depth = "brief"
    elif re.search(
        r"\b(?:explain|detail\w*|expand|elaborate|more|continue|in.depth|step.by.step)\b",
        clean,
        re.I,
    ):
        depth = "detailed"
    effective = f"{previous}\nFollow-up: {clean}" if previous else clean
    overview = is_overview(effective) or (follow_up and not previous)
    if overview and not is_overview(effective):
        effective = f"Explain this paper. {effective}"
    sources = tuple(dict.fromkeys(source for turn in history[-4:] for source in turn.source_ids))[
        :24
    ]
    return AnswerPlan(
        effective[:2000],
        depth,
        overview,
        bool(previous),
        sources if previous else (),
        (previous or clean)[:700],
        detail_requested=depth == "detailed",
    )
