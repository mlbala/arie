"""The per-answer trace: where an answer's time and tokens went, and why it fell back."""

from __future__ import annotations

import msgspec

from arie_core.contracts import TokenUsage

# Stages in pipeline order; ``total`` is the wall time of the whole answer and also
# covers work that has no stage of its own (guard, excerpt selection after a fallback).
STAGES = ("plan", "vector", "read", "rerank", "write", "tier3", "review")


class StageSeconds(msgspec.Struct, frozen=True):
    """Seconds spent per stage; a stage that did not run stays 0.0."""

    plan: float = 0.0
    vector: float = 0.0
    read: float = 0.0
    rerank: float = 0.0
    write: float = 0.0
    tier3: float = 0.0
    review: float = 0.0
    total: float = 0.0


class AnswerTrace(msgspec.Struct, frozen=True, kw_only=True):
    """Timing, token use, prompt version, model and fallback reason of one answer.

    ``tokens`` counts what the provider reported over every model call of the answer.
    ``prompt_version`` is empty when no prompt was sent. ``rerank_skipped_busy`` is true when
    reranking was skipped because the answer worker still ran a call its caller had abandoned
    (SPEC section 13.4). ``figure_hint`` is the figure number a strict abstention or excerpt
    fallback points the reader to (S24), or empty.
    """

    seconds: StageSeconds = StageSeconds()
    tokens: TokenUsage = TokenUsage(0, 0)
    prompt_version: str = ""
    model: str = ""
    mode: str = ""
    fallback_reason: str = ""
    rerank_skipped_busy: bool = False
    figure_hint: str = ""


__all__ = ["STAGES", "AnswerTrace", "StageSeconds"]
