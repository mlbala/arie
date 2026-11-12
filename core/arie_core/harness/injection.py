"""Paper text that may instruct a model: the guard prefilter, the untrusted rule and chunk loading.

Paper text is untrusted (ADR 0013). Two checks mark a chunk as instructions rather than
research (ADR 0014):

- ``is_untrusted_instruction`` (``reader.py``): a narrow regex, applied everywhere, always.
- The ``guard`` slot: a local classifier scores the chunks ``needs_guard`` passes, at indexing.
  A chunk scored at or over the configured threshold loads with ``untrusted=True``.

``needs_guard`` is deliberately broad: it passes text that addresses a model or tells it to drop
its instructions, in English and the common languages, about 1% of library chunks. Only those
are scored, so the classifier's cost and its false positives on equations and tables (which
it scores as injections) stay off the other 99%.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import msgspec

from arie_core.harness.reader import EvidenceChunk, is_untrusted_instruction

# Words that tell a model to drop what it was told.
_OVERRIDE = (
    r"ignor\w*|disregard\w*|forget\w*|overrid\w*|bypass\w*|set aside|put aside|reset|cancel\w*"
    r"|suspend\w*|no longer apply|start over|pretend|olvid\w*|vergiss\w*|vergessen|oubli\w*"
    r"|negeer\w*|игнорир\w*|забудь\w*|忽略|忘记|無視|忘れ|무시|잊"
)
# What is being dropped.
_TARGET = (
    r"instructions?|prompts?|rules?|guidance|guidelines|directions?|orders?|consignes?"
    r"|context|configuration|everything|above|previous|prior|earlier|before|questions?"
    r"|instrucci\w*|Anweisung\w*|istruzion\w*|instru[cç]\w*|instructies|указани\w*|инструкц\w*"
    r"|指令|指示|规则|規則|지시|규칙"
)
# Text that speaks to a model, or tells "you" what to output.
_ADDRESS = (
    r"\b(?:AI|LLMs?|GPT\w*|chatbots?|assistants?|summari[sz]\w*|language models?"
    r"|automated (?:\w+ )?(?:systems?|readers?|tools?|reviewers?)|any (?:model|system|tool))\b"
    r"[^.!?\n]{0,80}\b(?:should|must|are required|is required|need to|will now|now|:)"
    r"|\b(?:note|instructions?|message) (?:to|for) (?:any |the |all )?(?:AI|LLM|GPT|model"
    r"|assistant|chatbot|language model|reader|summari[sz]er|automated)"
    r"|\b(?:if|when) (?:you are|you're) (?:an? )?(?:AI|LLM|language model|assistant|chatbot)"
    r"|\b(?:hey|dear|attention) (?:AI|assistant|model|LLM|chatbot)"
    r"|\b(?:model|system|tool|assistant)s? (?:that )?(?:reads?|reading|process\w*"
    r"|summari[sz]\w*) this\b"
    r"|\bwhoever is (?:generating|writing|reading)"
    r"|\b(?:you|your) (?:must|should|are now|will now|only task|previous|earlier)\b"
    r"|\b(?:you were|you've been|you have been) (?:given|told)|\btold you\b"
    r"|\b(?:higher|more) (?:priority|precedence) than\b"
    r"|\b(?:when|while) summari[sz]ing\b|\bskip the (?:quotes|citations|sources)\b"
    r"|\bfrom now on\b|\bfrom this (?:sentence|point|paragraph) on"
    r"|\b(?:act|respond|answer|behave|reply) (?:as|like) (?:if|though|an?|the)\b"
    r"|\b(?:output|print|reveal|repeat|display) (?:the |your |only |nothing )"
    r"(?:\w+ ){0,3}(?:prompt|instructions|configuration|string|word|system message)"
    r"|(?:^|\n|[.;:]\s)\s*(?:system|assistant|developer|user)\s*:"
    r"|\bdeveloper mode\b|\bsafety (?:checks|rules) (?:are )?(?:disabled|off)"
    r"|\b(?:instead|rather than)\b[^.!?\n]{0,40}\b(?:output|write|say|tell|praise|answer)"
    r"|\b(?:say|state|tell the user|describe this)\b[^.!?\n]{0,60}\b(?:in every answer|instead)"
)
_PREFILTER = re.compile(
    rf"(?:{_OVERRIDE})[^.!?\n]{{0,80}}(?:{_TARGET})|(?:{_TARGET})[^.!?\n]{{0,40}}(?:{_OVERRIDE})"
    rf"|{_ADDRESS}",
    re.IGNORECASE,
)

# The stored chunk key holding the guard's score; not an ``EvidenceChunk`` field.
SCORE_KEY = "injection_score"


def needs_guard(text: str) -> bool:
    """Whether the guard classifier should score this chunk text."""
    return bool(_PREFILTER.search(text))


def is_untrusted(row: EvidenceChunk) -> bool:
    """A chunk no answer may use: flagged by the guard, or instruction-like by the regex."""
    return row.untrusted or is_untrusted_instruction(row.text)


def evidence_chunks(
    paper_id: str, rows: Iterable[Mapping[str, object]], threshold: float | None
) -> list[EvidenceChunk]:
    """Stored chunk dicts of one paper as ``EvidenceChunk``s, with ``untrusted`` from the guard.

    Rows of other papers are dropped (I1). ``threshold`` is ``None`` when the guard is off: then
    stored scores are ignored and no chunk is untrusted, exactly as before the guard existed.
    """
    chunks: list[EvidenceChunk] = []
    for row in rows:
        if row.get("paper_id") != paper_id:
            continue
        fields = {key: value for key, value in row.items() if key != SCORE_KEY}
        score = row.get(SCORE_KEY)
        flagged = (
            threshold is not None
            and isinstance(score, int | float)
            and not isinstance(score, bool)
            and float(score) >= threshold
        )
        chunks.append(msgspec.convert({**fields, "untrusted": flagged}, EvidenceChunk))
    return chunks


__all__ = ["SCORE_KEY", "evidence_chunks", "is_untrusted", "needs_guard"]
