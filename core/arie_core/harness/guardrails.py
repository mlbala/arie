from __future__ import annotations

import html
import re
import unicodedata

import msgspec


class InputDecision(msgspec.Struct, frozen=True):
    question: str
    allowed: bool = True
    code: str = ""
    message: str = ""


def inspect_question(question: str) -> InputDecision:
    """Reject explicit policy overrides and external reads before model or source access.

    This deterministic boundary is deliberately narrow: scientific discussions of
    prompts, attacks, comparative methods and quoted instructions remain valid questions.
    Semantic relevance and unsupported premises are evaluated against paper evidence later.
    """
    clean = unicodedata.normalize("NFKC", html.unescape(question))
    clean = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", clean)
    if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in clean):
        return InputDecision(
            "", False, "invalid_input", "Enter a readable question about this paper."
        )
    clean = " ".join(clean.split())
    if not clean or len(clean) > 2000 or len(clean.encode("utf-8")) > 6000:
        return InputDecision(
            "", False, "invalid_input", "Enter a shorter question about this paper."
        )
    # Match commands, not research questions such as “How does the paper prevent prompt injection?”
    override = re.search(
        r"(?:^|[.!?;:]\s*)(?:please\s+)?(?:ignore|disregard|override|forget|bypass)\s+"
        r"(?:(?:all|the|your|any|previous|prior|above|system|safety|source.only)\s+){0,5}"
        r"(?:instructions?|rules?|prompts?|guardrails?|restrictions?|citations?|evidence)\b"
        r"|(?:^|[.!?;:]\s*)(?:reveal|print|show|expose)\s+(?:your |the )?"
        r"(?:hidden |system |developer )+(?:prompt|instructions|secrets)\b"
        r"|(?:^|[.!?;:]\s*)(?:invent|fabricate|make up)\s+(?:the |some |any )?"
        r"(?:results?|citations?|evidence|facts?|data)\b"
        r"|<\|(?:system|developer|assistant)\|>|\[INST\]",
        clean,
        re.I,
    )
    if override:
        return InputDecision(
            clean,
            False,
            "instruction_override",
            "Ask about the paper's content. ARIE requires supporting passages for its answers.",
        )
    external = re.search(
        r"(?:^|[.!?;:]\s*|\band\s+)(?:(?:please|also|then)\s+)?"
        r"(?:search|browse|fetch|visit|open|use|consult|check)\s+(?:the |on |from )*"
        r"(?:web|internet|https?://|other papers?|another paper|external sources?)\b"
        r"|\b(?:across|from|in)\s+(?:all |my |the |entire )*(?:library|other papers)\b"
        r"|\bcompare\b.{0,100}\b(?:another|other) papers?\b"
        r"|\b(?:switch|change|set)\s+(?:the )?paper[_ ]id\b",
        clean,
        re.I,
    )
    if external:
        return InputDecision(
            clean,
            False,
            "outside_paper",
            "This chat reads the selected paper only. "
            "Ask about comparisons or findings reported in it.",
        )
    return InputDecision(clean)
