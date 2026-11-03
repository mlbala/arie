from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence

from arie_core.contracts import Claim, Evidence

# Entailment model: (premise quote, hypothesis claim) pairs in, P(entailment) per pair out.
EntailmentCheck = Callable[[Sequence[tuple[str, str]]], Awaitable[Sequence[float]]]
# Calibrated on cross-encoder/nli-deberta-v3-xsmall: paraphrases score >= 0.9, flips <= 0.01.
MIN_ENTAILMENT = 0.5
# A direct single-point answer whose every claim scores at least this skips the LLM
# reviewer (ADR 0008); on the same model, faithful restatements score 0.92-0.99.
REVIEW_OPTIONAL_ENTAILMENT = 0.9
_CONTENT_STOP = frozenset(
    {
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "into",
        "than",
        "then",
        "their",
        "there",
        "which",
        "while",
        "where",
        "were",
        "have",
        "been",
        "being",
        "also",
        "such",
        "both",
        "each",
        "only",
        "very",
        "more",
        "most",
        "over",
        "under",
        "about",
        "paper",
        "work",
    }
)


_AUTHOR_VOICE = (
    (re.compile(r"\bWe\b"), "The authors"),
    (re.compile(r"\bwe\b"), "the authors"),
    (re.compile(r"\bOur\b"), "The authors'"),
    (re.compile(r"\bour\b"), "the authors'"),
    (re.compile(r"\bUs\b"), "The authors"),
    (re.compile(r"\bus\b"), "the authors"),
)


def attributed_premise(quote: str) -> str:
    """Restate a paper's first-person voice ("we", "our") as "the authors" for Tier 3.

    Only pronouns change; every number, entity, negation and qualifier stays as quoted.
    The displayed quote is never rewritten: this text is used only as the NLI premise.
    """
    for pattern, replacement in _AUTHOR_VOICE:
        quote = pattern.sub(replacement, quote)
    return quote


def strip_unverified_quotes(quotes: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for quote in quotes:
        candidate = quote.strip()
        if len(candidate) < 15:
            continue
        if (
            any(token in candidate.lower() for token in ["#", "http://", "https://"])
            and len(candidate) > 200
        ):
            continue
        cleaned.append(candidate)
    return cleaned


def normalize_verification_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00ad", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("ﬀ", "ff")
    text = text.replace("ﬃ", "ffi")
    text = text.replace("ﬄ", "ffl")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"-\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[*_`|]", "", text)


def preserves_anchors(text: str, quote: str) -> bool:
    """Reject changed numbers, lost qualifications and new absolutes before model review."""

    def numbers(value: str) -> set[str]:
        return set(re.findall(r"\d+(?:\.\d+)?", value.replace(",", "")))

    if not numbers(text) <= numbers(quote):
        return False
    groups = (
        r"\b(?:not|no|never|without|cannot|neither|fails?|unable|lacks?|removes?|dispensing)\b|n't",
        r"\b(?:may|might|could|suggests?|appears?|potentially|preliminary)\b",
        r"\b(?:only|limited|small|restricted)\b",
    )
    if any(re.search(group, quote, re.I) and not re.search(group, text, re.I) for group in groups):
        return False
    absolutes = r"\b(?:always|never|every|guarantees?|proves?|perfect|all datasets)\b"
    return not (re.search(absolutes, text, re.I) and not re.search(absolutes, quote, re.I))


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", value.lower())
        if len(token) > 3 and token not in _CONTENT_STOP
    }


class EvidenceVerifier:
    """Verification ladder for I2.

    Tier 1 exact substring, Tier 2 normalized match, then Tier 3 model entailment
    (``tier_3``) for model-written claims. ``verify`` runs Tiers 1-2 plus the lexical
    pre-filter; it suffices only for claims restated from their own quote.
    """

    @staticmethod
    def tier_1(chunk_content: str, quote: str) -> bool:
        if len(quote) < 15:
            return False
        return quote in chunk_content

    @staticmethod
    def tier_2(chunk_content: str, quote: str, *, chunk_row: object | None = None) -> bool:
        if chunk_row is not None:
            fields = ["section_path", "page_start", "char_start", "char_end"]
            for field in fields:
                if hasattr(chunk_row, field):
                    value = getattr(chunk_row, field)
                    if field.startswith("char_") and isinstance(value, int):
                        setattr(chunk_row, field, value)
        norm_chunk = normalize_verification_text(chunk_content)
        norm_quote = normalize_verification_text(quote)
        if len(norm_quote) < 15:
            return False
        if norm_quote in norm_chunk:
            return True

        quote_tokens = [
            token for token in re.findall(r"[a-z0-9]+", norm_quote.lower()) if len(token) > 4
        ]
        chunk_tokens = [
            token for token in re.findall(r"[a-z0-9]+", norm_chunk.lower()) if len(token) > 4
        ]
        if not quote_tokens or not chunk_tokens:
            return False

        overlap = sum(
            1
            for quote_token in quote_tokens
            if any(quote_token in chunk_token for chunk_token in chunk_tokens)
        )
        return overlap >= max(2, len(quote_tokens) // 2)

    @staticmethod
    def lexical_support(claim_text: str, evidence_quotes: Sequence[str]) -> bool:
        """Cheap pre-filter before Tier 3: the evidence must carry the claim's content.

        Passes when at least half of the claim's content words (or their 5-letter stems)
        occur in the quotes and no number, negation, hedge or absolute changed. It cannot
        detect polarity flips such as "performs well" -> "performs badly"; only the
        Tier 3 entailment model can, so passing this check is never sufficient alone.
        """
        claim_tokens = _content_tokens(claim_text)
        if not evidence_quotes or not claim_tokens:
            return False
        joined = " ".join(evidence_quotes)
        stems = {token[:5] for token in _content_tokens(joined)}
        covered = sum(1 for token in claim_tokens if token[:5] in stems)
        return covered * 2 >= len(claim_tokens) and preserves_anchors(claim_text, joined)

    @staticmethod
    async def tier_3(
        entail: EntailmentCheck,
        pairs: Sequence[tuple[str, str]],
        *,
        floor: float = MIN_ENTAILMENT,
    ) -> list[bool]:
        """Tier 3: keep a (quote, claim) pair only if the model finds the quote entails it.

        Failure modes: model errors and timeouts propagate so callers abstain (I2);
        a wrong number of scores or a score outside [0, 1] raises ``ValueError``.
        """
        return [score >= floor for score in await EvidenceVerifier.tier_3_scores(entail, pairs)]

    @staticmethod
    async def tier_3_scores(
        entail: EntailmentCheck, pairs: Sequence[tuple[str, str]]
    ) -> list[float]:
        """Tier 3 probabilities for (quote, claim) pairs, one per pair in input order.

        Each premise is first restated in the third person (``attributed_premise``): the
        NLI model does not know that a paper's "we" is "the paper" or "the authors" and
        rejected "The paper proposes X" against "We propose X" (0.02; 0.92 restated).
        Failure modes as for ``tier_3``.
        """
        if not pairs:
            return []
        restated = [(attributed_premise(quote), claim) for quote, claim in pairs]
        scores = [float(score) for score in await entail(restated)]
        if len(scores) != len(pairs) or any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("Entailment model returned malformed scores")
        return scores

    def verify(
        self, claim: Claim, *, chunk_text: str | None = None, chunk_row: object | None = None
    ) -> bool:
        quotes = [evidence.quote for evidence in claim.evidence]
        sanitized = strip_unverified_quotes(quotes)
        if not sanitized:
            return False
        if chunk_text is not None:
            for quote in sanitized:
                if self.tier_1(chunk_text, quote):
                    return self.lexical_support(claim.text, sanitized)
        for quote in sanitized:
            if self.tier_2(chunk_text or "", quote, chunk_row=chunk_row):
                return self.lexical_support(claim.text, sanitized)
        return False

    @staticmethod
    def verify_evidence(evidence: Evidence, *, chunk_text: str | None = None) -> bool:
        if chunk_text is None:
            return False
        return EvidenceVerifier.tier_1(chunk_text, evidence.quote) or EvidenceVerifier.tier_2(
            chunk_text, evidence.quote
        )


__all__ = [
    "MIN_ENTAILMENT",
    "REVIEW_OPTIONAL_ENTAILMENT",
    "EntailmentCheck",
    "attributed_premise",
    "EvidenceVerifier",
    "normalize_verification_text",
    "preserves_anchors",
    "strip_unverified_quotes",
]
