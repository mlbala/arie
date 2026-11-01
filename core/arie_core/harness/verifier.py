from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from arie_core.contracts import Claim, Evidence


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


class EvidenceVerifier:
    """Verification ladder: exact byte match, normalized text match, and entailment check."""

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
    def semantic_entailment(claim_text: str, evidence_quotes: Sequence[str]) -> bool:
        if not evidence_quotes:
            return False
        claim_tokens = {
            token for token in re.findall(r"\w+", claim_text.lower()) if token and len(token) > 3
        }
        if not claim_tokens:
            return False
        for quote in evidence_quotes:
            quote_tokens = {
                token for token in re.findall(r"\w+", quote.lower()) if token and len(token) > 3
            }
            overlap = claim_tokens & quote_tokens
            if overlap:
                return True
        return False

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
                    return self.semantic_entailment(claim.text, sanitized)
        for quote in sanitized:
            if self.tier_2(chunk_text or "", quote, chunk_row=chunk_row):
                return self.semantic_entailment(claim.text, sanitized)
        return False

    @staticmethod
    def verify_evidence(evidence: Evidence, *, chunk_text: str | None = None) -> bool:
        if chunk_text is None:
            return False
        return EvidenceVerifier.tier_1(chunk_text, evidence.quote) or EvidenceVerifier.tier_2(
            chunk_text, evidence.quote
        )


__all__ = ["EvidenceVerifier", "normalize_verification_text", "strip_unverified_quotes"]
