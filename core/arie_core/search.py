from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher

_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "is",
        "are",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "you",
        "your",
    ]
)


def normalize_search(text: str) -> str:
    """Normalize case, entities, accents and punctuation without altering displayed titles."""
    folded = unicodedata.normalize("NFKD", html.unescape(text)).casefold()
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", plain))


def _match(term: str, words: set[str]) -> float:
    if term in words:
        return 1.0
    if len(term) >= 3 and any(word.startswith(term) for word in words):
        return 0.94
    if len(term) < 4:
        return 0.0
    best = max(
        (
            SequenceMatcher(None, term, word).ratio()
            for word in words
            if len(word) >= 4 and abs(len(word) - len(term)) <= 2
        ),
        default=0.0,
    )
    return best if best >= 0.78 else 0.0


def score_paper(query: str, title: str, abstract: str, topics: str = "") -> tuple[float, str]:
    """Rank partial or misspelled queries; reject weak title, topic and abstract matches."""
    normalized = normalize_search(query)
    terms = set(normalized.split()) - _STOP
    if not terms:
        terms = set(normalized.split())
    if not terms:
        return 0.0, ""
    normalized_title = normalize_search(title)
    title_words = set(normalized_title.split())
    title_scores = [_match(term, title_words) for term in terms]
    coverage = sum(score > 0 for score in title_scores) / len(terms)
    if coverage >= 0.65:
        strength = sum(title_scores) / len(terms)
        exact = normalized in normalized_title
        reason = "Title match" if all(score == 1 for score in title_scores) else "Similar title"
        return 60 + 30 * strength + (10 if exact else 0), reason
    for text, weight, reason in (
        (topics, 45, "Topic match"),
        (abstract[:5000], 30, "Abstract match"),
    ):
        words = set(normalize_search(text).split()) | title_words
        scores = [_match(term, words) for term in terms]
        if sum(score > 0 for score in scores) / len(terms) >= 0.8:
            return weight * sum(scores) / len(terms), reason
    return 0.0, ""
