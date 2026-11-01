from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    llm_tokens: int = 0
    reason: str = ""


class ZeroLLMRouter:
    """Select the likely evidence-seeking route without any model call."""

    def __init__(self, prototype_vocab: Mapping[str, Sequence[str]] | None = None) -> None:
        self._prototype_vocab: Mapping[str, Sequence[str]] = prototype_vocab or {
            "locate_fact": ("what", "which", "who", "where", "when", "quote", "evidence"),
            "method": ("method", "approach", "algorithm", "procedure", "setup", "training"),
            "results_tables": ("result", "table", "figure", "metric", "accuracy", "score"),
            "limitations": ("limit", "limitation", "failure", "weakness", "bias", "challenge"),
            "overview": ("summary", "overview", "describe", "explain", "context"),
        }

    @staticmethod
    def _token_count(value: str) -> int:
        return len(re.findall(r"\S+", value))

    @staticmethod
    def _prototype_score(query: str, seed_terms: Sequence[str]) -> int:
        lowered = query.lower()
        score = 0
        for term in seed_terms:
            if term in lowered:
                score += 1
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                score += 1
        return score

    def route(
        self, query: str, *, paper_id: str | None = None, paper_state: str | None = None
    ) -> RouteDecision:
        normalized = query.strip()
        if not normalized:
            return RouteDecision(route="abstain", llm_tokens=0, reason="empty_query")

        if self._token_count(normalized) > 2000:
            return RouteDecision(route="abstain", llm_tokens=0, reason="query_too_large")

        if paper_state is not None and paper_state not in {"INDEXED", "BRIEFED"}:
            return RouteDecision(route="abstain", llm_tokens=0, reason="paper_not_ready")

        lowered = normalized.lower()
        if bool(re.search(r"\b(?:compare|contrast|across|between)\b", lowered)):
            return RouteDecision(route="abstain", llm_tokens=0, reason="cross_paper_question")

        scored_routes = {
            name: self._prototype_score(lowered, tuple(terms))
            for name, terms in self._prototype_vocab.items()
        }
        if "result" in lowered or "main result" in lowered or "results" in lowered:
            scored_routes["results_tables"] += 3
        if "what" in lowered and ("main result" in lowered or "result" in lowered):
            scored_routes["locate_fact"] -= 1

        best_route = max(
            scored_routes.items(),
            key=lambda item: item[1],
            default=("overview", 0),
        )[0]
        if scored_routes[best_route] == 0:
            best_route = "overview"
        return RouteDecision(route=best_route, llm_tokens=0, reason=f"prototype_match:{best_route}")


__all__ = ["RouteDecision", "ZeroLLMRouter"]
