from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from arie_core.budget import BudgetHandle
from arie_core.contracts import ChunkRow, Claim, Evidence

from .builder import PromptCacheBuilder
from .prefetch import HybridPrefetcher
from .router import ZeroLLMRouter
from .tools import create_lexical_tool_session
from .verifier import EvidenceVerifier


@dataclass(frozen=True)
class GroundedAnswer:
    status: str
    route: str
    answer: str
    llm_tokens: int = 0
    evidence: list[Evidence] = field(default_factory=lambda: [])
    verified: bool = False


class GroundedQAHarness:
    """Simple, bounded Q&A harness that enforces the ARIE step order and K<=3 tool cap."""

    def __init__(
        self,
        *,
        paper_id: str,
        title: str,
        source: str,
        chunks: Sequence[ChunkRow | object],
        custom_metadata: dict[str, str] | None = None,
        outline: list[str] | None = None,
        budget: BudgetHandle | None = None,
        router: ZeroLLMRouter | None = None,
        builder: PromptCacheBuilder | None = None,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.paper_id = paper_id
        self.title = title
        self.source = source
        self.chunks = list(chunks)
        self.custom_metadata = custom_metadata or {}
        self.outline = list(outline or [])
        self.budget = (
            budget
            if budget is not None
            else BudgetHandle(
                max_tool_calls=3,
                max_tokens=12_000,
                max_wall_sec=25.0,
            )
        )
        self.router = router or ZeroLLMRouter()
        self.builder = builder or PromptCacheBuilder()
        self.verifier = verifier or EvidenceVerifier()
        self.prefetcher = HybridPrefetcher()

    @staticmethod
    def _paper_state_ok(state: str | None) -> bool:
        return state in {"INDEXED", "BRIEFED"}

    def _extract_text(self, chunk: object) -> str:
        if isinstance(chunk, ChunkRow):
            return chunk.text
        text_attr = getattr(chunk, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        if isinstance(chunk, Mapping):
            mapping = cast(Mapping[object, object], chunk)
            value = mapping.get("text")
            if isinstance(value, str):
                return value
        if chunk is None:
            return ""
        return str(cast(object, chunk))

    def _extract_chunk_id(self, chunk: object) -> str:
        if isinstance(chunk, ChunkRow):
            return chunk.chunk_id
        chunk_id = getattr(chunk, "chunk_id", None)
        if isinstance(chunk_id, str):
            return chunk_id
        if isinstance(chunk, Mapping):
            mapping = cast(Mapping[object, object], chunk)
            value = mapping.get("chunk_id")
            if isinstance(value, str):
                return value
        return "chunk"

    def answer(self, query: str, *, paper_state: str | None = "INDEXED") -> GroundedAnswer:
        if not self._paper_state_ok(paper_state):
            return GroundedAnswer(
                status="insufficient",
                route="abstain",
                answer="Paper state does not permit grounded Q&A.",
                llm_tokens=0,
            )
        if not query.strip():
            return GroundedAnswer(
                status="insufficient", route="abstain", answer="No query provided.", llm_tokens=0
            )

        route = self.router.route(query, paper_id=self.paper_id, paper_state=paper_state).route
        if route == "abstain":
            return GroundedAnswer(
                status="insufficient",
                route="abstain",
                answer=(
                    "Cross-paper or over-limit questions cannot be answered with zero LLM tokens."
                ),
                llm_tokens=0,
            )

        paper_chunks: list[ChunkRow] = [
            chunk
            for chunk in self.chunks
            if isinstance(chunk, ChunkRow) and chunk.paper_id == self.paper_id
        ]
        tool_session = create_lexical_tool_session(
            paper_id=self.paper_id,
            chunks=paper_chunks,
            tables=[],
            pages={},
        )
        selected = self.prefetcher.prefetch(query, paper_chunks)
        evidence: list[Evidence] = []
        for chunk in selected[:2]:
            text = self._extract_text(chunk)
            chunk_id = self._extract_chunk_id(chunk)
            item = Evidence(
                id=f"e:{chunk_id}",
                chunk_id=chunk_id,
                quote=text[:200],
            )
            evidence.append(item)

        use_tools = 0
        result = ""
        while use_tools < self.budget.max_tool_calls:
            if route == "locate_fact":
                tool = tool_session["search_paper"]
                result = tool(query)
            elif route == "method":
                tool = tool_session["read_section"]
                result = tool("methods")
            elif route == "results_tables":
                tool = tool_session["get_table"]
                result = tool("results")
            elif route == "limitations":
                tool = tool_session["read_section"]
                result = tool("limitations")
            else:
                tool = tool_session["search_paper"]
                result = tool(query)

            name = f"lexical_tool_{use_tools + 1}"
            self.budget.record_tool_call(name)
            use_tools += 1
            if result and "No matching" not in result and "not found" not in result:
                break
            if use_tools >= self.budget.max_tool_calls:
                break

        answer_text = "\n\n".join(
            result
            for result in [
                "Grounded answer:",
                *[self._extract_text(chunk) for chunk in selected[:2]],
            ]
            if result
        )
        claim = Claim(text=query, evidence=evidence)
        verified = (
            all(
                self.verifier.verify(claim, chunk_text=self._extract_text(chunk))
                for chunk in selected[:1]
            )
            if selected
            else False
        )

        return GroundedAnswer(
            status="answered" if verified or bool(answer_text) else "insufficient",
            route=route,
            answer=answer_text
            or "Insufficient evidence was available to answer this question reliably.",
            llm_tokens=0,
            evidence=evidence,
            verified=verified,
        )


__all__ = ["GroundedAnswer", "GroundedQAHarness"]
