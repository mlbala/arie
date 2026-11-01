from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from arie_core.contracts import ChunkRow


class ReciprocalRankFusion:
    """Combine ranks from multiple retrieval sources via RRF with the ARIE default k=60."""

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, *ranked_lists: Sequence[str]) -> dict[str, float]:
        aggregated: dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, item_id in enumerate(ranked_list, start=1):
                aggregated[item_id] = aggregated.get(item_id, 0.0) + 1.0 / (self.k + rank)
        return aggregated


def reciprocal_rank_fusion(*ranked_lists: Sequence[str], k: int = 60) -> dict[str, float]:
    return ReciprocalRankFusion(k=k).fuse(*ranked_lists)


class HybridPrefetcher:
    """Fuse BM25 and vector-exact retrieval results with RRF and normalize to a small chunk set."""

    def __init__(self, *, rrk: int = 60, max_chunks: int = 6, max_tokens: int = 2500) -> None:
        self.rrk = rrk
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens

    @staticmethod
    def _bm25(query: str, text: str) -> float:
        terms = [piece.lower() for piece in re.findall(r"\w+", query) if piece]
        if not terms or not text:
            return 0.0
        corpus_size = 1
        doc_count = 1
        avgdl = max(len(text.split()), 1)
        score = 0.0
        for term in terms:
            # Simple BM25 approximation for an in-memory retrieval prototype.
            tf = text.lower().count(term)
            if tf == 0:
                continue
            idf = math.log((corpus_size * 1.0 + 1.0) / (doc_count + 1.0) + 1.0)
            numerator = (1.2 + 1.0) * tf
            denominator = 1.2 * (1.0 - 0.75 + 0.75 * (len(text.split()) / avgdl)) + tf
            score += idf * (numerator / denominator)
        return score

    def prefetch(
        self,
        query: str,
        chunks: Sequence[ChunkRow],
        *,
        vector_scores: Mapping[str, float] | None = None,
    ) -> list[ChunkRow]:
        vector_scores = vector_scores or {}
        if not chunks:
            return []

        bm25_ranked: list[str] = []
        vector_ranked: list[str] = []
        bm25_scores: dict[str, float] = {
            chunk.chunk_id: self._bm25(query, chunk.text) for chunk in chunks if chunk.text
        }
        vector_values = {
            chunk.chunk_id: float(vector_scores.get(chunk.chunk_id, 0.0)) for chunk in chunks
        }

        for chunk in sorted(
            chunks, key=lambda item: bm25_scores.get(item.chunk_id, 0.0), reverse=True
        ):
            bm25_ranked.append(chunk.chunk_id)
        for chunk in sorted(
            chunks, key=lambda item: vector_values.get(item.chunk_id, 0.0), reverse=True
        ):
            vector_ranked.append(chunk.chunk_id)

        fused = reciprocal_rank_fusion(bm25_ranked, vector_ranked, k=self.rrk)
        selected: list[ChunkRow] = []
        used_tokens = 0
        for chunk in sorted(chunks, key=lambda item: fused.get(item.chunk_id, 0.0), reverse=True):
            chunk_tokens = max(int(chunk.token_count or (len(chunk.text.split()) * 0.75)), 1)
            if len(selected) >= self.max_chunks:
                break
            if used_tokens + chunk_tokens > self.max_tokens:
                continue
            selected.append(chunk)
            used_tokens += chunk_tokens
        return selected


__all__ = ["HybridPrefetcher", "ReciprocalRankFusion", "reciprocal_rank_fusion"]
