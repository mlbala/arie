from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(*ranked_lists: Sequence[str], k: int = 60) -> dict[str, float]:
    """Combine ranked id lists by reciprocal rank fusion with the ARIE default k=60."""
    fused: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


__all__ = ["reciprocal_rank_fusion"]
