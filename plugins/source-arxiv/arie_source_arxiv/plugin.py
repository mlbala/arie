from __future__ import annotations

import re
from typing import Any

import httpx
from arie_core.plugin_api import (
    ScrapedPaperInput,
    normalize_scraper_output,
    plugin,
)

ALLOWED_CATEGORIES = (
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.MA",
    "cs.IR",
    "cs.CV",
    "cs.NE",
    "stat.ML",
)
ALLOWED_HOSTS = ("export.arxiv.org", "arxiv.org")


def _strip_tags(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<.*?>", "", value, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_arxiv_id(raw_id: str | None) -> str:
    if raw_id is None:
        return ""
    candidate = raw_id.strip()
    if candidate.startswith("arxiv:"):
        candidate = candidate[6:]
    candidate = candidate.removeprefix("https://arxiv.org/abs/")
    candidate = candidate.removeprefix("http://arxiv.org/abs/")
    candidate = candidate.removeprefix("https://arxiv.org/pdf/")
    candidate = candidate.removeprefix("http://arxiv.org/pdf/")
    return re.sub(r"v\d+$", "", candidate).strip()


@plugin(slot="source", name="arxiv")
class ArxivSourcePlugin:
    """Fetches arXiv entries limited to allowed research categories."""

    allowed_hosts = ALLOWED_HOSTS

    async def fetch(self, ctx: object | None, source_id: str) -> list[ScrapedPaperInput]:
        del ctx
        if source_id != "arxiv":
            return []
        params = {
            "search_query": "all:ai OR all:machine learning",
            "start": 0,
            "max_results": 10,
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get(
                "https://export.arxiv.org/api/query",
                params=params,
            )
            response.raise_for_status()
            text = response.text

        items: list[dict[str, Any]] = []
        for entry in re.findall(r"<entry>(.*?)</entry>", text, flags=re.DOTALL):
            category_nodes = re.findall(r'<category\s*term="([^"]+)"\s*/>', entry)
            if not category_nodes or not any(
                category in ALLOWED_CATEGORIES for category in category_nodes
            ):
                continue

            title_match = re.search(r"<title>(.*?)</title>", entry, flags=re.DOTALL)
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, flags=re.DOTALL)
            id_match = re.search(r"<id>(.*?)</id>", entry, flags=re.DOTALL)
            title = _strip_tags(title_match.group(1) if title_match else "")
            summary = _strip_tags(summary_match.group(1) if summary_match else "")
            raw_id = _normalize_arxiv_id(id_match.group(1) if id_match else "")
            paper_id = raw_id or "unknown"
            if not paper_id:
                continue
            primary_category = next(
                (category for category in category_nodes if category in ALLOWED_CATEGORIES),
                "",
            )
            items.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "source": "arxiv",
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                    "abstract": summary,
                    "published_at": None,
                    "metadata": {"primary_category": primary_category},
                }
            )
        return normalize_scraper_output(items, "arxiv")


__all__ = ["ArxivSourcePlugin"]
