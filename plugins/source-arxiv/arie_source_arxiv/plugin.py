from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import httpx
from arie_core.plugin_api import Ctx, ScrapedPaperInput, SourcePage, plugin, wait_for_source_host

ALLOWED_CATEGORIES = ("cs.AI", "cs.CL", "cs.LG", "cs.MA", "cs.IR", "cs.CV", "cs.NE", "stat.ML")
ALLOWED_HOSTS = ("export.arxiv.org", "arxiv.org")
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


@plugin(slot="source", name="arxiv")
class ArxivSourcePlugin:
    """Page dated Atom records; reject HTTP/XML errors and honor arXiv's request spacing."""

    allowed_hosts = ALLOWED_HOSTS

    async def fetch(self, ctx: Ctx, source_id: str) -> list[ScrapedPaperInput]:
        if source_id != "arxiv":
            return []
        now = datetime.now(UTC)
        page = await self.fetch_page(ctx, (now - timedelta(days=7)).isoformat(), now.isoformat())
        return page.papers

    async def fetch_page(
        self,
        ctx: Ctx,
        since: str,
        until: str,
        cursor: str | None = None,
        limit: int = 20,
    ) -> SourcePage:
        def stamp(value: str) -> str:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d%H%M")

        start = int(cursor or "0")
        categories = " OR ".join(f"cat:{category}" for category in ALLOWED_CATEGORIES)
        params = {
            "search_query": f"({categories}) AND submittedDate:[{stamp(since)} TO {stamp(until)}]",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": limit,
        }
        root = await self._query(ctx, params)
        entries = root.findall("a:entry", NS)
        items = self._parse_entries(root)
        total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults", "0"))
        next_cursor = (
            str(start + len(entries)) if entries and start + len(entries) < total else None
        )
        return SourcePage(papers=items, next_cursor=next_cursor)

    @staticmethod
    def canonical_pdf_url(url: str) -> str:
        """Accept arXiv PDF identifiers only; never fetch a supplied host, query, or redirect."""
        parsed = urlsplit(url.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}
            or parsed.query
            or not parsed.path.startswith("/pdf/")
        ):
            raise ValueError("Use an arXiv PDF URL such as https://arxiv.org/pdf/1706.03762")
        identifier = parsed.path.removeprefix("/pdf/").removesuffix(".pdf")
        modern = r"\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}"
        legacy = r"[A-Za-z][A-Za-z.-]*/\d{7}"
        if ".." in identifier or not re.fullmatch(
            rf"(?:{modern}|{legacy})(?:v[1-9]\d*)?", identifier
        ):
            raise ValueError("The URL does not contain a valid arXiv paper identifier")
        return f"https://arxiv.org/pdf/{identifier}"

    async def fetch_paper(self, ctx: Ctx, url: str) -> ScrapedPaperInput:
        """Resolve one paper of any age through arXiv metadata, preserving explicit versions."""
        canonical = self.canonical_pdf_url(url)
        identifier = canonical.split("/pdf/", 1)[1]
        root = await self._query(ctx, {"id_list": identifier})
        papers = self._parse_entries(root)
        expected_id = re.sub(r"v\d+$", "", identifier)
        papers = [paper for paper in papers if paper.paper_id == expected_id]
        if not papers:
            raise ValueError(
                "Paper not found, or its primary category is outside the enabled AI topics"
            )
        return papers[0]

    async def _query(self, ctx: Ctx, params: dict[str, str | int]) -> ET.Element:
        await wait_for_source_host(ctx, "export.arxiv.org")
        timeout = max(0.1, min(30.0, ctx.deadline - time.monotonic()))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.get("https://export.arxiv.org/api/query", params=params)
            response.raise_for_status()
        return ET.fromstring(response.text)

    @staticmethod
    def _parse_entries(root: ET.Element) -> list[ScrapedPaperInput]:
        entries = root.findall("a:entry", NS)
        items: list[ScrapedPaperInput] = []
        for entry in entries:
            raw_id = (entry.findtext("a:id", "", NS)).rsplit("/abs/", 1)[-1]
            if raw_id.startswith("http") or raw_id == "":
                raise ValueError("arXiv returned an invalid entry identifier")
            primary = entry.find("x:primary_category", NS)
            category = primary.get("term", "") if primary is not None else ""
            if category not in ALLOWED_CATEGORIES:
                continue
            items.append(
                ScrapedPaperInput(
                    paper_id=re.sub(r"v\d+$", "", raw_id),
                    title=" ".join(entry.findtext("a:title", "", NS).split()),
                    source="arxiv",
                    pdf_url=f"https://arxiv.org/pdf/{raw_id}",
                    abstract=" ".join(entry.findtext("a:summary", "", NS).split()),
                    authors=[name.text or "" for name in entry.findall("a:author/a:name", NS)],
                    published_at=entry.findtext("a:published", None, NS),
                    metadata={"primary_category": category, "version_id": raw_id},
                )
            )
        return items
