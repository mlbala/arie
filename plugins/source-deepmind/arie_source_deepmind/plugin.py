from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
from arie_core.plugin_api import (
    Ctx,
    ScrapedPaperInput,
    SourcePage,
    normalize_scraper_output,
    plugin,
)

ALLOWED_HOSTS = ("api.openalex.org", "export.arxiv.org", "arxiv.org")
GOOGLE_DEEPMIND_ROR = "https://ror.org/00971b260"


@plugin(slot="source", name="deepmind")
class GoogleDeepMindSourcePlugin:
    """Fetches Google DeepMind papers from OpenAlex and narrows them to arXiv versions."""

    allowed_hosts = ALLOWED_HOSTS

    @staticmethod
    def _clean_string(value: object | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _extract_abstract(work: dict[str, Any]) -> str:
        abstract = work.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            return abstract.strip()

        index = work.get("abstract_inverted_index")
        if isinstance(index, dict):
            entries: list[tuple[int, str]] = []
            index_dict = cast(dict[str, Any], index)
            for token, raw_positions in index_dict.items():
                if not isinstance(raw_positions, list):
                    continue
                positions = cast(list[object], raw_positions)
                for pos in positions:
                    if isinstance(pos, int):
                        entries.append((pos, str(token)))
            if entries:
                ordered = sorted(entries, key=lambda item: item[0])
                return " ".join(token for _, token in ordered)
        return ""

    @staticmethod
    def _extract_arxiv_id(raw_id: object | None) -> str:
        value = str(raw_id or "").strip()
        if not value:
            return ""
        for prefix in (
            "https://arxiv.org/abs/",
            "http://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "http://arxiv.org/pdf/",
        ):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                break
        value = value.rstrip("/")
        return value if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", value) else ""

    @staticmethod
    def _coerce_affiliations(work: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        affiliations: list[str] = []
        authorships_raw = work.get("authorships")
        if isinstance(authorships_raw, list):
            authorships = cast(list[object], authorships_raw)
            for authorship in authorships:
                if not isinstance(authorship, dict):
                    continue
                authorship_dict = cast(dict[str, Any], authorship)
                institutions_raw = authorship_dict.get("institutions")
                if not isinstance(institutions_raw, list):
                    continue
                institutions = cast(list[object], institutions_raw)
                for institution in institutions:
                    if not isinstance(institution, dict):
                        continue
                    institution_dict = cast(dict[str, Any], institution)
                    name = str(
                        institution_dict.get("display_name") or institution_dict.get("name") or ""
                    ).strip()
                    if not name:
                        continue
                    if name not in seen:
                        seen.add(name)
                        affiliations.append(name)
        if not any("deepmind" in name.lower() for name in affiliations):
            affiliations.append("Google DeepMind")
        return affiliations

    async def fetch(self, ctx: Ctx, source_id: str) -> list[ScrapedPaperInput]:
        if source_id != "deepmind":
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
        del ctx
        params = {
            "filter": (
                f"institutions.ror:{GOOGLE_DEEPMIND_ROR},"
                f"from_publication_date:{since[:10]},to_publication_date:{until[:10]}"
            ),
            "per-page": str(limit),
            "sort": "publication_date:desc",
            "cursor": cursor or "*",
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
            payload_value = response.json()

        if not isinstance(payload_value, dict):
            raise ValueError("OpenAlex returned an invalid response")
        payload: dict[str, object] = cast(dict[str, object], payload_value)

        results_raw = payload.get("results")
        if not isinstance(results_raw, list):
            raise ValueError("OpenAlex returned no results list")
        results = cast(list[object], results_raw)
        items: list[dict[str, Any]] = []

        for result in results:
            if not isinstance(result, dict):
                continue
            work = cast(dict[str, Any], result)
            ids = work.get("ids")
            arxiv_value = ""
            if isinstance(ids, dict):
                ids_dict = cast(dict[str, Any], ids)
                arxiv_value = str(ids_dict.get("arxiv") or ids_dict.get("arxiv_id") or "")
            if not arxiv_value:
                arxiv_value = str(work.get("arxiv") or "")
            if not arxiv_value:
                locations = cast(list[dict[str, Any]], work.get("locations") or [])
                for location in locations:
                    for key in ("pdf_url", "landing_page_url"):
                        candidate = self._extract_arxiv_id(location.get(key))
                        if candidate:
                            arxiv_value = candidate
                            break
                    if arxiv_value:
                        break
            arxiv_id = self._extract_arxiv_id(arxiv_value)
            if not arxiv_id:
                continue
            title = self._clean_string(work.get("title"))
            if not title:
                continue

            affiliations = self._coerce_affiliations(work)
            metadata: dict[str, str] = {
                "institution_ror": GOOGLE_DEEPMIND_ROR,
                "affiliations_json": json.dumps(affiliations, separators=(",", ":")),
                "primary_category": "cs.AI",
                "version_id": arxiv_id,
            }
            primary_location = work.get("primary_location")
            if isinstance(primary_location, dict):
                primary_location_dict = cast(dict[str, Any], primary_location)
                venue_value = primary_location_dict.get("source")
                venue_mapping: dict[str, Any] | None = None
                if isinstance(venue_value, dict):
                    venue_mapping = cast(dict[str, Any], venue_value)
                if venue_mapping is not None:
                    venue_name = self._clean_string(
                        venue_mapping.get("display_name") or venue_mapping.get("name")
                    )
                    if venue_name:
                        metadata["venue"] = venue_name

            authorships_value = work.get("authorships")
            authors: list[str] = []
            if isinstance(authorships_value, list):
                authorships = cast(list[object], authorships_value)
                for author in authorships:
                    if not isinstance(author, dict):
                        continue
                    author_dict = cast(dict[str, Any], author)
                    display_name = self._clean_string(author_dict.get("display_name"))
                    if display_name:
                        authors.append(display_name)

            items.append(
                {
                    "paper_id": re.sub(r"v\d+$", "", arxiv_id),
                    "title": title,
                    "source": "deepmind",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                    "abstract": self._extract_abstract(work),
                    "authors": authors,
                    "published_at": self._clean_string(work.get("publication_date")),
                    "metadata": metadata,
                }
            )

        meta = cast(dict[str, object], payload.get("meta") or {})
        next_cursor = meta.get("next_cursor")
        return SourcePage(
            papers=normalize_scraper_output(items, "deepmind"),
            next_cursor=str(next_cursor) if next_cursor and results else None,
        )


__all__ = ["GoogleDeepMindSourcePlugin"]
