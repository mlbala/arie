from __future__ import annotations

from typing import Any, cast

import httpx
from arie_core.plugin_api import plugin

ALLOWED_HOSTS = ("api2.openreview.net", "openreview.net")


@plugin(slot="source", name="openreview")
class OpenReviewSourcePlugin:
    """Discover OpenReview papers and normalize them to the ARIE scraped-paper contract."""

    allowed_hosts = ALLOWED_HOSTS

    @staticmethod
    def _clean(value: object | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    async def discover(self, ctx: object | None, source_id: str) -> list[dict[str, Any]]:
        del ctx
        if source_id != "openreview":
            return []

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get(
                "https://api2.openreview.net/notes?content.venue=ICLR&details=replyCount&limit=50"
            )
            response.raise_for_status()
            payload_value = response.json()

        if not isinstance(payload_value, dict):
            return []
        payload = cast(dict[str, Any], payload_value)

        notes_value_raw = payload.get("notes")
        if not isinstance(notes_value_raw, list):
            return []
        notes_value = cast(list[object], notes_value_raw)

        items: list[dict[str, Any]] = []
        for note in notes_value:
            if not isinstance(note, dict):
                continue
            note_mapping = cast(dict[str, Any], note)
            content_raw = note_mapping.get("content")
            if not isinstance(content_raw, dict):
                continue
            content_mapping = cast(dict[str, Any], content_raw)

            title = self._clean(content_mapping.get("title"))
            abstract = self._clean(content_mapping.get("abstract"))
            if not title:
                continue

            paper_id = self._clean(note_mapping.get("id")) or self._clean(note_mapping.get("forum"))
            if not paper_id:
                continue

            metadata = {
                "venue": self._clean(content_mapping.get("venue")),
                "code_repository": self._clean(content_mapping.get("code")),
                "tldr": self._clean(
                    content_mapping.get("tldr")
                    if isinstance(content_mapping.get("tldr"), str)
                    else ""
                ),
            }
            authors_raw_value: object = content_mapping.get("authors") or []
            authors_list: list[str] = []
            if isinstance(authors_raw_value, list):
                authors_value = cast(list[object], authors_raw_value)
                for author in authors_value:
                    if isinstance(author, str):
                        cleaned = self._clean(author)
                        if cleaned:
                            authors_list.append(cleaned)
                    elif isinstance(author, dict):
                        author_dict = cast(dict[str, Any], author)
                        author_name = author_dict.get("name")
                        if author_name is None:
                            author_name = author_dict.get("full_name")
                        cleaned = self._clean(author_name)
                        if cleaned:
                            authors_list.append(cleaned)
            item = {
                "paper_id": paper_id,
                "title": title,
                "source": "openreview",
                "pdf_url": f"https://openreview.net/pdf?id={paper_id}",
                "abstract": abstract,
                "authors": authors_list,
                "published_at": self._clean(note_mapping.get("cdate")),
                "custom_metadata": metadata,
            }
            items.append(item)

        return items


__all__ = ["OpenReviewSourcePlugin"]
