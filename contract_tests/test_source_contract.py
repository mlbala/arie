from __future__ import annotations

import pytest
from arie_core.plugin_api import ScrapedPaperInput, normalize_scraper_output


class FakeSourcePlugin:
    def __init__(self) -> None:
        self.items = [
            ScrapedPaperInput(
                paper_id="paper-1",
                title="Alpha",
                source="arxiv",
                pdf_url="https://example.com/alpha.pdf",
            ),
            ScrapedPaperInput(
                paper_id="paper-2",
                title="Beta",
                source="crossref",
                pdf_url="https://example.com/beta.pdf",
            ),
        ]

    async def fetch(self, ctx: object | None, source_id: str) -> list[ScrapedPaperInput]:
        return [item for item in self.items if item.source == source_id]


@pytest.mark.asyncio
async def test_source_contract_normalizes_and_isolates_paper_ids() -> None:
    raw = [
        {
            "paper_id": "paper-3",
            "title": "  Gamma  ",
            "source": "arxiv",
            "pdf_url": " https://example.com/gamma.pdf ",
        },
        {
            "paper_id": "paper-4",
            "title": "Delta",
            "source": "arxiv",
            "pdf_url": "https://example.com/delta.pdf",
        },
    ]

    normalized = normalize_scraper_output(raw, "arxiv")
    assert all(item.source == "arxiv" for item in normalized)
    assert normalized[0].title == "Gamma"
    assert normalized[0].pdf_url == "https://example.com/gamma.pdf"

    source = FakeSourcePlugin()
    result = await source.fetch(None, "arxiv")
    assert all(item.source == "arxiv" for item in result)
    assert all(item.paper_id != "paper-2" for item in result)
