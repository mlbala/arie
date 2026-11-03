"""Source plugins against recorded API shapes: no network, declared hosts only (I4)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from arie_core.budget import BudgetHandle
from arie_core.plugin_api import SourcePage, normalize_scraper_output
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Context:
    deadline: float = field(default_factory=lambda: time.monotonic() + 10.0)
    trace_id: str = "source-contract"
    budget: BudgetHandle = field(default_factory=BudgetHandle)


def serve(monkeypatch: pytest.MonkeyPatch, payload: object) -> list[httpx.Request]:
    """Route every plugin request to ``payload`` and record it."""
    seen: list[httpx.Request] = []
    original = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        assert kwargs.get("follow_redirects") is False, "redirects could reach undeclared hosts"
        return original(timeout=kwargs.get("timeout"), transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


def source(name: str) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin("source", name)


@pytest.mark.asyncio
async def test_deepmind_keeps_arxiv_versions_and_rebuilds_abstracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(
        monkeypatch,
        {
            "meta": {"next_cursor": "page-2"},
            "results": [
                {
                    "title": "Scaling Agents",
                    "ids": {"arxiv": "https://arxiv.org/abs/2510.01234v2"},
                    "abstract_inverted_index": {"Agents": [0], "scale.": [1]},
                    "publication_date": "2026-10-20",
                    "authorships": [
                        {"display_name": "A. Researcher", "institutions": [{"display_name": "X"}]}
                    ],
                    "primary_location": {"source": {"display_name": "NeurIPS"}},
                },
                {"title": "Journal-only work", "ids": {"doi": "10.1/abc"}},
                {"title": "", "ids": {"arxiv": "2510.09999"}},
            ],
        },
    )
    plugin = source("deepmind")
    page = cast(SourcePage, await plugin.fetch_page(Context(), "2026-10-01", "2026-10-31"))
    assert [paper.paper_id for paper in page.papers] == ["2510.01234"]
    paper = page.papers[0]
    assert paper.pdf_url == "https://arxiv.org/pdf/2510.01234v2"
    assert paper.abstract == "Agents scale."
    assert paper.metadata["venue"] == "NeurIPS"
    assert "Google DeepMind" in json.loads(paper.metadata["affiliations_json"])
    assert page.next_cursor == "page-2"
    assert {request.url.host for request in seen} <= set(plugin.allowed_hosts)


@pytest.mark.asyncio
async def test_openreview_output_passes_the_scraper_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(
        monkeypatch,
        {
            "notes": [
                {
                    "id": "abc123",
                    "cdate": 1767225600000,
                    "content": {
                        "title": "Grounded Reading",
                        "abstract": "We study grounded reading.",
                        "authors": ["B. Author", {"name": "C. Author"}],
                        "venue": "ICLR 2027",
                    },
                },
                {"id": "no-title", "content": {"abstract": "Missing title"}},
                "not-a-note",
            ]
        },
    )
    plugin = source("openreview")
    assert await plugin.discover(None, "arxiv") == []
    items = await plugin.discover(None, "openreview")
    papers = normalize_scraper_output(items, "openreview")
    assert [paper.paper_id for paper in papers] == ["abc123"]
    assert papers[0].pdf_url == "https://openreview.net/pdf?id=abc123"
    assert papers[0].authors == ["B. Author", "C. Author"]
    assert papers[0].metadata["venue"] == "ICLR 2027"
    assert {request.url.host for request in seen} <= set(plugin.allowed_hosts)


def test_scraper_boundary_rejects_traversal_in_paper_ids() -> None:
    for paper_id in ("../etc/passwd", "a\\b"):
        with pytest.raises(ValueError):
            normalize_scraper_output(
                [{"paper_id": paper_id, "title": "T", "pdf_url": "https://arxiv.org/pdf/1"}],
                "openreview",
            )
