from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from arie_core.contracts import ChunkRow


def _truncate_text(value: str, *, max_tokens: int = 1500) -> str:
    words = value.split()
    if len(words) <= max_tokens:
        return value
    return " ".join(words[:max_tokens]) + " ..."


def _schema_for(
    name: str,
    description: str,
    properties: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": value["type"], "description": value["description"]}
                for key, value in properties.items()
            },
            "required": list(properties),
        },
    }


def build_lexical_tools(
    *,
    paper_id: str,
    chunks: Sequence[ChunkRow],
    tables: Sequence[Mapping[str, object]] | None = None,
    pages: Mapping[str, str] | None = None,
) -> dict[str, Callable[..., str]]:
    """Create paper-bound lexical tool closures.

    paper_id is closed over in Python and never exposed in the tool schema.
    """
    relevant_chunks = [chunk for chunk in chunks if chunk.paper_id == paper_id]
    table_rows = list(tables or [])
    page_map = dict(pages or {})

    def search_paper(query: str) -> str:
        needle = query.lower().strip()
        if not needle:
            return "No search term provided."
        matches = [
            chunk
            for chunk in relevant_chunks
            if needle in chunk.text.lower() or needle in chunk.section.lower()
        ]
        if not matches:
            return "No matching evidence in this paper."
        rendered = [f"[{match.section}] {match.text[:400]}" for match in matches[:5]]
        return "\n\n".join(rendered)

    def read_section(section: str) -> str:
        requested = section.strip()
        if not requested:
            return "No section requested."
        matches = [chunk for chunk in relevant_chunks if chunk.section.lower() == requested.lower()]
        if not matches:
            return "Section not found for this paper."
        block = "\n\n".join(chunk.text for chunk in matches[:5])
        return _truncate_text(block, max_tokens=1500)

    def get_table(table_name: str) -> str:
        requested = table_name.strip()
        if not requested:
            return "No table name requested."
        for row in table_rows:
            label = str(row.get("name") or row.get("title") or "")
            if label.lower() == requested.lower():
                payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
                return _truncate_text(payload, max_tokens=1500)
        return "Table not found in this paper."

    def get_page(page: int | str) -> str:
        key = str(page).strip()
        if not key:
            return "No page requested."
        page_value = page_map.get(key)
        if page_value is None:
            for page_key, text in page_map.items():
                if page_key.isdigit() and str(page) == page_key:
                    page_value = text
                    break
        if page_value is None:
            return "Page not found in this paper."
        return _truncate_text(page_value, max_tokens=1500)

    class SchemaTaggedCallable(Protocol):
        __schema__: dict[str, object]

    def _attach_schema(
        func: Callable[..., str],
        schema_name: str,
        schema_description: str,
        properties: Mapping[str, Mapping[str, str]],
    ) -> None:
        tagged = cast(SchemaTaggedCallable, func)
        tagged.__schema__ = _schema_for(schema_name, schema_description, properties)

    _attach_schema(
        search_paper,
        "search_paper",
        "Search within the bound paper for a phrase or concept using the current paper context.",
        {
            "query": {
                "type": "string",
                "description": "A phrase or concept to locate in the paper.",
            }
        },
    )
    _attach_schema(
        read_section,
        "read_section",
        "Read the requested section within the bound paper.",
        {
            "section": {
                "type": "string",
                "description": "The section name, such as 'abstract' or 'methods'.",
            }
        },
    )
    _attach_schema(
        get_table,
        "get_table",
        "Retrieve the named table from the current paper.",
        {
            "table_name": {
                "type": "string",
                "description": "The table name or title to fetch.",
            }
        },
    )
    _attach_schema(
        get_page,
        "get_page",
        "Retrieve the requested page from the current paper.",
        {
            "page": {
                "type": "integer",
                "description": "The zero-indexed or one-indexed page number to fetch.",
            }
        },
    )

    ordered_tools: dict[str, Callable[..., str]] = {
        "search_paper": search_paper,
        "read_section": read_section,
        "get_table": get_table,
        "get_page": get_page,
    }
    return ordered_tools


def create_lexical_tool_session(
    *,
    paper_id: str,
    chunks: Sequence[ChunkRow],
    tables: Sequence[Mapping[str, object]] | None = None,
    pages: Mapping[str, str] | None = None,
) -> dict[str, Callable[..., str]]:
    """Return a paper-bound tool closure bundle in canonical order and with paper_id closed over."""
    return build_lexical_tools(paper_id=paper_id, chunks=chunks, tables=tables, pages=pages)


__all__ = ["build_lexical_tools", "create_lexical_tool_session"]
