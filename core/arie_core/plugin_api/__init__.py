from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arie_core.contracts import (
    AnswerDraft,
    ChunkRow,
    Claim,
    Evidence,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    ParsedDoc,
    ToolCall,
)

T = TypeVar("T")


class BudgetHandle(Protocol):
    def reserve(self, amount: float) -> None: ...
    def consume(self, amount: float) -> None: ...


class Ctx(Protocol):
    deadline: float
    trace_id: str
    budget: BudgetHandle


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", version)
    if match is None:
        raise ValueError(f"Invalid API version constraint: {version!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    max_len = max(len(left), len(right))
    left_padded = left + (0,) * (max_len - len(left))
    right_padded = right + (0,) * (max_len - len(right))
    if left_padded < right_padded:
        return -1
    if left_padded > right_padded:
        return 1
    return 0


def _matches_api_constraint(version: str, constraint: str) -> bool:
    current = _version_tuple(version)
    for part in constraint.split(","):
        expression = part.strip()
        if not expression:
            continue
        if expression.startswith(">="):
            if _compare_versions(current, _version_tuple(expression[2:])) < 0:
                return False
        elif expression.startswith("<="):
            if _compare_versions(current, _version_tuple(expression[2:])) > 0:
                return False
        elif expression.startswith("=="):
            if _compare_versions(current, _version_tuple(expression[2:])) != 0:
                return False
        elif expression.startswith("!="):
            if _compare_versions(current, _version_tuple(expression[2:])) == 0:
                return False
        elif expression.startswith(">"):
            if _compare_versions(current, _version_tuple(expression[1:])) <= 0:
                return False
        elif expression.startswith("<"):
            if _compare_versions(current, _version_tuple(expression[1:])) >= 0:
                return False
        else:
            if _compare_versions(current, _version_tuple(expression)) != 0:
                return False
    return True


def plugin(slot: object, name: str, api: str = ">=1,<2") -> Callable[[T], T]:
    def decorator(target: T) -> T:
        if not _matches_api_constraint("1.0", api):
            raise ValueError(
                f"Plugin {name!r} requires API {api!r} but current core API is 1.0"
            )
        plugin_slot = "__arie_plugin_slot__"
        plugin_name = "__arie_plugin_name__"
        plugin_api = "__arie_plugin_api__"
        setattr(target, plugin_slot, slot)
        setattr(target, plugin_name, name)
        setattr(target, plugin_api, api)
        return target

    return decorator


class LLM(Protocol):
    async def complete(self, ctx: Ctx, request: LLMRequest) -> LLMResponse: ...
    async def stream(self, ctx: Ctx, request: LLMRequest) -> AsyncIterator[LLMDelta]: ...


class Embedder(Protocol):
    async def embed(self, ctx: Ctx, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def upsert(self, ctx: Ctx, rows: Sequence[object]) -> None: ...
    async def search(
        self,
        ctx: Ctx,
        query: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[object]: ...


class Database(Protocol):
    async def execute(
        self,
        ctx: Ctx,
        query: str,
        params: Sequence[object] = (),
    ) -> int: ...
    async def fetch(
        self,
        ctx: Ctx,
        query: str,
        params: Sequence[object] = (),
    ) -> list[dict[str, object]]: ...


class FileStorage(Protocol):
    async def read(self, ctx: Ctx, path: str) -> bytes: ...
    async def write(self, ctx: Ctx, path: str, content: bytes) -> None: ...


class SourcePlugin(Protocol):
    async def fetch(self, ctx: Ctx, source_id: str) -> list[ScrapedPaperInput]: ...


class Parser(Protocol):
    async def parse(self, ctx: Ctx, file_path: str) -> ParsedDoc: ...


class Reranker(Protocol):
    async def rerank(
        self,
        ctx: Ctx,
        query: str,
        candidates: Sequence[ChunkRow],
    ) -> list[ChunkRow]: ...


class Verifier(Protocol):
    async def verify(self, ctx: Ctx, claim: Claim) -> bool: ...


class Feature(Protocol):
    def add_tool(self, name: str, tool: Callable[[object], object]) -> None: ...
    def add_skill(self, name: str, skill: str) -> None: ...
    def add_pipeline_step(self, name: str, step: Callable[[object], object]) -> None: ...
    def on(self, event: str, callback: Callable[[dict[str, object]], None]) -> None: ...
    def add_brief_field(self, name: str, field_type: type[object]) -> None: ...
    def add_route(self, path: str, handler: Callable[[object], object]) -> None: ...
    def add_ui_panel(self, name: str, panel: object) -> None: ...


class ScrapedPaperInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    paper_id: str
    title: str
    source: str
    pdf_url: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("paper_id", "title", "source", "pdf_url", "abstract", "published_at")
    @classmethod
    def normalize_string_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @field_validator("paper_id")
    @classmethod
    def forbid_path_traversal(cls, value: str) -> str:
        if ".." in value or "\\" in value:
            raise ValueError("paper_id must not include path traversal sequences")
        return value

    @property
    def dedup_hash(self) -> str:
        payload = f"{self.title.strip().lower()}||{str(self.pdf_url).strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def storage_hash(self) -> str:
        return hashlib.sha256(self.paper_id.encode("utf-8")).hexdigest()[:16]


def normalize_scraper_output(
    raw_items: Sequence[ScrapedPaperInput | Mapping[str, object]], expected_source: str
) -> list[ScrapedPaperInput]:
    normalized: list[ScrapedPaperInput] = []
    for raw_item in raw_items:
        if isinstance(raw_item, ScrapedPaperInput):
            candidate = raw_item
        else:
            raw_mapping: Mapping[str, object] = raw_item
            data: dict[str, object] = dict(raw_mapping)
            if "source" not in data:
                data["source"] = expected_source
            if "paper_id" not in data and "id" in data:
                data["paper_id"] = data["id"]
            candidate = ScrapedPaperInput.model_validate(data)

        if candidate.source != expected_source:
            raise ValueError(
                f"Expected scraper output from {expected_source!r}, got {candidate.source!r}"
            )
        normalized.append(candidate)
    return normalized


__all__ = [
    "AnswerDraft",
    "BudgetHandle",
    "ChunkRow",
    "Claim",
    "Ctx",
    "Database",
    "Embedder",
    "Evidence",
    "Feature",
    "FileStorage",
    "LLM",
    "LLMDelta",
    "LLMRequest",
    "LLMResponse",
    "Parser",
    "Reranker",
    "ScrapedPaperInput",
    "SourcePlugin",
    "ToolCall",
    "VectorStore",
    "Verifier",
    "normalize_scraper_output",
    "plugin",
]
