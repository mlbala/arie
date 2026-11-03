from __future__ import annotations

import asyncio
import hashlib
import re
import time
import weakref
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Protocol, TypeVar, cast

import msgspec
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arie_core.contracts import (
    AnswerDraft,
    ChunkRow,
    Claim,
    Evidence,
    Hit,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    ParsedDoc,
    TokenUsage,
    ToolCall,
    VectorRow,
    VectorStats,
)

T = TypeVar("T")


def configured_api_key(value: str | None) -> str | None:
    """Return a stripped server credential, treating empty and example values as unconfigured."""
    cleaned = (value or "").strip()
    if not cleaned or cleaned.lower().startswith(("your-", "your_", "paste-", "paste_")):
        return None
    if cleaned.lower() in {"changeme", "replace-me", "replace_me"}:
        return None
    return cleaned


class BudgetHandle(Protocol):
    def record_tool_call(self, name: str, *, args: Mapping[str, object] | None = None) -> str: ...
    def reserve(self, amount: float) -> None: ...
    def consume(self, amount: float) -> None: ...


class Ctx(Protocol):
    @property
    def deadline(self) -> float: ...
    @property
    def trace_id(self) -> str: ...
    @property
    def budget(self) -> BudgetHandle: ...


_SOURCE_REQUESTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, float]] = (
    weakref.WeakKeyDictionary()
)


async def wait_for_source_host(ctx: Ctx, host: str) -> None:
    """Space source requests across plugin instances without reserving past the caller deadline."""
    loop = asyncio.get_running_loop()
    reservations = _SOURCE_REQUESTS.setdefault(loop, {})
    now = time.monotonic()
    due = max(now, reservations.get(host, now))
    if due >= ctx.deadline:
        raise TimeoutError("Source request deadline would expire while waiting for its host")
    reservations[host] = due + (3.0 if host == "export.arxiv.org" else 1.0)
    await asyncio.sleep(max(0.0, due - now))


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
            raise ValueError(f"Plugin {name!r} requires API {api!r} but current core API is 1.0")
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
    """Turn text into fixed-width vectors; runs only inside arie-worker processes.

    Contract: return exactly one vector per input text, in input order, all with the
    same dimension for a given configured model. Errors propagate to the caller, which
    keeps the paper keyword-searchable and retries indexing later.
    """

    async def embed(self, ctx: Ctx, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """Paper-scoped vector index; bring your own by registering ``@plugin(slot="vector_store")``.

    Every ``VectorRow`` carries ``metadata["paper_id"]`` and ``metadata["model"]`` (the
    embedder identity). Contract, checked by ``contract_tests/test_vector_store_contract.py``:

    - ``upsert`` replaces rows by ``id``; a row missing ``paper_id`` or ``model`` raises
      ``ValueError``, and a failed batch must not leave a partial write.
    - ``search`` returns at most ``k`` hits by descending cosine similarity, only rows whose
      metadata equals every ``filters`` item, with that metadata echoed in ``Hit.payload``.
    - ``delete`` removes every row matching all ``filters`` (at least ``paper_id``).
    - ``stats`` reports the total row count and, when measurable, bytes on disk.

    Core still post-filters hits by ``paper_id`` (I1), so a faulty store cannot mix papers.
    """

    async def upsert(self, ctx: Ctx, rows: Sequence[VectorRow]) -> None: ...
    async def search(
        self,
        ctx: Ctx,
        query: Sequence[float],
        k: int,
        *,
        filters: Mapping[str, str],
    ) -> list[Hit]: ...
    async def delete(self, ctx: Ctx, *, filters: Mapping[str, str]) -> None: ...
    async def stats(self, ctx: Ctx) -> VectorStats: ...


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


class PaperURLSource(SourcePlugin, Protocol):
    @property
    def allowed_hosts(self) -> tuple[str, ...]: ...

    def canonical_pdf_url(self, url: str) -> str: ...
    async def fetch_paper(self, ctx: Ctx, url: str) -> ScrapedPaperInput: ...


class Parser(Protocol):
    async def parse(self, ctx: Ctx, file_path: str) -> ParsedDoc: ...


class Reranker(Protocol):
    """Question-passage relevance scoring; runs only inside arie-worker processes.

    Contract: return one relevance score per passage for ``query``, in input order;
    higher means more relevant, and scores are compared only within one call. Errors
    propagate; the caller then keeps its own keyword and vector ranking.
    """

    async def score(self, ctx: Ctx, query: str, passages: Sequence[str]) -> list[float]: ...


class Verifier(Protocol):
    async def verify(self, ctx: Ctx, claim: Claim) -> bool: ...


class Entailment(Protocol):
    """Tier 3 of the quote verifier; runs only inside arie-worker processes.

    Contract: for each ``(premise, hypothesis)`` pair return the probability in [0, 1]
    that the premise (an exact source quote) entails the hypothesis (a drafted claim),
    one score per pair in input order. Errors propagate; the caller then abstains to
    source excerpts rather than showing unverified claims (I2).
    """

    async def score(self, ctx: Ctx, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class Feature(Protocol):
    def add_tool(self, name: str, tool: Callable[[object], object]) -> None: ...
    def add_skill(self, name: str, skill: str) -> None: ...
    def add_pipeline_step(self, name: str, step: Callable[[object], object]) -> None: ...
    def on(self, event: str, callback: Callable[[dict[str, object]], None]) -> None: ...
    def add_brief_field(self, name: str, field_type: type[object]) -> None: ...
    def add_route(self, path: str, handler: Callable[[object], object]) -> None: ...
    def add_ui_panel(self, name: str, panel: object) -> None: ...


def _normalize_metadata_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, str] = {}
    mapping = cast(Mapping[object, object], value)
    for key, item in mapping.items():
        normalized[str(key)] = "" if item is None else str(item)
    return normalized


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

    @model_validator(mode="before")
    @classmethod
    def normalize_custom_metadata(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        value_mapping = cast(Mapping[object, object], value)
        data: dict[str, object] = {str(key): item for key, item in value_mapping.items()}
        custom = data.pop("custom_metadata", None)
        if isinstance(custom, Mapping):
            existing_metadata = data.get("metadata")
            merged_metadata = _normalize_metadata_mapping(
                cast(Mapping[object, object], existing_metadata)
                if isinstance(existing_metadata, Mapping)
                else {}
            )
            custom_mapping = cast(Mapping[object, object], custom)
            for key, item in custom_mapping.items():
                merged_metadata[str(key)] = "" if item is None else str(item)
            data["metadata"] = merged_metadata

        metadata = data.get("metadata")
        if isinstance(metadata, Mapping):
            data["metadata"] = _normalize_metadata_mapping(cast(Mapping[object, object], metadata))

        if "paper_id" not in data and "id" in data:
            paper_id = data.get("id")
            if paper_id is not None:
                data["paper_id"] = paper_id
        return data

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


class SourcePage(msgspec.Struct, frozen=True):
    papers: list[ScrapedPaperInput]
    next_cursor: str | None = None


class PagedSourcePlugin(SourcePlugin, Protocol):
    allowed_hosts: tuple[str, ...]

    async def fetch_page(
        self,
        ctx: Ctx,
        since: str,
        until: str,
        cursor: str | None = None,
        limit: int = 20,
    ) -> SourcePage: ...


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

            custom_metadata = data.get("custom_metadata")
            if isinstance(custom_metadata, Mapping):
                existing_metadata = data.get("metadata")
                metadata = _normalize_metadata_mapping(
                    cast(Mapping[object, object], existing_metadata)
                    if isinstance(existing_metadata, Mapping)
                    else {}
                )
                custom_mapping = cast(Mapping[object, object], custom_metadata)
                for key, value in custom_mapping.items():
                    metadata[str(key)] = "" if value is None else str(value)
                data["metadata"] = metadata

            metadata = data.get("metadata")
            if isinstance(metadata, Mapping):
                data["metadata"] = _normalize_metadata_mapping(
                    cast(Mapping[object, object], metadata)
                )

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
    "configured_api_key",
    "Database",
    "Embedder",
    "Entailment",
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
    "PaperURLSource",
    "SourcePage",
    "PagedSourcePlugin",
    "ToolCall",
    "TokenUsage",
    "VectorStore",
    "Verifier",
    "normalize_scraper_output",
    "plugin",
    "wait_for_source_host",
]
