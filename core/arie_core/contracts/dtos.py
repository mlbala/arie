from __future__ import annotations

from collections.abc import Sequence

import msgspec


class TokenUsage(msgspec.Struct, frozen=True):
    prompt: int
    completion: int
    cache_hit: int = 0
    cache_miss: int = 0


class ToolCall(msgspec.Struct, frozen=True):
    name: str
    args: dict[str, str | int | float | bool | None]
    call_id: str


class LLMRequest(msgspec.Struct, frozen=True, kw_only=True):
    model: str = "default"
    prompt: str = ""
    system_prompt: str | None = None
    tools: list[str] = []
    max_output_tokens: int = 700


class LLMResponse(msgspec.Struct, frozen=True):
    text: str
    usage: TokenUsage
    tool_calls: list[ToolCall] = []
    finish: bool = True


class LLMDelta(msgspec.Struct, frozen=True):
    text: str = ""
    tool_calls: list[ToolCall] = []
    finish: bool = False
    # Set on the final delta only, and None when the provider reported no usage.
    usage: TokenUsage | None = None


class FileRef(msgspec.Struct, frozen=True):
    path: str
    kind: str = "file"
    sha256: str | None = None
    size: int = 0


class VectorRow(msgspec.Struct, frozen=True):
    id: str
    vector: list[float]
    metadata: dict[str, str] = {}


class Hit(msgspec.Struct, frozen=True):
    id: str
    score: float
    payload: dict[str, str | int | float | bool | None] = {}


class VectorStats(msgspec.Struct, frozen=True):
    vectors: int
    size_bytes: int | None = None


class PaperRow(msgspec.Struct, frozen=True):
    paper_id: str
    title: str
    source: str
    pdf_url: str | None = None
    abstract: str | None = None
    metadata: dict[str, str] = {}


class ChunkRow(msgspec.Struct, frozen=True):
    chunk_id: str
    paper_id: str
    section: str = "body"
    text: str = ""
    order: int = 0
    token_count: int = 0


class ParsedDoc(msgspec.Struct, frozen=True):
    paper_id: str
    title: str
    source: str
    sections: list[str] = []
    metadata: dict[str, str] = {}


class Evidence(msgspec.Struct, frozen=True):
    id: str
    chunk_id: str
    quote: str


class Claim(msgspec.Struct, frozen=True):
    text: str
    evidence: Sequence[Evidence]


class AnswerDraft(msgspec.Struct, frozen=True):
    status: str
    evidence: Sequence[Evidence] = []
    claims: Sequence[Claim] = []
    tool_call: ToolCall | None = None
