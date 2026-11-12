"""Shared state of the research runtime: config, plugins, worker pools and queues."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from arie_core.budget import BudgetHandle
from arie_core.harness.session import EXPLAIN_SECONDS
from arie_core.plugin_api import (
    Database,
    PaperURLSource,
    VectorStore,
)
from arie_core.registry.manager import ArieConfig, RegistryManager
from arie_core.registry.sync import months_ago as months_ago
from arie_core.sched import ForegroundGate
from arie_worker.extraction import ExtractionLimits, ExtractionPool
from arie_worker.models import LocalModel

from arie_api.models import AnswerModels
from arie_api.paper_eval import PaperEvaluations
from arie_api.providers import (
    DEFAULT_LLM,
    LLMProviders,
    config_path,
)

Document = dict[str, Any]

logger = logging.getLogger(__name__)

# Chunks per embedding call: about 2 s on CPU, the longest a chat query waits behind it.
EMBED_BATCH = 16
# bge-small-en-v1.5 scores unrelated questions up to ~0.58 and paraphrases from ~0.6.
DEFAULT_MIN_SCORE = 0.6
# Saved chats: the most recent turns of the most recently used papers; older ones are dropped.
CHAT_PAPERS = 10
CHAT_TURNS = 10


@dataclass(frozen=True)
class CallContext:
    deadline: float
    trace_id: str
    budget: BudgetHandle


def context(trace_id: str, seconds: float = 120.0) -> CallContext:
    return CallContext(time.monotonic() + seconds, trace_id, BudgetHandle(max_wall_sec=seconds))


def explain_context(
    trace_id: str, context_tokens: int, seconds: float = EXPLAIN_SECONDS
) -> CallContext:
    """A context for one explain call: tokens up to the model's window, 95 s of wall time.

    ``context()`` gives 12,000 tokens, which a paper pack alone would exceed. The judge's call
    over the same pack takes its own context, with ``seconds`` of its own (S13).
    """
    budget = BudgetHandle(max_tokens=context_tokens, max_wall_sec=seconds)
    return CallContext(time.monotonic() + seconds, trace_id, budget)


class RuntimeBase:
    """State every runtime layer shares; ``ResearchRuntime`` adds models and evaluations."""

    models: AnswerModels
    evaluations: PaperEvaluations
    entail_floor: float

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        self.root = root
        self.config = config or ArieConfig.from_yaml(config_path(root))
        self.manager = RegistryManager(project_root=root, config=self.config)
        self.db = cast(Database, self.instance("database"))
        file_config = cast(dict[str, object], self.config.plugins.get("file_storage", {}))
        self.storage_root = root / str(file_config.get("root") or "data/papers")
        parser_config = dict(cast(dict[str, object], self.config.plugins.get("parser", {})))
        parser_name = str(parser_config.pop("use", "") or "pymupdf4llm")
        parser_limits = ExtractionLimits.pop_from(parser_config)
        self.pool = ExtractionPool(
            root,
            parser_name,
            self.config.runtime_qos.utility_worker_processes,
            parser_config,
            parser_limits,
            self.config.ingestion.pdf_check,
            self.config.ingestion.hidden_text,
        )
        self.file_storage = self.instance("file_storage")
        self.jobs: dict[str, Document] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.import_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.lock = asyncio.Lock()
        self.url_source = cast(PaperURLSource, self.manager.instantiate_plugin("source", "arxiv"))
        self.subscribers: set[asyncio.Queue[Document]] = set()
        self.providers = LLMProviders(root, self.config, self.manager, self.lock)
        self.min_score = DEFAULT_MIN_SCORE
        # Vectors are comparable only within one embedder model; rows are tagged with this.
        self.embedder_id = "none"
        self.embedder_reason = ""
        self.vector_store: VectorStore | None = None
        self.embedding: LocalModel | None = None
        self.vector_wakeup = asyncio.Event()
        self.vector_errors: dict[str, str] = {}
        # Paper guides (S14): papers waiting for one, and the builder's wake-up.
        self.guide_pending: dict[str, None] = {}
        self.guide_wakeup = asyncio.Event()
        self.guide_errors: dict[str, str] = {}
        # Held by every answer in flight; PDF parsing, embedding and guide builds wait on it
        # before starting new work, so a question never competes with them for the CPU (S16).
        self.foreground = ForegroundGate()

    def llm_provider(self) -> str:
        """The name of the configured LLM plugin."""
        settings = cast(dict[str, object], self.config.plugins.get("llm", {}))
        return str(settings.get("use") or DEFAULT_LLM)

    def eval_judge_provider(self) -> str:
        """The LLM provider that judges explain against strict in evaluations.

        ``eval.judge_provider`` if set. Otherwise the first configured provider other than the
        writer's whose key is set, so the judge does not grade its own writing, and the
        writer's provider if there is none.
        """
        writer = self.llm_provider()
        named = self.config.eval.judge_provider
        if named:
            return named
        for name in self.config.llm_providers:
            try:
                keyed = bool(self.providers.settings(name)["api_key"])
            except ValueError:  # a profile with no plugin installed
                continue
            if name != writer and keyed:
                return name
        return writer

    def instance(self, slot: str, name: str | None = None) -> object:
        settings = cast(dict[str, object], self.config.plugins.get(slot, {}))
        if slot == "llm":
            provider = name or self.llm_provider()
            return self.manager.instantiate_plugin(
                slot, provider, **self.providers.settings(provider)
            )
        kwargs = {key: value for key, value in settings.items() if key != "use"}
        if slot in {"database", "vector_store"} and "path" in kwargs:
            kwargs["db_path"] = self.root / str(kwargs.pop("path"))
        if slot == "file_storage" and "root" in kwargs:
            kwargs["root"] = self.root / str(kwargs["root"])
        return self.manager.instantiate_plugin(slot, name, **kwargs)
