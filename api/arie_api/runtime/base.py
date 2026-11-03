"""Shared state of the research runtime: config, plugins, worker pools and queues."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from arie_core.budget import BudgetHandle
from arie_core.plugin_api import (
    Database,
    PaperURLSource,
    VectorStore,
)
from arie_core.registry.manager import ArieConfig, RegistryManager
from arie_core.registry.sync import months_ago as months_ago
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

    def instance(self, slot: str, name: str | None = None) -> object:
        settings = cast(dict[str, object], self.config.plugins.get(slot, {}))
        if slot == "llm":
            provider = name or str(settings.get("use") or DEFAULT_LLM)
            return self.manager.instantiate_plugin(
                slot, provider, **self.providers.settings(provider)
            )
        kwargs = {key: value for key, value in settings.items() if key != "use"}
        if slot in {"database", "vector_store"} and "path" in kwargs:
            kwargs["db_path"] = self.root / str(kwargs.pop("path"))
        if slot == "file_storage" and "root" in kwargs:
            kwargs["root"] = self.root / str(kwargs["root"])
        return self.manager.instantiate_plugin(slot, name, **kwargs)
