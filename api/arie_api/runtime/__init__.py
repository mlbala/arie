"""The research runtime: papers, sync jobs, vectors, chat and model workers.

``ResearchRuntime`` is assembled from one layer per concern (``base`` → ``library`` →
``vectors`` → ``sync`` → ``chat``); each layer may call the ones before it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from arie_core.harness.verifier import MIN_ENTAILMENT
from arie_core.registry.manager import ArieConfig
from arie_core.registry.sync import months_ago as months_ago

from arie_api.models import AnswerModels
from arie_api.paper_eval import PaperEvaluations
from arie_api.runtime.base import (
    CHAT_PAPERS,
    CHAT_TURNS,
    CallContext,
    Document,
    context,
    logger,
)
from arie_api.runtime.chat import ChatHistory


class ResearchRuntime(ChatHistory):
    """Persist real papers and resumable sync jobs; supervise failures per paper and source."""

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        super().__init__(root, config)
        self.models = AnswerModels(root, self.config)
        self.evaluations = PaperEvaluations(self)
        self.entail_floor = self.models.option("entailment", "min_score", MIN_ENTAILMENT)

    async def initialize(self) -> None:
        ctx = context("startup")
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_library (
            paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
            published_at TEXT NOT NULL, summary_json TEXT NOT NULL, document_json TEXT NOT NULL,
            topics_text TEXT NOT NULL DEFAULT '|')""",
        )
        try:
            await self.db.fetch(ctx, "SELECT topics_text FROM arie_library LIMIT 1")
        except Exception:
            await self.db.execute(
                ctx, "ALTER TABLE arie_library ADD COLUMN topics_text TEXT NOT NULL DEFAULT '|'"
            )
        await self.db.execute(
            ctx,
            "CREATE INDEX IF NOT EXISTS arie_library_date ON arie_library(status, published_at)",
        )
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_vector_index (
            paper_id TEXT PRIMARY KEY, model TEXT NOT NULL, revision TEXT NOT NULL,
            vectors INTEGER NOT NULL, updated_at TEXT NOT NULL)""",
        )
        while True:
            untagged = await self.db.fetch(
                ctx, "SELECT document_json FROM arie_library WHERE topics_text = '|' LIMIT 50"
            )
            if not untagged:
                break
            for row in untagged:
                await self.save_paper(cast(Document, json.loads(str(row["document_json"]))))
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_chat_turns (
            turn_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question TEXT NOT NULL,
            events_json TEXT NOT NULL, created_at TEXT NOT NULL)""",
        )
        await self.db.execute(
            ctx,
            """CREATE INDEX IF NOT EXISTS arie_chat_turns_paper
            ON arie_chat_turns(paper_id, created_at)""",
        )
        await self.db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_sync_runs (
            job_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, payload_json TEXT NOT NULL)""",
        )
        await self.evaluations.initialize()
        rows = await self.db.fetch(
            ctx, "SELECT payload_json FROM arie_sync_runs ORDER BY updated_at"
        )
        for row in rows:
            job = cast(Document, json.loads(str(row["payload_json"])))
            self.jobs[job["job_id"]] = job
            if job["status"] in {"running", "queued"}:
                job.update(status="interrupted", message="Sync interrupted. Resume to continue.")
                await self.update_job(job)

    def close(self) -> None:
        self.pool.close()
        if self.embedding is not None:
            self.embedding.close()
        self.models.close()


__all__ = [
    "CHAT_PAPERS",
    "CHAT_TURNS",
    "CallContext",
    "Document",
    "ResearchRuntime",
    "context",
    "logger",
    "months_ago",
]
