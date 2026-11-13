"""The guard: a local prompt-injection classifier that scores paper chunks in the background.

ADR 0014, SPEC 6.2 and 9. Off unless ``plugins.guard.use`` names a plugin. When on, every
indexed paper's chunks that ``needs_guard`` passes (about 1%) are scored in a ``Guard``
worker at background priority, in batches that each wait behind the foreground gate (S16).
Scores live in ``arie_guard_scores``, keyed like the vector index by guard model and paper
revision, so a new model or a re-parsed paper is scored again; a new threshold re-applies
the stored scores without the model. When a paper's set of untrusted chunks changes, its
stored pack and guide are deleted, so they are rebuilt without them. A paper with no
untrusted chunk keeps both: its pack bytes are the same either way.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from arie_core.harness.injection import SCORE_KEY, evidence_chunks, needs_guard
from arie_core.harness.reader import EvidenceChunk
from arie_core.registry.manager import ArieConfig
from arie_worker.models import LocalModel, ModelPool
from arie_worker.pool import BACKGROUND_NICE

from arie_api.runtime.base import CallContext, Document, context, logger
from arie_api.runtime.library import LibraryStore

# Chunks per worker call; each batch waits for the foreground gate first.
GUARD_BATCH = 32
# One batch of long chunks takes ~25 s on 2 threads (S20); a model reload adds a few more.
GUARD_BATCH_SECONDS = 180.0
# The first call may download the model (~740 MB).
GUARD_WARMUP_SECONDS = 600.0
GUARD_RETRY_SECONDS = 300.0


def paper_revision(paper: Document) -> str:
    """The vector index's revision of a paper, ``<dedup_hash>:<chunk_count>``."""
    return f"{paper.get('dedup_hash', '')}:{len(paper.get('chunks', []))}"


def _untrusted(scores: dict[str, float], threshold: float) -> set[str]:
    return {chunk_id for chunk_id, score in scores.items() if score >= threshold}


class PaperGuard(LibraryStore):
    """Guard scores per paper, the background scorer, and chunk loading with ``untrusted``."""

    def __init__(self, root: Path, config: ArieConfig | None = None) -> None:
        super().__init__(root, config)
        self.guard_config = self.config.guard()
        self.guard_model: LocalModel | None = None
        if self.guard_config.enabled:
            pool = ModelPool(root, "Guard", nice=BACKGROUND_NICE)
            self.guard_model = LocalModel(
                pool, "guard", self.guard_config.use, self.guard_config.plugin_settings()
            )
        self.guard_ready = False
        self.guard_wakeup = asyncio.Event()
        self.guard_errors: dict[str, str] = {}

    def guard_threshold(self) -> float | None:
        """The score at or over which a chunk is untrusted; ``None`` while the guard is off."""
        return self.guard_config.threshold if self.guard_model is not None else None

    async def evidence(self, paper: Document) -> list[EvidenceChunk]:
        """The paper's chunks, re-filtered by its id (I1), with ``untrusted`` from its scores.

        A chunk dict that already carries a score (the evaluation's planted chunk) keeps it.
        Stored OCR chunks (S25) are left out while ``ingestion.figure_ocr`` is off.
        """
        paper_id = str(paper["id"])
        rows = cast(list[Document], paper.get("chunks", []))
        if not self.config.ingestion.figure_ocr:
            rows = [row for row in rows if not row.get("ocr")]
        scores: dict[str, float] = {}
        if self.guard_model is not None:
            stored = await self._stored_row(context(paper_id), paper_id)
            if stored is not None and self._scores_current(stored, paper):
                scores = self._row_scores(stored)
        if scores:
            rows = [
                {**row, SCORE_KEY: scores[str(row.get("chunk_id"))]}
                if str(row.get("chunk_id")) in scores
                else row
                for row in rows
            ]
        return evidence_chunks(paper_id, rows, self.guard_threshold())

    async def _stored_row(self, ctx: CallContext, paper_id: str) -> Document | None:
        rows = await self.db.fetch(
            ctx,
            "SELECT paper_id, model, revision, threshold, scores_json FROM arie_guard_scores "
            "WHERE paper_id = ?",
            (paper_id,),
        )
        return next((dict(row) for row in rows if row["paper_id"] == paper_id), None)

    def _scores_current(self, row: Document, paper: Document) -> bool:
        """Whether a stored row holds this guard model's scores for this paper revision."""
        return row["model"] == self.guard_config.model_id and row["revision"] == paper_revision(
            paper
        )

    @staticmethod
    def _row_scores(row: Document) -> dict[str, float]:
        stored = cast(dict[str, float], json.loads(str(row["scores_json"])))
        return {str(key): float(value) for key, value in stored.items()}

    async def _guard_backlog(self) -> tuple[list[str], int]:
        """Indexed papers whose scores are missing, stale or saved under another threshold."""
        rows = await self.db.fetch(
            context("guard-backlog", 10.0),
            """SELECT l.paper_id, l.summary_json, g.model AS guard_model,
            g.revision AS guard_revision, g.threshold AS guard_threshold FROM arie_library l
            LEFT JOIN arie_guard_scores g ON g.paper_id = l.paper_id
            WHERE l.status = 'INDEXED' ORDER BY l.published_at DESC""",
        )
        config = self.guard_config
        pending: list[str] = []
        for row in rows:
            summary = cast(Document, json.loads(str(row["summary_json"])))
            revision = f"{summary.get('revision', '')}:{summary.get('chunk_count', 0)}"
            current = (row["guard_model"], row["guard_revision"]) == (config.model_id, revision)
            threshold = row["guard_threshold"]
            if not current or threshold is None or float(str(threshold)) != config.threshold:
                pending.append(str(row["paper_id"]))
        return pending, len(rows)

    async def score_paper(self, paper_id: str) -> None:
        """Score one paper's prefiltered chunks and save them; rebuild what they change.

        Scores already current for this model and revision are reused, so a threshold change
        costs no model call. The pack and guide are deleted only when the set of untrusted
        chunks changed. The row is written before that, and the backlog skips a paper only
        once its row is current, so an interrupted paper is picked up again.
        """
        model, config = self.guard_model, self.guard_config
        if model is None:
            return
        paper = await self.get_paper(paper_id)
        if paper is None or paper.get("status") != "INDEXED":
            return
        ctx = context(f"guard:{paper_id}", 30.0)
        previous = await self._stored_row(ctx, paper_id)
        if previous is not None and self._scores_current(previous, paper):
            scores = self._row_scores(previous)
        else:
            scores = await self._score_chunks(model, paper)
        before = (
            _untrusted(self._row_scores(previous), float(str(previous["threshold"])))
            if previous is not None
            else set[str]()
        )
        after = _untrusted(scores, config.threshold)
        await self.db.execute(
            ctx,
            """INSERT INTO arie_guard_scores
            (paper_id, model, revision, threshold, scores_json, scored, untrusted, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(paper_id) DO UPDATE SET
            model=excluded.model, revision=excluded.revision, threshold=excluded.threshold,
            scores_json=excluded.scores_json, scored=excluded.scored,
            untrusted=excluded.untrusted, updated_at=excluded.updated_at""",
            (
                paper_id,
                config.model_id,
                paper_revision(paper),
                config.threshold,
                json.dumps(scores, sort_keys=True),
                len(scores),
                len(after),
                datetime.now(UTC).isoformat(),
            ),
        )
        if after != before:
            # The stored pack and guide may hold a chunk that is now untrusted (or leave out
            # one that no longer is); the next reader rebuilds the pack, the builder the guide.
            for table in ("arie_paper_packs", "arie_paper_guides"):
                await self.db.execute(
                    ctx, f"DELETE FROM {table} WHERE paper_id = ?", (paper_id,)
                )
            self.guide_pending[paper_id] = None
            self.guide_wakeup.set()

    async def _score_chunks(self, model: LocalModel, paper: Document) -> dict[str, float]:
        paper_id = str(paper["id"])
        candidates = [
            (str(row["chunk_id"]), str(row.get("text", "")))
            for row in cast(list[Document], paper.get("chunks", []))
            if row.get("paper_id") == paper_id
            and row.get("chunk_id")
            and needs_guard(str(row.get("text", "")))
        ]
        scores: dict[str, float] = {}
        for start in range(0, len(candidates), GUARD_BATCH):
            batch = candidates[start : start + GUARD_BATCH]
            # One batch is the unit of waiting: a question that starts mid-paper gets the CPU
            # at the next batch boundary (S16).
            await self.foreground.wait_quiet()
            values = await model.scores(
                "score",
                [text for _, text in batch],
                count=len(batch),
                kill_after=GUARD_BATCH_SECONDS,
            )
            for (chunk_id, _), value in zip(batch, values, strict=True):
                scores[chunk_id] = round(value, 4)
        return scores

    async def run_guard(self) -> None:
        """Keep guard scores in step with the library: backfill, re-score on change, retry.

        Wakes at startup, whenever sync indexes a paper, and every 5 minutes while any paper
        failed. An unscored paper is still filtered by the regex (ADR 0014).
        """
        model = self.guard_model
        if model is None:
            return
        while True:
            self.guard_wakeup.clear()
            pending, _ = await self._guard_backlog()
            try:
                if pending and not self.guard_ready:
                    await self.foreground.wait_quiet()
                    await model.call("score", ["warm-up"], kill_after=GUARD_WARMUP_SECONDS)
                    self.guard_ready = True
                    self.guard_errors.pop("*", None)
                for paper_id in pending:
                    try:
                        await self.score_paper(paper_id)
                        self.guard_errors.pop(paper_id, None)
                    except Exception as exc:
                        logger.warning("Guard scoring failed for %s", paper_id, exc_info=True)
                        self.guard_errors[paper_id] = str(exc) or type(exc).__name__
            except Exception as exc:
                logger.warning("Guard model %s failed to load", model.name, exc_info=True)
                self.guard_errors["*"] = str(exc) or type(exc).__name__
            try:
                timeout = GUARD_RETRY_SECONDS if self.guard_errors else None
                await asyncio.wait_for(self.guard_wakeup.wait(), timeout)
            except TimeoutError:
                pass

    async def guard_status(self) -> Document:
        """The guard's health for the system pane, shaped like an answer model's status."""
        config = self.guard_config
        if self.guard_model is None:
            return {
                "plugin": "none",
                "status": "disabled",
                "detail": "The injection classifier is off; instruction-like text is caught "
                "by the regex filter only.",
            }
        pending, total = await self._guard_backlog()
        rows = await self.db.fetch(
            context("guard-health", 3.0),
            "SELECT COALESCE(SUM(untrusted), 0) AS untrusted FROM arie_guard_scores "
            "WHERE model = ?",
            (config.model_id,),
        )
        untrusted = int(str(rows[0]["untrusted"])) if rows else 0
        error = self.guard_errors.get("*") or next(
            (self.guard_errors[paper_id] for paper_id in pending if paper_id in self.guard_errors),
            None,
        )
        scored = total - len(pending)
        if error:
            status, detail = "degraded", f"{config.model} failing: {error}."
        elif not self.guard_ready and pending:
            status, detail = "loading", f"Loading {config.model}."
        else:
            status, detail = "healthy", "Instruction-like chunks are scored and left out."
        return {
            "plugin": config.use,
            "model": config.model,
            "status": status,
            "detail": f"{detail} {scored} of {total} papers scored; {untrusted} chunks "
            f"untrusted at threshold {config.threshold:g}.",
            "threshold": config.threshold,
            "papers_scored": scored,
            "papers_total": total,
            "untrusted_chunks": untrusted,
        }

    def close_guard(self) -> None:
        if self.guard_model is not None:
            self.guard_model.close()


__all__ = ["GUARD_BATCH", "PaperGuard", "paper_revision"]
