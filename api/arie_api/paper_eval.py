"""Per-paper evaluation runs and answer feedback, stored next to the library.

A user picks a paper; that paper is the ground truth. The first run asks the configured
LLM to write questions answered by exact sentences of the paper (see
``arie_core.harness.paper_eval``), saves them, and later runs reuse the same questions
so results stay comparable across settings. Each question is answered through
``ResearchRuntime.answer_paper_question``, the same path as chat, and scored without a
model judge. Answers a user marked unhelpful join the run as review cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import msgspec
from arie_core.harness.paper_eval import (
    PaperEvalCase,
    PaperEvalResult,
    check_feedback_facts,
    feedback_case,
    fixed_cases,
    generate_cases,
    locate_sentence,
    score_case,
    summarize,
)
from arie_core.harness.reader import EvidenceChunk
from arie_core.plugin_api import LLM

if TYPE_CHECKING:
    from arie_api.runtime import ResearchRuntime

logger = logging.getLogger(__name__)

Document = dict[str, Any]
RUNS_KEPT = 10
ANSWER_SECONDS = 25.0
RATINGS = {"up", "down"}
MODES = ("strict", "explain")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def require_available_mode(mode: str) -> None:
    """Raise ``ValueError`` for a mode an evaluation cannot run yet."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if mode == "explain":
        # Explain mode has no answer path until S09; refuse instead of scoring strict answers.
        raise ValueError("Explain mode is not available yet")


class PaperEvaluations:
    """Generate, store and run one paper's evaluation cases; record answer feedback.

    One run executes at a time across the library, so an evaluation never competes
    with itself for the LLM or the model worker: ``start`` queues a run and ``serve``,
    supervised in the API's task group, works through the queue. Runs live in memory
    while active and are saved after every case, so a restart keeps the partial result.
    """

    def __init__(self, runtime: ResearchRuntime) -> None:
        self.runtime = runtime
        self.runs: dict[str, Document] = {}
        self.queue: asyncio.Queue[tuple[Document, Document, bool]] = asyncio.Queue()

    async def initialize(self) -> None:
        db, ctx = self.runtime.db, self._ctx("eval-startup")
        await db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_eval_cases (
            case_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, case_json TEXT NOT NULL,
            created_at TEXT NOT NULL)""",
        )
        await db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_eval_runs (
            run_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL)""",
        )
        await db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_feedback (
            turn_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question TEXT NOT NULL,
            rating TEXT NOT NULL, note TEXT NOT NULL, answer_json TEXT NOT NULL,
            created_at TEXT NOT NULL)""",
        )
        await db.execute(
            ctx,
            """CREATE TABLE IF NOT EXISTS arie_feedback_facts (
            turn_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, key_facts_json TEXT NOT NULL,
            source_sentence TEXT NOT NULL, created_at TEXT NOT NULL)""",
        )
        # A run cut short by a restart is reported as interrupted, not left running.
        rows = await db.fetch(ctx, "SELECT run_id, payload_json FROM arie_eval_runs")
        for row in rows:
            run = cast(Document, json.loads(str(row["payload_json"])))
            if run.get("status") in {"queued", "running"}:
                run.update(status="interrupted", message="Interrupted by a restart.")
                await self._save_run(run)

    def _ctx(self, name: str, seconds: float = 10.0) -> Any:
        from arie_api.runtime import context

        return context(name, seconds)

    async def cases(self, paper_id: str) -> list[PaperEvalCase]:
        rows = await self.runtime.db.fetch(
            self._ctx(f"eval-cases:{paper_id}"),
            "SELECT case_json FROM arie_eval_cases WHERE paper_id = ? ORDER BY created_at, rowid",
            (paper_id,),
        )
        return [msgspec.json.decode(str(row["case_json"]), type=PaperEvalCase) for row in rows]

    async def feedback(self, paper_id: str) -> list[Document]:
        """Rated answers, newest first, with the key facts attached to them (if any)."""
        ctx = self._ctx(f"feedback:{paper_id}")
        rows = await self.runtime.db.fetch(
            ctx,
            """SELECT turn_id, question, rating, note, created_at FROM arie_feedback
            WHERE paper_id = ? ORDER BY created_at DESC""",
            (paper_id,),
        )
        facts = await self.runtime.db.fetch(
            ctx,
            "SELECT turn_id, key_facts_json, source_sentence FROM arie_feedback_facts "
            "WHERE paper_id = ?",
            (paper_id,),
        )
        attached = {str(row["turn_id"]): row for row in facts}
        return [
            {
                **dict(row),
                "key_facts": cast(list[str], json.loads(str(found["key_facts_json"])))
                if (found := attached.get(str(row["turn_id"])))
                else [],
                "source_sentence": str(found["source_sentence"]) if found else "",
            }
            for row in rows
        ]

    async def overview(self, paper_id: str) -> Document:
        """Saved cases, recent run summaries (newest first) and feedback for one paper."""
        rows = await self.runtime.db.fetch(
            self._ctx(f"eval-runs:{paper_id}"),
            """SELECT payload_json FROM arie_eval_runs WHERE paper_id = ?
            ORDER BY created_at DESC LIMIT ?""",
            (paper_id, RUNS_KEPT),
        )
        runs = [cast(Document, json.loads(str(row["payload_json"]))) for row in rows]
        return {
            "paper_id": paper_id,
            "cases": [msgspec.to_builtins(case) for case in await self.cases(paper_id)],
            "runs": [
                {key: value for key, value in run.items() if key != "results"} for run in runs
            ],
            "latest": runs[0] if runs else None,
            "feedback": await self.feedback(paper_id),
        }

    async def get_run(self, paper_id: str, run_id: str) -> Document | None:
        run = self.runs.get(run_id)
        if run is None:
            rows = await self.runtime.db.fetch(
                self._ctx(f"eval-run:{run_id}"),
                "SELECT payload_json FROM arie_eval_runs WHERE run_id = ? AND paper_id = ?",
                (run_id, paper_id),
            )
            run = cast(Document, json.loads(str(rows[0]["payload_json"]))) if rows else None
        return run if run and run.get("paper_id") == paper_id else None

    def _settings(self) -> Document:
        plugins = self.runtime.config.plugins

        def use(slot: str) -> str:
            return str(cast(dict[str, object], plugins.get(slot, {})).get("use") or "none")

        return {
            "llm": use("llm"),
            "embedder": self.runtime.embedder_id,
            "entailment": use("entailment"),
            "reranker": use("reranker"),
        }

    async def start(
        self,
        paper_id: str,
        *,
        regenerate: bool = False,
        compare_reranker: bool = False,
        mode: str = "strict",
    ) -> Document:
        """Queue a run for an indexed paper; raise ``LookupError`` or ``ValueError``."""
        run, paper = await self._new_run(paper_id, compare_reranker, mode)
        await self.queue.put((run, paper, regenerate))
        return run

    async def serve(self) -> None:
        """Run queued evaluations one at a time, forever."""
        while True:
            run, paper, regenerate = await self.queue.get()
            try:
                await self._run(run, paper, regenerate)
            finally:
                self.queue.task_done()

    async def run_to_completion(
        self,
        paper_id: str,
        *,
        regenerate: bool = False,
        compare_reranker: bool = False,
        mode: str = "strict",
    ) -> Document:
        """Run one evaluation in the caller's task and return it (for the command line)."""
        run, paper = await self._new_run(paper_id, compare_reranker, mode)
        await self._run(run, paper, regenerate)
        return run

    async def _new_run(
        self, paper_id: str, compare_reranker: bool, mode: str
    ) -> tuple[Document, Document]:
        require_available_mode(mode)
        paper = await self.runtime.get_paper(paper_id)
        if paper is None:
            raise LookupError("Paper not found")
        if paper.get("status") != "INDEXED":
            raise ValueError("Wait for PDF extraction and indexing to finish")
        if any(run["status"] in {"queued", "running"} for run in self.runs.values()):
            raise ValueError("An evaluation is already running; wait for it to finish")
        run: Document = {
            "run_id": uuid.uuid4().hex,
            "paper_id": paper_id,
            "status": "queued",
            "message": "Queued",
            "created_at": _now(),
            "finished_at": None,
            "settings": self._settings(),
            "mode": mode,
            "compare_reranker": compare_reranker,
            "progress": {"done": 0, "total": 0},
            "summary": {},
            "results": [],
        }
        self.runs[run["run_id"]] = run
        await self._save_run(run)
        return run, paper

    async def _run(self, run: Document, paper: Document, regenerate: bool) -> None:
        try:
            await self._execute(run, paper, regenerate)
            run.update(status="complete", message="Evaluation complete")
        except Exception as exc:
            logger.warning("Evaluation of %s failed", run["paper_id"], exc_info=True)
            run.update(status="failed", message=str(exc) or type(exc).__name__)
        finally:
            run["finished_at"] = _now()
            await self._save_run(run)
            self.runs.pop(run["run_id"], None)

    async def _execute(self, run: Document, paper: Document, regenerate: bool) -> None:
        paper_id = str(paper["id"])
        rows = self._chunks(paper)
        cases = [] if regenerate else await self.cases(paper_id)
        if not cases:
            run.update(status="running", message="Writing questions from the paper's own sentences")
            cases = await self._generate(paper, rows)
        # Answers rated down are re-asked; once a person attached key facts to one, it is
        # scored like a fact case, otherwise it is only shown for review.
        review = [
            feedback_case(
                str(item["turn_id"]),
                str(item["question"]),
                str(item["note"]),
                cast(list[str], item["key_facts"]),
                locate_sentence(rows, paper_id, str(item["source_sentence"])),
            )
            for item in await self.feedback(paper_id)
            if item["rating"] == "down"
        ]
        variants = [("configured", True)]
        if run["compare_reranker"] and self.runtime.models.get("reranker"):
            variants.append(("no_reranker", False))
        work = [
            (case, variant, rerank) for case in [*cases, *review] for variant, rerank in variants
        ]
        run.update(status="running", message="Answering", progress={"done": 0, "total": len(work)})
        scored: dict[str, list[PaperEvalResult]] = {variant: [] for variant, _ in variants}
        for case, variant, rerank in work:
            result = await self._answer(paper, rows, case, rerank, str(run["mode"]))
            scored[variant].append(result)
            run["results"].append({"variant": variant, **msgspec.to_builtins(result)})
            run["progress"]["done"] += 1
            run["summary"] = {name: summarize(items) for name, items in scored.items()}
            await self._save_run(run)

    async def _generate(
        self, paper: Document, rows: Sequence[EvidenceChunk]
    ) -> list[PaperEvalCase]:
        paper_id = str(paper["id"])
        llm = cast(LLM, self.runtime.instance("llm"))
        if not getattr(llm, "api_key", None):
            raise ValueError("Writing evaluation questions needs a model provider API key")
        generated = await generate_cases(
            self._ctx(f"eval-generate:{paper_id}", 45.0),
            paper_id,
            str(paper.get("title") or ""),
            rows,
            llm,
            project_root=self.runtime.root,
        )
        cases = [*generated, *fixed_cases(paper_id, rows)]
        ctx = self._ctx(f"eval-save:{paper_id}")
        await self.runtime.db.execute(
            ctx, "DELETE FROM arie_eval_cases WHERE paper_id = ?", (paper_id,)
        )
        for case in cases:
            await self.runtime.db.execute(
                ctx,
                "INSERT OR REPLACE INTO arie_eval_cases VALUES (?, ?, ?, ?)",
                (case.case_id, paper_id, msgspec.json.encode(case).decode(), _now()),
            )
        return cases

    @staticmethod
    def _chunks(paper: Document) -> list[EvidenceChunk]:
        paper_id = str(paper["id"])
        return [
            EvidenceChunk(**row)
            for row in paper.get("chunks", [])
            if row.get("paper_id") == paper_id
        ]

    async def _answer(
        self,
        paper: Document,
        rows: Sequence[EvidenceChunk],
        case: PaperEvalCase,
        rerank: bool,
        mode: str,
    ) -> PaperEvalResult:
        started = time.monotonic()
        try:
            async with asyncio.timeout(ANSWER_SECONDS):
                run = await self.runtime.answer_paper_question(
                    self._ctx(f"eval:{case.case_id}", ANSWER_SECONDS),
                    paper,
                    case.question,
                    rerank=rerank,
                )
        except Exception as exc:
            return PaperEvalResult(
                case_id=case.case_id,
                question=case.question,
                kind=case.kind,
                passed=False if case.kind != "feedback" else None,
                grounded=False,
                evidence_hit=False,
                fact_recall=0.0,
                abstained=True,
                mode="error",
                claims=0,
                model_calls=0,
                seconds=round(time.monotonic() - started, 2),
                error="Timed out after 25 s"
                if isinstance(exc, TimeoutError)
                else type(exc).__name__,
                passed_written=False if case.kind == "fact" else None,
                requested_mode=mode,
            )
        return score_case(
            case,
            rows,
            run.answer,
            seconds=time.monotonic() - started,
            reranked=run.reading.reranked,
            requested_mode=mode,
        )

    async def _save_run(self, run: Document) -> None:
        ctx = self._ctx(f"eval-run:{run['run_id']}")
        await self.runtime.db.execute(
            ctx,
            """INSERT INTO arie_eval_runs VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET payload_json = excluded.payload_json""",
            (run["run_id"], run["paper_id"], run["created_at"], json.dumps(run)),
        )
        await self.runtime.db.execute(
            ctx,
            """DELETE FROM arie_eval_runs WHERE paper_id = ? AND run_id NOT IN (
            SELECT run_id FROM arie_eval_runs WHERE paper_id = ?
            ORDER BY created_at DESC LIMIT ?)""",
            (run["paper_id"], run["paper_id"], RUNS_KEPT),
        )

    async def save_feedback(
        self, paper_id: str, turn_id: str, rating: str | None, note: str = ""
    ) -> Document:
        """Rate a saved answer ``up`` or ``down``, or clear the rating with ``None``.

        The question and streamed answer are copied, so the rating survives when old
        chat turns are pruned. Raises ``LookupError`` for an unknown turn.
        """
        ctx = self._ctx(f"feedback:{turn_id}")
        if rating is None:
            await self.runtime.db.execute(
                ctx,
                "DELETE FROM arie_feedback WHERE turn_id = ? AND paper_id = ?",
                (turn_id, paper_id),
            )
            await self.runtime.db.execute(
                ctx,
                "DELETE FROM arie_feedback_facts WHERE turn_id = ? AND paper_id = ?",
                (turn_id, paper_id),
            )
            return {"turn_id": turn_id, "rating": None}
        if rating not in RATINGS:
            raise ValueError("rating must be up, down or null")
        rows = await self.runtime.db.fetch(
            ctx,
            "SELECT question, events_json FROM arie_chat_turns WHERE turn_id = ? AND paper_id = ?",
            (turn_id, paper_id),
        )
        if not rows:
            raise LookupError("Chat turn not found")
        await self.runtime.db.execute(
            ctx,
            """INSERT INTO arie_feedback VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET rating = excluded.rating,
            note = excluded.note, created_at = excluded.created_at""",
            (
                turn_id,
                paper_id,
                str(rows[0]["question"]),
                rating,
                note[:500],
                str(rows[0]["events_json"]),
                _now(),
            ),
        )
        return {"turn_id": turn_id, "rating": rating, "note": note[:500]}

    async def save_feedback_facts(
        self, paper_id: str, turn_id: str, key_facts: Sequence[str], source_sentence: str = ""
    ) -> Document:
        """Attach 1-3 key facts (and optionally the paper sentence that holds them) to an answer.

        With key facts, the answer's question is scored like a fact case in the next run.
        Replaces earlier facts. Raises ``LookupError`` for an answer that was not rated and
        ``ValueError`` for facts that fail ``check_feedback_facts`` or an answer not rated down.
        """
        ctx = self._ctx(f"feedback-facts:{turn_id}")
        rated = await self.runtime.db.fetch(
            ctx,
            "SELECT rating FROM arie_feedback WHERE turn_id = ? AND paper_id = ?",
            (turn_id, paper_id),
        )
        paper = await self.runtime.get_paper(paper_id)
        if not rated or paper is None:
            raise LookupError("Rate the answer down before adding key facts")
        if rated[0]["rating"] != "down":
            raise ValueError("Only answers rated down become evaluation cases")
        facts, source = check_feedback_facts(
            self._chunks(paper), paper_id, key_facts, source_sentence
        )
        sentence = source.text if source else ""
        await self.runtime.db.execute(
            ctx,
            """INSERT INTO arie_feedback_facts VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET key_facts_json = excluded.key_facts_json,
            source_sentence = excluded.source_sentence, created_at = excluded.created_at""",
            (turn_id, paper_id, json.dumps(facts), sentence, _now()),
        )
        return {"turn_id": turn_id, "key_facts": facts, "source_sentence": sentence}

    async def forget(self, paper_id: str) -> None:
        """Delete a removed paper's cases, runs and feedback."""
        ctx = self._ctx(f"eval-forget:{paper_id}")
        for table in ("arie_eval_cases", "arie_eval_runs", "arie_feedback", "arie_feedback_facts"):
            await self.runtime.db.execute(
                ctx, f"DELETE FROM {table} WHERE paper_id = ?", (paper_id,)
            )


__all__ = ["MODES", "PaperEvaluations", "require_available_mode"]
