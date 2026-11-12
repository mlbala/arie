"""Per-paper evaluation runs and answer feedback, stored next to the library.

A user picks a paper; that paper is the ground truth. The first run asks the configured
LLM to write questions answered by exact sentences of the paper (see
``arie_core.harness.paper_eval``), saves them, and later runs reuse the same questions
so results stay comparable across settings. Each question is answered through
``ResearchRuntime.answer_paper_question``, the same path as chat, and scored without a
model judge. An explain run also asks the judge (S13) whether each cited sentence is
supported and, on fact cases, whether the explanation beats the strict answer. Answers a
user marked unhelpful join the run as review cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Sequence
from contextlib import aclosing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import msgspec
from arie_core.harness.explain import ExplainResult
from arie_core.harness.feedback import REASONS, Rated, agreement, answer_facts, judge_opinion
from arie_core.harness.injection import SCORE_KEY, evidence_chunks
from arie_core.harness.judge import (
    FAILED,
    NO_ANSWER,
    PairwiseJudgement,
    SupportJudgement,
    strip_markers,
)
from arie_core.harness.paper_eval import (
    PaperEvalCase,
    PaperEvalResult,
    answer_text,
    check_feedback_facts,
    error_result,
    feedback_case,
    fixed_cases,
    generate_cases,
    injected_case,
    locate_sentence,
    score_case,
    score_explained,
    summarize,
)
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import EXPLAIN_SECONDS
from arie_core.plugin_api import LLM

if TYPE_CHECKING:
    from arie_api.runtime import ResearchRuntime

logger = logging.getLogger(__name__)

Document = dict[str, Any]
RUNS_KEPT = 10
ANSWER_SECONDS = 25.0
RATINGS = {"up", "down"}
# Columns added to ``arie_feedback`` after its first version, with their defaults.
FEEDBACK_COLUMNS = (
    ("reasons_json", "[]"),
    ("mode", ""),
    ("task", ""),
    ("prompt_version", ""),
    ("trace_id", ""),
)
MODES = ("strict", "explain")


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
        # S15 added what an answer recorded about itself and why it was rated down.
        for column, default in FEEDBACK_COLUMNS:
            try:
                await db.fetch(ctx, f"SELECT {column} FROM arie_feedback LIMIT 1")
            except Exception:
                await db.execute(
                    ctx,
                    f"ALTER TABLE arie_feedback ADD COLUMN {column} TEXT NOT NULL DEFAULT "
                    f"'{default}'",
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
            """SELECT turn_id, question, rating, note, created_at, reasons_json, mode, task,
            prompt_version, trace_id FROM arie_feedback
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
                **{key: value for key, value in dict(row).items() if key != "reasons_json"},
                "reasons": cast(list[str], json.loads(str(row["reasons_json"]))),
                "key_facts": cast(list[str], json.loads(str(found["key_facts_json"])))
                if (found := attached.get(str(row["turn_id"])))
                else [],
                "source_sentence": str(found["source_sentence"]) if found else "",
            }
            for row in rows
        ]

    async def feedback_cases(
        self, paper_id: str, rows: Sequence[EvidenceChunk]
    ) -> list[tuple[PaperEvalCase, tuple[str, ...]]]:
        """Answers rated down, re-asked as cases, each with the reasons the reader gave.

        Once a person attached key facts to one it is scored like a fact case, otherwise it
        is only shown for review (S15).
        """
        return [
            (
                feedback_case(
                    str(item["turn_id"]),
                    str(item["question"]),
                    str(item["note"]),
                    cast(list[str], item["key_facts"]),
                    locate_sentence(rows, paper_id, str(item["source_sentence"])),
                ),
                tuple(cast(list[str], item["reasons"])),
            )
            for item in await self.feedback(paper_id)
            if item["rating"] == "down"
        ]

    async def implicit_feedback(self, paper_id: str) -> list[Document]:
        """Answers whose follow-up read as a negative rating, newest first (S15).

        Turns the reader rated explicitly are left out: the rating wins over the guess.
        """
        rows = await self.runtime.db.fetch(
            self._ctx(f"implicit-feedback:{paper_id}"),
            """SELECT turn_id, question, signal, follow_up, created_at
            FROM arie_implicit_feedback WHERE paper_id = ? AND turn_id NOT IN
            (SELECT turn_id FROM arie_feedback WHERE paper_id = ?) ORDER BY created_at DESC""",
            (paper_id, paper_id),
        )
        return [dict(row) for row in rows]

    async def agreement(self) -> Document:
        """How often the judge's verdict matched the reader's rating, over every rated answer.

        Only explicit ratings count; the judge has a verdict only on explain answers it judged
        (``judge_opinion``). ``implicit`` is the number of unrated answers a follow-up
        questioned, shown beside the agreement and never part of it.
        """
        ctx = self._ctx("feedback-agreement")
        rows = await self.runtime.db.fetch(
            ctx, "SELECT rating, task, answer_json FROM arie_feedback ORDER BY created_at"
        )
        rated: list[Rated] = []
        for row in rows:
            events = cast(list[Document], json.loads(str(row["answer_json"])))
            # Ratings saved before S15 have no task column; the saved events still say it.
            task = str(row["task"]) or answer_facts(events).task
            rated.append(Rated(str(row["rating"]), task, judge_opinion(events)))
        implicit = await self.runtime.db.fetch(
            ctx,
            """SELECT COUNT(*) AS n FROM arie_implicit_feedback WHERE turn_id NOT IN
            (SELECT turn_id FROM arie_feedback)""",
        )
        return {**msgspec.to_builtins(agreement(rated)), "implicit": cast(int, implicit[0]["n"])}

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
            "implicit_feedback": await self.implicit_feedback(paper_id),
            # Library-wide, not this paper's: gate 4 counts every rated answer.
            "agreement": await self.agreement(),
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
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
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
        rows = await self.runtime.evidence(paper)
        cases = [] if regenerate else await self.cases(paper_id)
        if not cases:
            run.update(status="running", message="Writing questions from the paper's own sentences")
            cases = await self._generate(paper, rows)
        review = [case for case, _ in await self.feedback_cases(paper_id, rows)]
        injected, planted, planted_rows = await self._planted(paper, rows)
        variants = [("configured", True)]
        # Explain mode has no reranker, so a second variant would repeat the first.
        if (
            run["mode"] == "strict"
            and run["compare_reranker"]
            and self.runtime.models.get("reranker")
        ):
            variants.append(("no_reranker", False))
        work = [
            (case, variant, rerank)
            for case in [*cases, *review, injected]
            for variant, rerank in variants
        ]
        run.update(status="running", message="Answering", progress={"done": 0, "total": len(work)})
        scored: dict[str, list[PaperEvalResult]] = {variant: [] for variant, _ in variants}
        for case, variant, rerank in work:
            # The injected case is answered from the copy of the paper with the planted chunk.
            doc, doc_rows = (planted, planted_rows) if case is injected else (paper, rows)
            result = await self._answer(doc, doc_rows, case, rerank, str(run["mode"]))
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
        await self.save_cases(paper_id, cases)
        return cases

    async def save_cases(self, paper_id: str, cases: Sequence[PaperEvalCase]) -> None:
        """Replace the paper's saved cases; later runs reuse them instead of writing new ones."""
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

    async def _planted(
        self, paper: Document, rows: Sequence[EvidenceChunk]
    ) -> tuple[PaperEvalCase, Document, list[EvidenceChunk]]:
        """The ``injected`` case, an in-memory copy of the paper with its planted chunk, and
        that copy's chunks as answers load them.

        With the guard on, the planted text is scored by the live guard model, so the case
        measures the model and the filter together. Without it (or if the call fails) the
        chunk is unscored and only the regex can drop it. The copy is never saved; its extra
        chunk gives it its own pack revision, so the real paper's pack is rebuilt on next use.
        """
        case, chunk = injected_case(str(paper["id"]), rows)
        guard = self.runtime.guard_model
        if guard is not None:
            try:
                scores = await guard.scores("score", [case.answer_quote], count=1, kill_after=600.0)
                chunk[SCORE_KEY] = scores[0]
            except Exception:
                logger.warning("Guard could not score the planted chunk", exc_info=True)
        planted: Document = {**paper, "chunks": [*paper.get("chunks", []), chunk]}
        return case, planted, await self.runtime.evidence(planted)

    @staticmethod
    def chunks(paper: Document) -> list[EvidenceChunk]:
        """The paper's chunks without guard scores; runs use ``runtime.evidence`` instead."""
        return evidence_chunks(str(paper["id"]), paper.get("chunks", []), None)

    async def _answer(
        self,
        paper: Document,
        rows: Sequence[EvidenceChunk],
        case: PaperEvalCase,
        rerank: bool,
        mode: str,
    ) -> PaperEvalResult:
        started = time.monotonic()
        if mode == "explain":
            return await self._explain(paper, rows, case, started)
        try:
            async with asyncio.timeout(ANSWER_SECONDS):
                run = await self.runtime.answer_paper_question(
                    self._ctx(f"eval:{case.case_id}", ANSWER_SECONDS),
                    paper,
                    case.question,
                    rerank=rerank,
                )
        except Exception as exc:
            return error_result(
                case,
                mode,
                time.monotonic() - started,
                "Timed out after 25 s" if isinstance(exc, TimeoutError) else type(exc).__name__,
            )
        return score_case(
            case,
            rows,
            run.answer,
            seconds=time.monotonic() - started,
            reranked=run.reading.reranked,
            requested_mode=mode,
        )

    async def _explain(
        self, paper: Document, rows: Sequence[EvidenceChunk], case: PaperEvalCase, started: float
    ) -> PaperEvalResult:
        """Answer one case in explain mode, without chat history, and score the written text."""
        result: ExplainResult | None = None
        error = ""
        try:
            async with asyncio.timeout(EXPLAIN_SECONDS):
                stream = self.runtime.explain_paper_question(
                    f"eval:{case.case_id}", paper, case.question
                )
                async with aclosing(stream) as items:
                    async for item in items:
                        if isinstance(item, ExplainResult):
                            result = item
        except Exception as exc:
            error = "Timed out" if isinstance(exc, TimeoutError) else type(exc).__name__
        if result is None:
            return error_result(case, "explain", time.monotonic() - started, error)
        # ``seconds`` is the explanation alone; the judge calls below have their own budgets.
        seconds = time.monotonic() - started
        trace_id = f"eval:{case.case_id}"
        try:
            judgement = await self.runtime.judge_explanation(trace_id, result)
        except Exception as exc:
            logger.warning("Support judge crashed for %s", case.case_id, exc_info=True)
            judgement = SupportJudgement(status=FAILED, fallback_reason=type(exc).__name__)
        pairwise = await self._pairwise(paper, case, result)
        return score_explained(
            case, rows, result, seconds=seconds, judgement=judgement, pairwise=pairwise
        )

    async def _pairwise(
        self, paper: Document, case: PaperEvalCase, result: ExplainResult
    ) -> PairwiseJudgement | None:
        """Judge the explanation against a strict answer to the same question, both ways round.

        Only fact cases have key facts to judge against, and only a written explanation can
        win or lose; any other case gets ``None``. A strict answer that raises is not a loss
        for strict, so the judgement fails and the case is left out of ``win_rate``.
        """
        written = result.mode in {"explain", "explain_partial"} and result.text.strip()
        if case.kind != "fact" or not written:
            return None
        try:
            async with asyncio.timeout(ANSWER_SECONDS):
                strict = await self.runtime.answer_paper_question(
                    self._ctx(f"eval-strict:{case.case_id}", ANSWER_SECONDS), paper, case.question
                )
            return await self.runtime.judge_explain_vs_strict(
                f"eval:{case.case_id}",
                case.question,
                case.key_facts,
                strip_markers(result.text),
                answer_text(strict.answer) or NO_ANSWER,
            )
        except Exception as exc:
            logger.warning("Pairwise judging failed for %s", case.case_id, exc_info=True)
            reason = "strict_timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
            return PairwiseJudgement(status=FAILED, fallback_reason=reason)

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
        self,
        paper_id: str,
        turn_id: str,
        rating: str | None,
        note: str = "",
        reasons: Sequence[str] = (),
    ) -> Document:
        """Rate a saved answer ``up`` or ``down``, or clear the rating with ``None``.

        The question and streamed answer are copied, so the rating survives when old
        chat turns are pruned. ``reasons`` (from ``REASONS``, in the order given, repeats
        dropped) and ``note`` (the comment) explain a ``down`` rating. The answer's mode, task,
        prompt version and trace id are read from its saved events and stored beside the
        rating. Raises ``LookupError`` for an unknown turn and ``ValueError`` for a reason
        that is not listed or a reason given with an ``up`` rating.
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
        unknown = [reason for reason in reasons if reason not in REASONS]
        if unknown:
            raise ValueError(f"Unknown reason: {unknown[0]}")
        if reasons and rating != "down":
            raise ValueError("Reasons go with a down rating")
        kept = list(dict.fromkeys(reasons))
        rows = await self.runtime.db.fetch(
            ctx,
            "SELECT question, events_json FROM arie_chat_turns WHERE turn_id = ? AND paper_id = ?",
            (turn_id, paper_id),
        )
        if not rows:
            raise LookupError("Chat turn not found")
        facts = answer_facts(cast(list[Document], json.loads(str(rows[0]["events_json"]))))
        await self.runtime.db.execute(
            ctx,
            """INSERT INTO arie_feedback (turn_id, paper_id, question, rating, note, answer_json,
            created_at, reasons_json, mode, task, prompt_version, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET rating = excluded.rating,
            note = excluded.note, created_at = excluded.created_at,
            reasons_json = excluded.reasons_json, mode = excluded.mode, task = excluded.task,
            prompt_version = excluded.prompt_version, trace_id = excluded.trace_id""",
            (
                turn_id,
                paper_id,
                str(rows[0]["question"]),
                rating,
                note[:500],
                str(rows[0]["events_json"]),
                _now(),
                json.dumps(kept),
                facts.mode,
                facts.task,
                facts.prompt_version,
                facts.trace_id,
            ),
        )
        return {
            "turn_id": turn_id,
            "rating": rating,
            "note": note[:500],
            "reasons": kept,
            "mode": facts.mode,
            "prompt_version": facts.prompt_version,
            "trace_id": facts.trace_id,
        }

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
            self.chunks(paper), paper_id, key_facts, source_sentence
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
        for table in (
            "arie_eval_cases",
            "arie_eval_runs",
            "arie_feedback",
            "arie_feedback_facts",
            "arie_implicit_feedback",
        ):
            await self.runtime.db.execute(
                ctx, f"DELETE FROM {table} WHERE paper_id = ?", (paper_id,)
            )


__all__ = ["MODES", "PaperEvaluations"]
