"""Propose a better explain prompt offline with GEPA (S17).

    uv run python -m arie_api.optimize_prompts explain --max-calls 200 [--cases N] [--seed S]

GEPA reads how the explain writer's rules fail on a set of cases (key facts it missed,
citation flags, the support judge's reasons, a reader's thumbs-down reasons) and rewrites the
rules. The cases are every saved fact case of the library (Qasper, a paper's own, scored
feedback). A seeded split keeps some out of GEPA: they score the old and the new rules after
the search, so the report shows whether a gain on the train cases carries over.

Nothing live changes. The best rules go to ``prompts/explain.v<N+1>.yaml``, which no code
loads; a person decides whether to switch ``explain`` to it. Every model call (writer, judge
and reflection) is counted, and one beyond ``--max-calls`` is refused. The run needs the
library's database and an LLM key, as ``eval_paper`` does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Any, cast

import msgspec
from arie_core.contracts import LLMRequest
from arie_core.harness.explain import (
    ExplainPrompts,
    ExplainResult,
    explain_answer,
    load_explain_prompts,
)
from arie_core.harness.judge import judge_support
from arie_core.harness.optimize import (
    EXCERPT_PENALTY,
    PERFECT_SCORE,
    BudgetExhausted,
    CallBudget,
    CountingLLM,
    OptCase,
    Rollout,
    check_rules,
    feedback_text,
    metric,
    next_version,
    pick_cases,
    render_candidate,
    split_cases,
    summarize_rollouts,
)
from arie_core.harness.paper_eval import score_explained
from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import JUDGE_SECONDS, ModelSession, StreamBudget
from arie_core.pipeline.pack import PaperPack
from arie_core.plugin_api import LLM
from gepa.api import optimize  # pyright: ignore[reportUnknownVariableType]  # BatchSampler[?, ?]
from gepa.core.adapter import EvaluationBatch, ProposalFn
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState

from arie_api.runtime import ResearchRuntime, context
from arie_api.runtime.base import explain_context

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ("explain",)
# The one part of the explain prompt GEPA rewrites. The shared header stays byte-identical so
# the judge and the guide keep reusing the writer's cached prefix.
COMPONENT = "writer_rules"
SEED = 17
# A rollout is the writer's call and then the support judge's.
CALLS_PER_CASE = 2
MINIBATCH = 3
REFLECTION_TOKENS = 2000
REFLECTION_BUDGET = StreamBudget(first_token=30.0, stall=30.0, total=120.0)
REFLECTION_SECONDS = REFLECTION_BUDGET.total + 5.0
ANSWER_CHARS = 1200
# What a rollout scores when its rules cannot be tried: the worst an answer can get.
WORST = -EXCERPT_PENALTY

REFLECTION_TEMPLATE = """I gave a model these writer rules. The model reads one research paper and \
explains it to a reader who asked a question:
```
<curr_param>
```

Below are questions, the model's answers and feedback on each. A score runs from -1 to 2: the \
share of key facts the answer states, plus the share of its cited sentences the paper supports, \
minus 1 if no written answer came back.
```
<side_info>
```

Write improved writer rules. They apply to every paper and every question, so keep them general: \
do not copy facts, numbers, names or sentences from the examples. Keep the rules' plain-bullet \
form and its [cN] citation marker format, keep the line that the paper is the only source for \
claims, and keep the answer in Markdown. Fix the failures the feedback names without adding \
length for its own sake. Reply with the new rules only, inside one ``` block."""


class Material(msgspec.Struct, frozen=True):
    """One paper's stored pack and its chunks: what the explain path reads."""

    pack: PaperPack
    rows: list[EvidenceChunk]


class Trace(msgspec.Struct, frozen=True):
    """What GEPA's reflection sees of one rollout."""

    question: str
    answer: str
    score: float
    feedback: str


class ExplainRollouts:
    """Answer and score one case with given writer rules, on the real explain path.

    Every model call goes through ``budget`` (the writer, the judge), so the run's cap holds
    however many rollouts the search asks for.
    """

    def __init__(
        self,
        seed: ExplainPrompts,
        papers: Mapping[str, Material],
        llm: LLM,
        budget: CallBudget,
        *,
        context_tokens: int,
        pack_share: float,
    ) -> None:
        self.seed = seed
        self.papers = papers
        self.budget = budget
        self.llm = CountingLLM(llm, budget)
        self.context_tokens = context_tokens
        self.pack_share = pack_share

    async def run(self, item: OptCase, rules: str) -> Rollout:
        material = self.papers[item.paper_id]
        prompts = msgspec.structs.replace(self.seed, writer_rules=rules.strip())
        name = f"optimize:{item.paper_id}:{item.case.case_id}"
        started = time.monotonic()
        final: ExplainResult | None = None
        stream = explain_answer(
            explain_context(name, self.context_tokens),
            item.paper_id,
            item.case.question,
            material.pack,
            material.rows,
            self.llm,
            context_tokens=self.context_tokens,
            pack_share=self.pack_share,
            prompts=prompts,
        )
        async with aclosing(stream) as pieces:
            async for piece in pieces:
                if isinstance(piece, ExplainResult):
                    final = piece
        if final is None:
            raise RuntimeError("The explain path ended without a result")
        seconds = time.monotonic() - started
        judgement = await judge_support(
            explain_context(f"{name}:judge", self.context_tokens, JUDGE_SECONDS),
            self.llm,
            final.pack_text,
            final.checks,
            context_tokens=self.context_tokens,
        )
        result = score_explained(
            item.case, material.rows, final, seconds=seconds, judgement=judgement
        )
        feedback = feedback_text(item, result, final.text, final.checks, judgement)
        return Rollout(item, final.text, result, metric(result), feedback)


class ExplainAdapter:
    """GEPA's view of the explain prompt: ``{"writer_rules": text}`` in, scores and traces out.

    GEPA is synchronous and runs in a worker thread; each rollout runs on the main event loop.
    ``seen`` lists every case GEPA was given, so a test can show the held-out ones never are.
    """

    # GEPA's own reflection proposes the new rules; this adapter does not.
    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        rollouts: ExplainRollouts,
        loop: asyncio.AbstractEventLoop,
        guard: Sequence[str],
    ) -> None:
        self.rollouts = rollouts
        self.loop = loop
        self.guard = guard
        self.seen: list[tuple[str, str]] = []

    def evaluate(
        self, batch: list[OptCase], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[Trace, str]:
        rules = candidate[COMPONENT]
        problem = check_rules(rules, self.guard)
        outputs: list[str] = []
        scores: list[float] = []
        traces: list[Trace] = []
        for item in batch:
            self.seen.append(item.key)
            if problem:
                answer, score = "", WORST
                feedback = f"These writer rules cannot be used ({problem}); no model was called."
            else:
                rollout = asyncio.run_coroutine_threadsafe(
                    self.rollouts.run(item, rules), self.loop
                ).result()
                if self.rollouts.budget.refused:
                    # A call the cap refused was scored as a failed answer; drop the batch.
                    raise BudgetExhausted("The call cap ended this batch")
                answer, score, feedback = rollout.text, rollout.score, rollout.feedback
            outputs.append(answer)
            scores.append(score)
            traces.append(Trace(item.case.question, answer[:ANSWER_CHARS], score, feedback))
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trace, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        records = [
            {
                "Inputs": {"question": trace.question},
                "Generated Outputs": trace.answer or "(no answer)",
                "Feedback": f"Score {trace.score} of {PERFECT_SCORE}. {trace.feedback}".strip(),
            }
            for trace in eval_batch.trajectories or []
        ]
        return {name: records for name in components_to_update}


class Reflector:
    """GEPA's reflection model: one counted, budgeted call on the writer's provider."""

    def __init__(
        self, llm: CountingLLM, loop: asyncio.AbstractEventLoop, context_tokens: int
    ) -> None:
        self.llm = llm
        self.loop = loop
        self.context_tokens = context_tokens

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):
            raise TypeError("The reflection prompt must be text")
        return asyncio.run_coroutine_threadsafe(self.ask(prompt), self.loop).result()

    async def ask(self, prompt: str) -> str:
        ctx = explain_context("optimize:reflect", self.context_tokens, REFLECTION_SECONDS)
        request = LLMRequest(prompt=prompt, tools=[], max_output_tokens=REFLECTION_TOKENS)
        session = ModelSession(ctx, self.llm)
        stream = session.stream(
            request, input_limit=self.context_tokens - REFLECTION_TOKENS, budget=REFLECTION_BUDGET
        )
        async with aclosing(stream) as deltas:
            return "".join([text async for text in deltas])


class CallCap:
    """GEPA's stop condition: stop once another step could eat into the held-out reserve.

    A step is two minibatch scorings, a reflection and, if the new rules are kept, a scoring
    of the whole train set. The cap itself is enforced by ``CountingLLM``, which refuses the
    call; this only lets a run end cleanly before that.
    """

    def __init__(self, budget: CallBudget, reserve: int, step: int) -> None:
        self.budget = budget
        self.reserve = reserve
        self.step = step

    def __call__(self, gepa_state: GEPAState[Any, Any]) -> bool:
        del gepa_state
        return self.budget.refused or self.budget.left < self.reserve + self.step


class StderrLog:
    def log(self, message: str) -> None:
        print(f"  [gepa] {message}", file=sys.stderr, flush=True)


class Report(msgspec.Struct, frozen=True, kw_only=True):
    """What a run found; printed as one JSON line and meant for PROGRESS."""

    prompt: str
    seed_version: str
    candidate: str
    candidate_path: str
    cap: int
    calls_used: int
    calls_reserved: int
    train_cases: int
    held_out_cases: int
    train_old: float
    train_new: float
    held_out_old: dict[str, float | int]
    held_out_new: dict[str, float | int]
    note: str = ""


def step_calls(train: int) -> int:
    """The most calls one GEPA step can use for a train set of ``train`` cases."""
    return CALLS_PER_CASE * (2 * MINIBATCH + train) + 1


def plan(train: Sequence[OptCase], held_out: Sequence[OptCase], cap: int) -> int:
    """The calls kept for scoring both prompts on the held-out cases.

    Raises ``ValueError`` if ``cap`` cannot cover the starting prompt on the train cases, both
    prompts on the held-out cases and one search step.
    """
    if not train or not held_out:
        raise ValueError("Need at least 2 fact cases to split into train and held-out")
    reserve = 2 * CALLS_PER_CASE * len(held_out)
    needed = CALLS_PER_CASE * len(train) + reserve + step_calls(len(train))
    if cap < needed:
        raise ValueError(
            f"--max-calls {cap} is too low for {len(train)} train and {len(held_out)} held-out "
            f"cases: the starting prompt on train, both prompts on held-out and one search step "
            f"take {needed}"
        )
    return reserve


def origins(items: Sequence[OptCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.case.origin] = counts.get(item.case.origin, 0) + 1
    return dict(sorted(counts.items()))


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def optimize_prompt(
    rollouts: ExplainRollouts,
    train: Sequence[OptCase],
    held_out: Sequence[OptCase],
    *,
    root: Path,
    seed_number: int = SEED,
    context_tokens: int,
) -> Report:
    """Search for better writer rules on ``train``; score old and new on ``held_out``.

    ``held_out`` is used only after the search, by this function, never by GEPA, its adapter
    or its reflection. The call cap is ``rollouts.budget``.
    """
    budget = rollouts.budget
    reserve = plan(train, held_out, budget.limit)
    seed = rollouts.seed
    say(
        f"Optimizing {seed.version}: {len(train)} train cases {origins(train)}, "
        f"{len(held_out)} held-out cases {origins(held_out)}."
    )
    say(
        f"Hard cap: {budget.limit} model calls (writer, judge and reflection), "
        f"{reserve} of them kept for scoring old and new rules on the held-out cases."
    )
    loop = asyncio.get_running_loop()
    guard = [
        *(fact for item in train for fact in item.case.key_facts),
        *{item.paper_id for item in train},
    ]
    adapter = ExplainAdapter(rollouts, loop, guard)

    def search() -> GEPAResult[str, int]:
        return optimize(
            seed_candidate={COMPONENT: seed.writer_rules},
            trainset=list(train),
            adapter=adapter,
            reflection_lm=Reflector(rollouts.llm, loop, context_tokens),
            reflection_prompt_template=REFLECTION_TEMPLATE,
            reflection_minibatch_size=MINIBATCH,
            perfect_score=PERFECT_SCORE,
            stop_callbacks=CallCap(budget, reserve, step_calls(len(train))),
            seed=seed_number,
            raise_on_exception=False,
            logger=StderrLog(),
        )

    result = await asyncio.to_thread(search)
    best = result.best_idx
    train_old, train_new = result.val_aggregate_scores[0], result.val_aggregate_scores[best]
    rules = result.candidates[best][COMPONENT]
    note = ""
    old: list[Rollout] = []
    new: list[Rollout] = []
    try:
        for item in held_out:
            old.append(await _held_out(rollouts, item, seed.writer_rules))
        for item in held_out if best else ():
            new.append(await _held_out(rollouts, item, rules))
    except BudgetExhausted:
        note = "The call cap ended the held-out scoring early."
    new = new if best else old
    path = ""
    version = ""
    if best:
        number = next_version(root / "prompts", "explain")
        held = (summarize_rollouts(old).get("score"), summarize_rollouts(new).get("score"))
        text = render_candidate(
            seed,
            rules,
            number,
            f"Seed {seed.version}; train {train_old:.3f} -> {train_new:.3f}, "
            f"held-out {held[0]} -> {held[1]}.",
        )
        target = root / "prompts" / f"explain.v{number}.yaml"
        with target.open("x", encoding="utf-8") as handle:
            handle.write(text)
        try:
            load_explain_prompts(root, number)
        except Exception:
            target.unlink()
            raise
        path, version = str(target.relative_to(root)), f"explain.v{number}"
    else:
        note = f"{note} GEPA found nothing better than the current rules; no file written.".strip()
    return Report(
        prompt="explain",
        seed_version=seed.version,
        candidate=version,
        candidate_path=path,
        cap=budget.limit,
        calls_used=budget.used,
        calls_reserved=reserve,
        train_cases=len(train),
        held_out_cases=len(held_out),
        train_old=round(train_old, 3),
        train_new=round(train_new, 3),
        held_out_old=summarize_rollouts(old),
        held_out_new=summarize_rollouts(new),
        note=note,
    )


async def _held_out(rollouts: ExplainRollouts, item: OptCase, rules: str) -> Rollout:
    rollout = await rollouts.run(item, rules)
    if rollouts.budget.refused:
        raise BudgetExhausted("The call cap ended this case")
    return rollout


async def load_items(runtime: ResearchRuntime) -> list[OptCase]:
    """Every saved or feedback fact case of the library that has key facts to score."""
    rows = await runtime.db.fetch(
        context("optimize-papers", 10.0),
        "SELECT DISTINCT paper_id FROM arie_eval_cases "
        "UNION SELECT DISTINCT paper_id FROM arie_feedback_facts",
    )
    evaluations = runtime.evaluations
    items: list[OptCase] = []
    for row in rows:
        paper_id = str(row["paper_id"])
        paper = await runtime.get_paper(paper_id)
        if paper is None or paper.get("status") != "INDEXED":
            continue
        chunks = evaluations.chunks(paper)
        items += [
            OptCase(paper_id, case)
            for case in await evaluations.cases(paper_id)
            if case.kind == "fact" and case.key_facts
        ]
        items += [
            OptCase(paper_id, case, reasons)
            for case, reasons in await evaluations.feedback_cases(paper_id, chunks)
            if case.kind == "fact"
        ]
    return items


async def load_materials(runtime: ResearchRuntime, paper_ids: set[str]) -> dict[str, Material]:
    materials: dict[str, Material] = {}
    for paper_id in sorted(paper_ids):
        paper = await runtime.get_paper(paper_id)
        if paper is None:
            continue
        materials[paper_id] = Material(
            await runtime.ensure_pack(paper), runtime.evaluations.chunks(paper)
        )
    return materials


async def run(prompt: str, max_calls: int, cases: int | None, seed_number: int) -> Report:
    if prompt not in PROMPTS:
        raise ValueError(f"Only {', '.join(PROMPTS)} can be optimized")
    runtime = ResearchRuntime(ROOT)
    try:
        await runtime.initialize()
        llm = cast(LLM, runtime.instance("llm"))
        if not getattr(llm, "api_key", None):
            raise ValueError("Optimizing needs a model provider API key")
        items = await load_items(runtime)
        train, held_out = split_cases(pick_cases(items, cases, seed_number), seed=seed_number)
        plan(train, held_out, max_calls)
        papers = await load_materials(runtime, {item.paper_id for item in [*train, *held_out]})
        context_tokens = runtime.config.context_tokens_for(runtime.llm_provider())
        rollouts = ExplainRollouts(
            load_explain_prompts(ROOT),
            papers,
            llm,
            CallBudget(max_calls),
            context_tokens=context_tokens,
            pack_share=runtime.config.explain.max_pack_share,
        )
        report = await optimize_prompt(
            rollouts,
            train,
            held_out,
            root=ROOT,
            seed_number=seed_number,
            context_tokens=context_tokens,
        )
    finally:
        runtime.close()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose a better prompt offline with GEPA.")
    parser.add_argument("prompt", choices=PROMPTS)
    parser.add_argument("--max-calls", type=int, required=True, help="hard cap on model calls")
    parser.add_argument("--cases", type=int, help="use at most this many cases")
    parser.add_argument("--seed", type=int, default=SEED, help="fixes the train/held-out split")
    args = parser.parse_args(argv)
    if args.max_calls < 1 or (args.cases is not None and args.cases < 2):
        parser.exit(2, "error: --max-calls must be at least 1 and --cases at least 2\n")
    return args


def main() -> None:
    args = parse_args()
    try:
        report = asyncio.run(run(args.prompt, args.max_calls, args.cases, args.seed))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(msgspec.to_builtins(report)))


if __name__ == "__main__":
    main()
