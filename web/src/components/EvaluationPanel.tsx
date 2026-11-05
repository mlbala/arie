import { useEffect, useState } from "react";
import { CircleCheck, CircleSlash, CircleX, FlaskConical, LoaderCircle, ThumbsDown } from "lucide-react";
import {
  getEvalRun,
  getPaperEvaluation,
  startPaperEvaluation,
  type EvalResult,
  type EvalRun,
  type EvalSummary,
  type PaperEvaluation,
} from "../api/client";
import { FeedbackPanel } from "./FeedbackPanel";

const ACTIVE = new Set(["queued", "running"]);
const VARIANT_LABEL: Record<EvalResult["variant"], string> = {
  configured: "As configured",
  no_reranker: "Without reranker",
};

/** How the judge scored an explanation against the strict answer, by `win_rate`. */
const VERSUS: Record<number, string> = { 1: "better than strict", 0.5: "tied with strict", 0: "worse than strict" };

/** Answer stages in pipeline order; `total` is left out because it contains the others. */
const STAGES = ["plan", "vector", "read", "rerank", "write", "tier3", "review"];

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;
}

/** The stage that took longest, or null when the result has no trace (runs saved before traces existed). */
function slowestStage(result: EvalResult): { stage: string; seconds: number } | null {
  const seconds = result.trace?.seconds;
  if (!seconds) return null;
  let slowest: { stage: string; seconds: number } | null = null;
  for (const stage of STAGES) {
    const value = seconds[stage] ?? 0;
    if (value > (slowest?.seconds ?? 0)) slowest = { stage, seconds: value };
  }
  return slowest;
}

/** Fell back to excerpts, as opposed to a refusal, which can also carry a reason. */
function fellBack(result: EvalResult): boolean {
  return Boolean(result.fallback_reason) && result.mode !== "abstention";
}

/**
 * Evaluate ARIE on this paper, with the paper's own text as the ground truth.
 *
 * Questions are written once from exact sentences of the paper and reused, so runs stay
 * comparable after a setting changes. Strict answers are scored without a model judge; an
 * explain run adds the judge's citation support and its comparison with the strict answer.
 */
export function EvaluationPanel({ paperId, ready }: { paperId: string; ready: boolean }) {
  const [overview, setOverview] = useState<PaperEvaluation | null>(null);
  const [run, setRun] = useState<EvalRun | null>(null);
  const [compare, setCompare] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getPaperEvaluation(paperId)
      .then((data) => {
        if (cancelled) return;
        setOverview(data);
        setRun(data.latest);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Could not load the evaluation.");
      });
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const running = run !== null && ACTIVE.has(run.status);
  const runId = run?.run_id;
  useEffect(() => {
    if (!running || !runId) return;
    const timer = window.setInterval(() => {
      getEvalRun(paperId, runId)
        .then((next) => {
          setRun(next);
          if (!ACTIVE.has(next.status)) void getPaperEvaluation(paperId).then(setOverview).catch(() => undefined);
        })
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [paperId, runId, running]);

  const start = (regenerate: boolean) => {
    setError("");
    startPaperEvaluation(paperId, { regenerate, compare_reranker: compare })
      .then(setRun)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Could not start the evaluation."));
  };

  const hasCases = Boolean(overview?.cases.length);
  const variants = run ? (Object.keys(run.summary) as EvalResult["variant"][]) : [];
  return (
    <section className="evaluation-panel" aria-label="Evaluate answers on this paper">
      <div className="evaluation-intro">
        <FlaskConical size={14} />
        <p>
          This paper is the ground truth. ARIE writes questions from exact sentences in it, answers them the
          same way as chat, and scores each answer by its quotes and key facts, without a model judge.
        </p>
      </div>
      <div className="evaluation-actions">
        <button className="button button-primary" disabled={!ready || running} onClick={() => start(false)}>
          {running ? <LoaderCircle className="spin" size={13} /> : null}
          {running ? "Evaluating…" : hasCases ? "Run evaluation" : "Write questions and run"}
        </button>
        {hasCases && (
          <button className="button" disabled={!ready || running} onClick={() => start(true)}>
            Write new questions
          </button>
        )}
        <label className="evaluation-compare">
          <input type="checkbox" checked={compare} disabled={running} onChange={(event) => setCompare(event.target.checked)} />
          Also answer without the reranker
        </label>
      </div>
      {error && <p className="answer-error" role="alert">{error}</p>}
      {run && (
        <div className="evaluation-run">
          <p className="evaluation-status">
            <span className={`health-pill ${run.status === "complete" ? "healthy" : ACTIVE.has(run.status) ? "running" : "degraded"}`}>{run.status}</span>
            {ACTIVE.has(run.status) ? `${run.message} · ${run.progress.done}/${run.progress.total}` : run.message}
            <small>
              {Object.entries(run.settings).map(([slot, name]) => `${slot}: ${name}`).join(" · ")}
            </small>
          </p>
          {variants.map((variant) => (
            <div key={variant}>
              <SummaryRow label={VARIANT_LABEL[variant]} summary={run.summary[variant]} />
              <FallbackCounts label={VARIANT_LABEL[variant]} results={run.results.filter((result) => result.variant === variant)} />
            </div>
          ))}
          {run.results.length > 0 && (
            <ol className="evaluation-results">
              {run.results.map((result, index) => <ResultRow key={`${result.variant}-${result.case_id}-${index}`} result={result} showVariant={variants.length > 1} />)}
            </ol>
          )}
        </div>
      )}
      {overview && (
        <FeedbackPanel
          paperId={paperId}
          feedback={overview.feedback}
          implicit={overview.implicit_feedback ?? []}
          agreement={overview.agreement}
          onChanged={() => {
            // A saved case changes the next run, not the one on screen.
            void getPaperEvaluation(paperId).then(setOverview).catch(() => undefined);
          }}
        />
      )}
    </section>
  );
}

function SummaryRow({ label, summary }: { label: string; summary: EvalSummary | undefined }) {
  if (!summary) return null;
  return (
    <dl className="evaluation-summary" aria-label={`${label} summary`}>
      <div><dt>{label}</dt><dd>{summary.passed}/{summary.scored} passed</dd></div>
      <div><dt>Grounded</dt><dd>{percent(summary.grounded_rate)}</dd></div>
      <div><dt>Cited the source</dt><dd>{percent(summary.evidence_hit_rate)}</dd></div>
      <div><dt>Key facts</dt><dd>{percent(summary.mean_fact_recall)}</dd></div>
      <div><dt>Refused correctly</dt><dd>{percent(summary.abstain_correct_rate)}</dd></div>
      <div><dt>Median time</dt><dd>{summary.median_seconds === null ? "–" : `${summary.median_seconds} s`}</dd></div>
      {(summary.citation_support != null || summary.win_rate != null || Boolean(summary.judge_failed)) && (
        <>
          <div><dt>Citation support</dt><dd>{percent(summary.citation_support)}</dd></div>
          <div>
            <dt>Wins against strict</dt>
            <dd>{percent(summary.win_rate)}{summary.win_cases ? ` of ${summary.win_cases}` : ""}</dd>
          </div>
          {Boolean(summary.judge_failed) && <div><dt>Judge failed</dt><dd>{summary.judge_failed}</dd></div>}
        </>
      )}
    </dl>
  );
}

/** How many answers fell back to excerpts, by reason; says so when the run did not record reasons. */
function FallbackCounts({ label, results }: { label: string; results: EvalResult[] }) {
  if (results.length === 0) return null;
  const counts = new Map<string, number>();
  for (const result of results) {
    if (fellBack(result)) counts.set(result.fallback_reason!, (counts.get(result.fallback_reason!) ?? 0) + 1);
  }
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  const text = results.every((result) => result.fallback_reason === undefined)
    ? "Fallback reasons were not recorded for this run."
    : total === 0
      ? "No answer fell back to excerpts."
      : `${total} of ${results.length} fell back to excerpts: ${[...counts].map(([reason, count]) => `${reason} ×${count}`).join(", ")}`;
  return <p className="evaluation-fallbacks" aria-label={`${label} fallbacks`}>{text}</p>;
}

function ResultRow({ result, showVariant }: { result: EvalResult; showVariant: boolean }) {
  const slowest = slowestStage(result);
  const icon = result.passed === null ? <ThumbsDown size={13} /> : result.passed ? <CircleCheck size={13} /> : result.kind === "abstain" ? <CircleSlash size={13} /> : <CircleX size={13} />;
  const state = result.passed === null ? "review" : result.passed ? "passed" : "failed";
  const detail = result.error
    ? result.error
    : result.kind === "abstain"
      ? result.abstained ? "Refused, as expected" : "Answered a question it should refuse"
      : result.kind === "feedback"
        ? "You rated an earlier answer down; review this one"
        : result.abstained
          ? "No answer"
          : `${result.grounded ? "Grounded" : "Quote mismatch"} · ${result.evidence_hit ? "cited the source sentence" : "did not cite the source sentence"} · key facts ${percent(result.fact_recall)}`;
  return (
    <li className={`evaluation-result ${state}`}>
      <div>
        {icon}
        <strong>{result.question}</strong>
        {showVariant && <span className="evaluation-variant">{VARIANT_LABEL[result.variant]}</span>}
      </div>
      <p>{detail}{result.cited_pages.length ? ` · p. ${result.cited_pages.join(", ")}` : ""}</p>
      <p className="evaluation-meta">
        {result.mode && <span>{result.mode}</span>}
        {result.fallback_reason && <span className="evaluation-fallback">{`${fellBack(result) ? "fallback" : "reason"}: ${result.fallback_reason}`}</span>}
        <span>{result.reranked ? "reranked" : "not reranked"}</span>
        {result.seconds ? <span>{`${result.seconds} s`}</span> : null}
        {result.citation_support != null && <span>{`citation support ${percent(result.citation_support)}`}</span>}
        {result.judge_status === "failed" && <span className="evaluation-fallback">judge failed</span>}
        {result.win_rate != null && <span>{VERSUS[result.win_rate] ?? `vs strict ${result.win_rate}`}</span>}
        {slowest && <span>{`slowest: ${slowest.stage} ${slowest.seconds.toFixed(1)} s`}</span>}
      </p>
      {result.answer && <details><summary>Answer</summary><p>{result.answer}</p></details>}
    </li>
  );
}
