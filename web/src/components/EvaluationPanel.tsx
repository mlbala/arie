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

const ACTIVE = new Set(["queued", "running"]);
const VARIANT_LABEL: Record<EvalResult["variant"], string> = {
  configured: "As configured",
  no_reranker: "Without reranker",
};

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;
}

/**
 * Evaluate ARIE on this paper, with the paper's own text as the ground truth.
 *
 * Questions are written once from exact sentences of the paper and reused, so runs stay
 * comparable after a setting changes. Answers are scored without a model judge.
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
            <SummaryRow key={variant} label={VARIANT_LABEL[variant]} summary={run.summary[variant]} />
          ))}
          {run.results.length > 0 && (
            <ol className="evaluation-results">
              {run.results.map((result, index) => <ResultRow key={`${result.variant}-${result.case_id}-${index}`} result={result} showVariant={variants.length > 1} />)}
            </ol>
          )}
        </div>
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
    </dl>
  );
}

function ResultRow({ result, showVariant }: { result: EvalResult; showVariant: boolean }) {
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
      <p>{detail}{result.seconds ? ` · ${result.seconds} s` : ""}{result.cited_pages.length ? ` · p. ${result.cited_pages.join(", ")}` : ""}</p>
      {result.answer && <details><summary>Answer</summary><p>{result.answer}</p></details>}
    </li>
  );
}
