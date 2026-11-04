import { Activity, Fragment, useEffect, useRef, useState, type MouseEvent } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Check,
  FileText,
  LoaderCircle,
  Quote,
  Send,
  ShieldCheck,
  Sparkles,
  Square,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import { apiUrl, getChat, getGuide, rateAnswer, type AnswerRating, type CitationRange, type PaperGuideView, type PaperSummary } from "../api/client";
import {
  mergeSentences,
  normalizeCitations,
  normalizeSentences,
  renderExplanation,
  type ExplainCitation,
  type SentenceStatus,
} from "../explanation";
import { DEFAULT_MODE, type AnswerMode } from "../paperSessions";
import { EvaluationPanel } from "./EvaluationPanel";
import { PaperGuideMessage } from "./PaperGuide";

export interface ChatProgress {
  stage: string;
  message: string;
}
interface ChatWorkspaceProps {
  papers: PaperSummary[];
  activePaperId: string | null;
  onSelectPaper: (paperId: string) => void;
  /** Unpin a paper from the top pane; the parent shifts it into sidebar history. */
  onClosePaper: (paperId: string) => void;
  onCitation: (citation: CitationRange) => void;
  onProgress?: (paperId: string, progress: ChatProgress) => void;
  /** Answer mode per paper (from `paperSessions`); a paper without one uses DEFAULT_MODE. */
  modes?: Record<string, AnswerMode> | undefined;
  onModeChange?: ((paperId: string, mode: AnswerMode) => void) | undefined;
}
interface VerifiedClaim {
  text: string;
  quote: string;
  paragraph?: number;
  label?: string | undefined;
  citation?: CitationRange | undefined;
}
interface ChatEntry {
  id: string;
  question: string;
  contextQuestion?: string;
  claims: VerifiedClaim[];
  error?: string | undefined;
  message?: string | undefined;
  mode?: string | undefined;
  heading?: string | undefined;
  reviewedClaims?: number | undefined;
  rejectedClaims?: number | undefined;
  answerRelevant?: boolean | undefined;
  fallbackReason?: string | undefined;
  coveredAspects?: number;
  requiredAspects?: number;
  /** Set once the answer is saved; needed to rate it. */
  turnId?: string | undefined;
  rating?: AnswerRating | undefined;
  /** Explain mode: the streamed Markdown, its resolved markers and per-sentence verdicts. */
  text?: string | undefined;
  citations?: ExplainCitation[] | undefined;
  sentences?: SentenceStatus[] | undefined;
  incomplete?: boolean | undefined;
}
interface RequestCallbacks {
  onStart: (controller: AbortController) => void;
  onFinish: (controller: AbortController) => void;
}

export function ChatWorkspace({
  papers,
  activePaperId,
  onSelectPaper,
  onClosePaper,
  onCitation,
  onProgress,
  modes,
  onModeChange,
}: ChatWorkspaceProps) {
  const activeRequest = useRef<AbortController | null>(null);
  const previousActiveId = useRef(activePaperId);
  useEffect(() => {
    if (previousActiveId.current !== activePaperId) {
      activeRequest.current?.abort();
      activeRequest.current = null;
      previousActiveId.current = activePaperId;
    }
  }, [activePaperId]);
  useEffect(() => () => activeRequest.current?.abort(), []);
  const callbacks: RequestCallbacks = {
    onStart: (controller) => {
      activeRequest.current = controller;
    },
    onFinish: (controller) => {
      if (activeRequest.current === controller) activeRequest.current = null;
    },
  };
  return (
    <section className="research-chat" aria-label="Grounded paper Q&A">
      {papers.length > 0 && (
        <nav
          className="conversation-tabs"
          aria-label="Active paper conversations"
        >
          {papers.map((paper) => (
            <div
              key={paper.id}
              className={`conversation-tab${paper.id === activePaperId ? " active" : ""}`}
            >
              <button
                title={paper.title}
                aria-current={paper.id === activePaperId ? "page" : undefined}
                onClick={() => onSelectPaper(paper.id)}
              >
                <FileText size={12} />
                <span>{paper.title}</span>
              </button>
              <button
                className="conversation-tab-close"
                aria-label={`Close ${paper.title}`}
                title="Close conversation"
                onClick={() => onClosePaper(paper.id)}
              >
                <X size={11} />
              </button>
            </div>
          ))}
        </nav>
      )}
      {!papers.length ? (
        <div className="welcome-empty">
          <div className="welcome-symbol">
            <Sparkles size={26} />
          </div>
          <p className="eyebrow">YOUR RESEARCH COMPANION</p>
          <h1>
            Good research starts
            <br />
            with a closer look.
          </h1>
          <p>
            Bring your papers together. Ask focused questions.
            <br />
            Follow every answer back to the evidence.
          </p>
          <div className="welcome-features">
            <span>
              <FileText size={16} /> Full PDF extraction
            </span>
            <span>
              <ShieldCheck size={16} /> Source-grounded answers
            </span>
            <span>
              <Quote size={16} /> Page-level citations
            </span>
          </div>
          <p className="welcome-hint">
            <RefreshHint /> Import a PDF link or sync your sources to begin.
          </p>
        </div>
      ) : (
        papers.map((paper) => (
          <Activity
            key={`${paper.id}:${paper.revision ?? ""}`}
            mode={paper.id === activePaperId ? "visible" : "hidden"}
          >
            <ChatPanel
              paper={paper}
              callbacks={callbacks}
              onCitation={onCitation}
              onProgress={onProgress}
              mode={modes?.[paper.id] ?? DEFAULT_MODE}
              onModeChange={(mode) => onModeChange?.(paper.id, mode)}
            />
          </Activity>
        ))
      )}
    </section>
  );
}

function RefreshHint() {
  return <ArrowUpRight size={14} />;
}

interface ChatPanelProps {
  paper: PaperSummary;
  callbacks: RequestCallbacks;
  onCitation: (citation: CitationRange) => void;
  onProgress: ChatWorkspaceProps["onProgress"];
  mode: AnswerMode;
  onModeChange: (mode: AnswerMode) => void;
}
function ChatPanel({
  paper,
  callbacks,
  onCitation,
  onProgress,
  mode,
  onModeChange,
}: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");
  const controllerRef = useRef<AbortController | null>(null);
  const logEnd = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    logEnd.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [entries, status]);
  useEffect(() => () => controllerRef.current?.abort(), []);
  // Restore this paper's saved turns, replayed through the same event handling as live answers.
  useEffect(() => {
    const controller = new AbortController();
    getChat(paper.id, controller.signal).then(({ turns }) => {
      const restored = turns.map((turn, index) => turn.events
        .filter(({ data }) => data.paper_id === paper.id)
        .reduce<ChatEntry>((entry, { event, data }) => applyAnswerEvent(entry, event, data),
          { id: `saved-${index}`, question: turn.question, claims: [], turnId: turn.turn_id, rating: turn.rating ?? null }));
      if (restored.length) setEntries((current) => [...restored, ...current]);
    }).catch(() => { /* Without saved history the chat simply starts empty. */ });
    return () => controller.abort();
  }, [paper.id]);
  // The paper's reading guide opens the chat once it is written (built after indexing).
  const [guide, setGuide] = useState<PaperGuideView | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    getGuide(paper.id, controller.signal).then(setGuide)
      .catch(() => { /* Without a guide the chat opens with the generic suggestions. */ });
    return () => controller.abort();
  }, [paper.id, paper.status]);
  const readyGuide = guide?.status === "ready" && guide.paper_id === paper.id ? guide.guide : undefined;
  const ready = paper.status === "INDEXED";
  const [showEvaluation, setShowEvaluation] = useState(false);
  const rate = (entry: ChatEntry, rating: AnswerRating, note = "") => {
    if (!entry.turnId) return;
    const previous = entry.rating ?? null;
    const patch = (value: AnswerRating) => setEntries((current) =>
      current.map((item) => item.id === entry.id ? { ...item, rating: value } : item));
    patch(rating);
    rateAnswer(paper.id, entry.turnId, rating, note).catch(() => patch(previous));
  };

  const submitQuestion = async (suggestion?: string) => {
    const prompt = (suggestion ?? question).trim();
    if (!prompt || isLoading || !ready) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    const id = crypto.randomUUID();
    const patchEntry = (patch: Partial<ChatEntry>) =>
      setEntries((current) =>
        current.map((entry) =>
          entry.id === id ? { ...entry, ...patch } : entry,
        ),
      );
    const report = (stage: string, message: string) => {
      setStatus(message);
      onProgress?.(paper.id, { stage, message });
    };
    setQuestion("");
    setIsLoading(true);
    setEntries((current) => [...current, { id, question: prompt, claims: [] }]);
    callbacks.onStart(controller);
    const explain = mode === "explain";
    report("retrieval", explain ? "Reading the whole paper…" : "Finding relevant passages…");
    try {
      // Explain mode reads its history from the server's saved turns, so it sends none.
      const response = await fetch(
        apiUrl(`/v1/papers/${encodeURIComponent(paper.id)}/questions`),
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question: prompt, mode, ...(!explain && entries.length ? {
            history: entries.filter((entry) => !entry.error).slice(-3).map((entry) => ({
              question: (entry.contextQuestion ?? entry.question).slice(0, 700),
              source_ids: [...new Set([
                ...entry.claims.flatMap((claim) => claim.citation?.chunk_id ? [claim.citation.chunk_id] : []),
                ...(entry.citations ?? []).map((citation) => citation.chunkId),
              ])].slice(0, 8),
            })),
          } : {}) }),
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as {
          detail?: string;
        };
        throw new Error(body.detail ?? `Question failed (${response.status}).`);
      }
      if (!response.body)
        throw new Error("The server did not return a response stream.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      const consume = (complete: boolean) => {
        buffer = buffer.replace(/\r\n/g, "\n");
        const blocks = buffer.split("\n\n");
        buffer = complete ? "" : (blocks.pop() ?? "");
        for (const block of blocks) {
          const lines = block.split("\n");
          const kind = lines
            .find((line) => line.startsWith("event:"))
            ?.slice(6)
            .trim();
          const data = lines
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (!data) continue;
          const parsed = JSON.parse(data) as Record<string, unknown>;
          if (parsed.paper_id !== paper.id) continue;
          if (kind === "progress" && typeof parsed.status === "string")
            report(parsed.status, String(parsed.message ?? parsed.status));
          setEntries((current) =>
            current.map((entry) => entry.id === id ? applyAnswerEvent(entry, kind, parsed) : entry),
          );
          if (kind === "error") {
            finished = true;
            throw new Error(String(parsed.message ?? "Answer failed."));
          }
          if (kind === "done") {
            finished = true;
            report("complete", explain ? "Citations checked." : "Source quotes and page citations checked.");
          }
        }
      };
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          consume(false);
        }
        buffer += decoder.decode();
        consume(true);
        if (!finished)
          throw new Error("The response was interrupted. Please try again.");
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      const message = controller.signal.aborted
        ? "Response stopped."
        : error instanceof Error
          ? error.message
          : "Question failed.";
      patchEntry({ error: message });
      report("error", message);
    } finally {
      setIsLoading(false);
      callbacks.onFinish(controller);
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  };
  return (
    <div className="paper-conversation">
      <div className="conversation-scroll">
        <header className="selected-research">
          <div className="paper-label">
            <span>{paper.source}</span>
            <span>·</span>
            <span>
              {paper.page_count
                ? `${paper.page_count} pages extracted`
                : ready
                  ? "PDF indexed"
                  : "Awaiting extraction"}
            </span>
          </div>
          <h1>{paper.title}</h1>
          <details className="paper-abstract">
            <summary>About this paper</summary>
            <p>{paper.abstract}</p>
          </details>
          <button
            className="text-button evaluation-toggle"
            aria-expanded={showEvaluation}
            onClick={() => setShowEvaluation((open) => !open)}
          >
            {showEvaluation ? "Hide answer evaluation" : "Evaluate answers on this paper"}
          </button>
          {showEvaluation && <EvaluationPanel paperId={paper.id} ready={ready} />}
        </header>
        <div
          className="conversation-log"
          aria-live="polite"
          aria-busy={isLoading}
        >
          {readyGuide && (
            <PaperGuideMessage
              guide={readyGuide}
              citations={normalizeCitations(guide?.citations)}
              disabled={!ready || isLoading}
              onAsk={(prompt) => void submitQuestion(prompt)}
              onCitation={onCitation}
            />
          )}
          {!entries.length && !readyGuide ? (
            <div className="conversation-start">
              <div className="assistant-avatar">
                <Sparkles size={18} />
              </div>
              <h2>What would you like to understand?</h2>
              <p>
                Explore this paper with answers you can trace to the source.
              </p>
              <div className="question-suggestions">
                {[
                  "What is this paper about?",
                  "How does the method work?",
                  "What are the key results?",
                  "What limitations are discussed?",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => void submitQuestion(prompt)}
                    disabled={!ready}
                  >
                    {prompt}
                    <ArrowUpRight size={14} />
                  </button>
                ))}
              </div>
              {guide?.status === "pending" && (
                <p className="scope-note">The reading guide for this paper is being written.</p>
              )}
              <span className="scope-note">
                <ShieldCheck size={12} /> This conversation stays within one
                paper.
              </span>
            </div>
          ) : (
            entries.map((entry) => (
              <article className="conversation-turn" key={entry.id}>
                <div className="user-turn">
                  <span>You</span>
                  <p>{entry.question}</p>
                </div>
                <div className="assistant-turn">
                  <div className="answer-heading">
                    <span className="assistant-avatar">
                      <Sparkles size={13} />
                    </span>
                    <strong>ARIE</strong>
                    <span>{entry.text === undefined ? "Grounded in this paper" : "Explained from this paper"}</span>
                  </div>
                  {entry.heading && (
                    <h3 className="answer-title">{entry.heading}</h3>
                  )}
                  {entry.claims.length > 0 && <GroundedAnswer entry={entry} onCitation={onCitation} />}
                  {entry.text !== undefined && <ExplainedAnswer entry={entry} onCitation={onCitation} />}
                  {entry.message && (
                    <p className="answer-notice">{entry.message}</p>
                  )}
                  {entry.error && (
                    <p className="answer-error" role="alert">
                      {entry.error}
                    </p>
                  )}
                  {["source_excerpts", "extractive_summary"].includes(
                    entry.mode ?? "",
                  ) &&
                    entry.claims.length > 0 && (
                      <p className="excerpt-mode">
                        {["provider_http_401", "provider_http_403"].includes(entry.fallbackReason ?? "")
                          ? "Model connection rejected. This extractive answer uses saved paper text. Reconnect in Settings for a model-reviewed summary."
                          : entry.fallbackReason === "entailment_rejected"
                            ? "The drafted summary was not supported by its quotations, so exact source excerpts are shown."
                          : ["entailment_unavailable", "entailment_timeout"].includes(entry.fallbackReason ?? "")
                            ? "The claim checker is loading or unavailable, so exact source excerpts are shown. See Storage & health."
                          : entry.mode === "extractive_summary"
                            ? "Extractive summary from this paper. A reviewed model summary was unavailable."
                            : "Showing source excerpts. A reviewed model answer was unavailable."}
                      </p>
                    )}
                  {entry.turnId && !entry.error && <AnswerFeedback entry={entry} onRate={rate} />}
                </div>
              </article>
            ))
          )}
          {isLoading && (
            <div className="answer-working" role="status">
              <LoaderCircle className="spin" size={15} />
              <span>{status}</span>
            </div>
          )}
          <div ref={logEnd} />
        </div>
      </div>
      <div className="composer-area">
        <form
          className="research-composer"
          onSubmit={(event) => {
            event.preventDefault();
            void submitQuestion();
          }}
        >
          <textarea
            aria-label={`Ask about ${paper.title}`}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                void submitQuestion();
              }
            }}
            placeholder={
              ready
                ? "Ask anything about this paper…"
                : "PDF extraction is still in progress…"
            }
            maxLength={2000}
            disabled={!ready}
            value={question}
            rows={2}
          />
          <div className="composer-bottom">
            <div className="composer-tools">
              <div className="answer-mode" role="group" aria-label="Answer mode">
                {(["explain", "strict"] as const).map((option) => (
                  <button
                    key={option}
                    type="button"
                    aria-pressed={mode === option}
                    disabled={isLoading}
                    title={option === "explain"
                      ? "A readable explanation; every sentence cited or marked as background"
                      : "Only verified statements, each with its exact quote"}
                    onClick={() => onModeChange(option)}
                  >
                    {option === "explain" ? "Explain" : "Strict"}
                  </button>
                ))}
              </div>
              <span>
                <BookOpen size={12} /> One paper. Every citation.
              </span>
            </div>
            {isLoading ? (
              <button
                className="composer-send"
                aria-label="Stop response"
                type="button"
                onClick={() => controllerRef.current?.abort()}
              >
                <Square size={13} />
              </button>
            ) : (
              <button
                className="composer-send"
                aria-label="Send question"
                disabled={!question.trim() || !ready}
                type="submit"
              >
                <Send size={15} />
              </button>
            )}
          </div>
        </form>
        <p className="composer-hint">
          Answers cite saved paper text. When evidence is insufficient,
          ARIE will say so.<span>↵ send</span>
        </p>
      </div>
    </div>
  );
}

/** Assemble only verified sentences; keep original evidence separate from the answer. */
const THEMES = new Set(["Contribution", "Method", "Evaluation", "Findings", "Limitations"]);

function GroundedAnswer({ entry, onCitation }: { entry: ChatEntry; onCitation: (citation: CitationRange) => void }) {
  const cite = (claim: VerifiedClaim) => {
    if (claim.citation) onCitation({ ...claim.citation, quote: claim.quote });
  };
  const reviewed = entry.answerRelevant && Boolean(entry.reviewedClaims)
    && ["grounded_summary", "reviewed_answer"].includes(entry.mode ?? "");
  const paragraphs = [...new Set(entry.claims.map((claim) => claim.paragraph ?? 0))].sort((a, b) => a - b);
  // A multi-paragraph answer leads each paragraph with its theme, e.g. "Method." or "Findings."
  const theme = (paragraph: number) => {
    const label = entry.claims.find((claim) => (claim.paragraph ?? 0) === paragraph)?.label;
    return paragraphs.length > 1 && label && THEMES.has(label) ? label : undefined;
  };
  return <div className="grounded-answer">
    {paragraphs.map((paragraph) => <p className="answer-summary" key={paragraph}>{theme(paragraph) && <strong className="answer-theme">{theme(paragraph)}. </strong>}{entry.claims.map((claim, index) => (claim.paragraph ?? 0) === paragraph && <Fragment key={index}>
      {index > 0 ? " " : ""}{displaySourceText(claim.text)}
      <button className="inline-citation" aria-label={`Source ${index + 1}, page ${claim.citation?.page_start}`} onClick={() => cite(claim)}>[{index + 1}]</button>
    </Fragment>)}</p>)}
    <section className="answer-evidence" aria-label="Verified sources">
      <div className="answer-evidence-heading"><Check size={12} /><strong>Verified sources</strong><span>{entry.claims.length} supporting passage{entry.claims.length === 1 ? "" : "s"}</span></div>
      <details className="supporting-passages"><summary>Supporting passages</summary>
        <ol>{entry.claims.map((claim, index) => <li key={index}>
          <button className="source-citation" onClick={() => cite(claim)}><span>{index + 1}</span><FileText size={12} /> Page {claim.citation?.page_start}<ArrowUpRight size={11} /></button>
          {isTableQuote(claim.quote)
            ? <blockquote className="table-quote">{claim.quote.trim()}</blockquote>
            : <blockquote>{displaySourceText(claim.quote)}</blockquote>}
        </li>)}</ol>
      </details>
      {entry.mode === "entailed_answer" && <details className="answer-evaluation"><summary>Model evaluation</summary>
        <p>{entry.claims.length} statement{entry.claims.length === 1 ? "" : "s"} checked · each strongly entailed by its quote</p>
        <p>Direct answer: the separate model review was not needed</p>
      </details>}
      {reviewed && <details className="answer-evaluation"><summary>Model evaluation</summary>
        <p>{entry.reviewedClaims} statements checked · {entry.claims.length} accepted · {entry.rejectedClaims ?? 0} rejected</p>
        <p>Question relevance: passed · Evidence support: passed</p>
        {Boolean(entry.requiredAspects) && <p>Requested points covered: {entry.coveredAspects ?? 0}/{entry.requiredAspects}</p>}
      </details>}
    </section>
  </div>;
}

/** An explain-mode answer: streamed Markdown whose page chips open the cited chunk. */
function ExplainedAnswer({ entry, onCitation }: { entry: ChatEntry; onCitation: (citation: CitationRange) => void }) {
  const html = renderExplanation(entry.text ?? "", entry.sentences ?? [], entry.citations);
  // The chips are inside rendered HTML, so one handler on the container opens them.
  const open = (event: MouseEvent<HTMLDivElement>) => {
    const chip = event.target instanceof Element ? event.target.closest("button[data-marker]") : null;
    const cited = entry.citations?.find((citation) => String(citation.marker) === chip?.getAttribute("data-marker"));
    if (cited) onCitation({
      page_start: cited.page,
      char_start: cited.charStart,
      char_end: cited.charEnd,
      chunk_id: cited.chunkId,
    });
  };
  return <div className="explained-answer">
    <div className="explanation-text" onClick={open} dangerouslySetInnerHTML={{ __html: html }} />
    {entry.incomplete && <p className="answer-notice">This explanation stopped before it finished.</p>}
  </div>;
}

/** Thumbs up or down on a saved answer; a down rating can carry a short note. */
function AnswerFeedback({ entry, onRate }: { entry: ChatEntry; onRate: (entry: ChatEntry, rating: AnswerRating, note?: string) => void }) {
  const [note, setNote] = useState("");
  const [noting, setNoting] = useState(false);
  const toggle = (rating: "up" | "down") => {
    const next = entry.rating === rating ? null : rating;
    setNoting(next === "down");
    onRate(entry, next);
  };
  return <div className="answer-feedback">
    <span>Was this answer helpful?</span>
    <button className="icon-button" aria-label="Helpful" aria-pressed={entry.rating === "up"} onClick={() => toggle("up")}><ThumbsUp size={13} /></button>
    <button className="icon-button" aria-label="Not helpful" aria-pressed={entry.rating === "down"} onClick={() => toggle("down")}><ThumbsDown size={13} /></button>
    {noting && entry.rating === "down" && <form onSubmit={(event) => { event.preventDefault(); onRate(entry, "down", note.trim()); setNoting(false); }}>
      <input aria-label="What was wrong?" placeholder="What was wrong? (optional)" maxLength={500} value={note} onChange={(event) => setNote(event.target.value)} />
      <button className="text-button" type="submit">Save</button>
    </form>}
  </div>;
}

/** Fold one streamed answer event into its chat entry; used for live and restored turns. */
function applyAnswerEvent(entry: ChatEntry, kind: string | undefined, parsed: Record<string, unknown>): ChatEntry {
  const evaluation = typeof parsed.evaluation === "object" && parsed.evaluation !== null
    ? parsed.evaluation as Record<string, unknown> : {};
  if (kind === "claim" && parsed.verified === true) {
    const claim = normalizeClaim(parsed.claim);
    return claim ? { ...entry, claims: [...entry.claims, claim] } : entry;
  }
  if (kind === "delta" && typeof parsed.text === "string")
    return { ...entry, text: (entry.text ?? "") + parsed.text };
  if (kind === "citations") return { ...entry, citations: normalizeCitations(parsed.citations) };
  if (kind === "answer") return {
    ...entry,
    incomplete: parsed.mode === "explained_answer" && evaluation.complete === false,
    mode: String(parsed.mode ?? ""),
    heading: String(parsed.heading ?? "Answer"),
    contextQuestion: typeof parsed.context_question === "string" ? parsed.context_question.slice(0, 700) : entry.question,
    reviewedClaims: "reviewed_claims" in evaluation ? Number(evaluation.reviewed_claims) : 0,
    rejectedClaims: "rejected_claims" in evaluation ? Number(evaluation.rejected_claims) : 0,
    answerRelevant: evaluation.answer_relevant === true,
    fallbackReason: "fallback_reason" in evaluation ? String(evaluation.fallback_reason) : "",
    coveredAspects: typeof evaluation.covered_aspects === "number" ? evaluation.covered_aspects : 0,
    requiredAspects: Array.isArray(evaluation.required_aspects) ? evaluation.required_aspects.length : 0,
  };
  if (kind === "abstention") return { ...entry, message: String(parsed.message) };
  if (kind === "done") return {
    ...entry,
    mode: String(parsed.mode ?? ""),
    turnId: typeof parsed.turn_id === "string" ? parsed.turn_id : entry.turnId,
  };
  // `checks` today; any later checker that sends sentence verdicts merges the same way.
  if (Array.isArray(parsed.sentences))
    return { ...entry, sentences: mergeSentences(entry.sentences ?? [], normalizeSentences(parsed.sentences)) };
  return entry;
}

function normalizeClaim(value: unknown): VerifiedClaim | null {
  if (
    typeof value !== "object" ||
    value === null ||
    !("text" in value) ||
    !("quote" in value)
  )
    return null;
  if (typeof value.text !== "string" || typeof value.quote !== "string")
    return null;
  if (!displaySourceText(value.text) || displaySourceText(value.quote).length < 15) return null;
  const citation =
    "citation" in value ? normalizeCitation(value.citation) : undefined;
  if (!citation) return null;
  const label = "label" in value && typeof value.label === "string" ? value.label : undefined;
  const paragraph = "paragraph" in value && typeof value.paragraph === "number"
    && Number.isInteger(value.paragraph) && value.paragraph >= 0 && value.paragraph <= 2 ? value.paragraph : 0;
  return { text: value.text, quote: value.quote, label, citation, paragraph };
}
function normalizeCitation(value: unknown): CitationRange | undefined {
  if (
    typeof value !== "object" ||
    value === null ||
    !("page_start" in value) ||
    !("char_start" in value) ||
    !("char_end" in value)
  )
    return undefined;
  const { page_start, char_start, char_end } = value;
  if (
    typeof page_start !== "number" ||
    typeof char_start !== "number" ||
    typeof char_end !== "number" ||
    !Number.isInteger(page_start) ||
    page_start < 1 ||
    !Number.isInteger(char_start) ||
    char_start < 0 ||
    !Number.isInteger(char_end) ||
    char_end <= char_start
  )
    return undefined;
  const chunkId = "chunk_id" in value && typeof value.chunk_id === "string" && value.chunk_id.length <= 256 ? value.chunk_id : undefined;
  return { page_start, char_start, char_end, ...(chunkId ? { chunk_id: chunkId } : {}) };
}

/** A quoted Markdown table keeps its rows on separate lines. */
function isTableQuote(text: string): boolean {
  return /^\s*\|.*\|\s*$/m.test(text) && text.includes("\n");
}

function displaySourceText(text: string): string {
  return text
    .replace(/<\/?(?:sup|sub|b|i|em|strong|u|s|del)>/gi, "")
    .replace(/\*\*|~~/g, "")
    .replace(/(?<!\w)_([^_\n]+)_(?!\w)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}
