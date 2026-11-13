import type { paths } from "./generated";

export type ApiPath = keyof paths;

export interface PaperSummary {
  id: string;
  title: string;
  source: string;
  status: string;
  published_at: string;
  abstract: string;
  relevance_gate: boolean;
  overridden_by_user: number;
  affiliation?: string | undefined;
  revision?: string | undefined;
  page_count?: number | undefined;
  status_reason?: string | undefined;
  custom_metadata?: Record<string, string> | undefined;
  tags?: TopicTag[];
  chunk_count?: number;
  pdf_deleted?: boolean;
  match_reason?: string;
}

export interface TopicTag {
  tag: string;
  label: string;
  basis: string;
  quote?: string;
  page?: number | null;
}

export const DOMAIN_LABELS: Record<string, string> = {
  ai: "Artificial intelligence", llm: "LLMs", agents: "Agents", nlp: "Language / NLP",
  vision: "Computer vision", robotics: "Robotics", ml: "Machine learning",
  retrieval: "Retrieval / RAG", safety: "Safety & alignment", other: "Other research",
};

export interface CitationRange {
  page_start: number;
  char_start: number;
  char_end: number;
  label?: string | undefined;
  quote?: string | undefined;
  chunk_id?: string | undefined;
  /** Figures the cited chunk captions or names (S24); the drawer shows their crops. */
  figures?: string[] | undefined;
  /** OCR text read from a bitmap figure (S25): a transcription, shown next to its crop. */
  ocr?: boolean | undefined;
}

/** A figure as the paper response lists it; the crop is the paper's own figure, never a description. */
export interface PaperFigure {
  number: string;
  page: number;
  caption: string;
  has_crop: boolean;
  crop_url: string | null;
}

export interface Paper extends PaperSummary {
  outline: string[];
  pdf_url?: string | undefined;
  figures?: PaperFigure[] | undefined;
}

/** Figure numbers from a citation: short strings only, at most eight. */
export function figureNumbers(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((n): n is string => typeof n === "string" && n.length > 0 && n.length <= 16).slice(0, 8);
}

export interface PaperPage {
  items: PaperSummary[];
  next_cursor: string | null;
  has_more: boolean;
}

export interface SkippedPaper {
  paper_id: string;
  title: string;
  source: string;
  status: string;
  reason: string;
  overridden_by_user: number;
}

/** A selectable model provider or embedder; only model providers report connection status. */
export interface ProviderOption {
  name: string;
  label: string;
  model: string;
  key_env: string | null;
  key_configured: boolean;
  /** The provider ARIE ships with; others need the customer's own key. */
  default?: boolean;
  status?: "missing_key" | "not_tested" | "verified" | "error";
  message?: string;
}

export interface PluginSlot {
  slot: string;
  configured: Record<string, unknown>;
  available: string[];
  providers?: ProviderOption[];
  /** Embedder slot only: the embedder in use and why automatic mode chose it. */
  effective?: { embedder: string; reason: string };
}

export interface PluginResponse {
  plugins: PluginSlot[];
}

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

const ROUTES: {
  papers: ApiPath;
  paper: ApiPath;
  questions: ApiPath;
  skips: ApiPath;
  override: ApiPath;
  sync: ApiPath;
  plugins: ApiPath;
  plugin: ApiPath;
  testPlugin: ApiPath;
} = {
  papers: "/v1/papers",
  paper: "/v1/papers/{paper_id}",
  questions: "/v1/papers/{paper_id}/questions",
  skips: "/v1/skips",
  override: "/v1/papers/{paper_id}/override",
  sync: "/v1/sync",
  plugins: "/v1/plugins",
  plugin: "/v1/plugins/{slot}",
  testPlugin: "/v1/plugins/{slot}/test",
};

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const response = await fetch(apiUrl(path), { ...init, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        detail = String(body.detail);
      }
    } catch {
      detail = `${response.status} ${response.statusText}`;
    }
    if (response.status === 404 && path === "/v1/imports" && init.method === "POST") {
      detail = "The running API has no import route. Restart the ARIE API to load the latest code.";
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function getPapers(
  cursor: string | null,
  signal?: AbortSignal,
  filters: Record<string, string> = {},
): Promise<PaperPage> {
  const params = new URLSearchParams();
  if (cursor) params.set("cursor", cursor);
  for (const [key, value] of Object.entries(filters)) if (value) params.set(key, value);
  const query = params.size > 0 ? `?${params.toString()}` : "";
  return requestJson<PaperPage>(
    `${ROUTES.papers}${query}`,
    signal ? { signal } : {},
  );
}

export async function getPaper(paperId: string): Promise<Paper> {
  return requestJson<Paper>(`/v1/papers/${encodeURIComponent(paperId)}`);
}

/** One saved Q&A turn: the question and the events the server streamed for its answer. */
export interface SavedChatTurn {
  turn_id?: string;
  question: string;
  events: { event: string; data: Record<string, unknown> }[];
  rating?: AnswerRating;
}

export type AnswerRating = "up" | "down" | null;

/** Why a reader rated an answer down, as the API names them, with the label the chips show. */
export const FEEDBACK_REASONS = [
  ["wrong", "Wrong"],
  ["not_in_paper", "Not in the paper"],
  ["too_shallow", "Too shallow"],
  ["too_long", "Too long"],
  ["missing_citation", "Missing citation"],
  ["did_not_answer", "Did not answer"],
] as const;

export type FeedbackReason = (typeof FEEDBACK_REASONS)[number][0];

/** What a reader adds to a down rating: the reasons that apply and a comment. */
export interface RatingDetail {
  note?: string;
  reasons?: readonly FeedbackReason[];
}

/** Rate a saved answer; `null` clears it. Answers rated down join the paper's evaluation. */
export function rateAnswer(
  paperId: string,
  turnId: string,
  rating: AnswerRating,
  { note = "", reasons = [] }: RatingDetail = {},
): Promise<{ turn_id: string; rating: AnswerRating }> {
  return requestJson(
    `/v1/papers/${encodeURIComponent(paperId)}/chat/${encodeURIComponent(turnId)}/feedback`,
    { method: "PUT", body: JSON.stringify({ rating, note, ...(reasons.length ? { reasons } : {}) }) },
  );
}

/** Make an answer rated down a scored case: 1-3 key facts and, optionally, the paper's own sentence that holds them. */
export function addFeedbackFacts(
  paperId: string,
  turnId: string,
  keyFacts: readonly string[],
  sourceSentence = "",
): Promise<{ turn_id: string; key_facts: string[]; source_sentence: string }> {
  return requestJson(
    `/v1/papers/${encodeURIComponent(paperId)}/chat/${encodeURIComponent(turnId)}/feedback/facts`,
    { method: "PUT", body: JSON.stringify({ key_facts: keyFacts, source_sentence: sourceSentence }) },
  );
}

/** A question written from one exact sentence of the paper (the ground truth). */
export interface EvalCase {
  case_id: string;
  question: string;
  kind: "fact" | "abstain" | "feedback";
  origin: string;
  answer_quote: string;
  page_start: number;
  key_facts: string[];
  note: string;
}

export interface EvalResult {
  variant: "configured" | "no_reranker";
  case_id: string;
  question: string;
  kind: EvalCase["kind"];
  /** null for feedback cases, which have no reference answer. */
  passed: boolean | null;
  grounded: boolean;
  evidence_hit: boolean;
  fact_recall: number;
  abstained: boolean;
  mode: string;
  claims: number;
  seconds: number;
  reranked: boolean;
  answer: string;
  cited_pages: number[];
  error: string;
  /** Why the answer fell back to excerpts or abstained; empty otherwise. Absent in runs saved before the trace existed. */
  fallback_reason?: string;
  /** Seconds per stage (`plan`, `vector`, `read`, `rerank`, `write`, `tier3`, `review`, `total`). */
  trace?: { seconds: Record<string, number>; prompt_version: string; model: string };
  /** Explain runs only: the share of cited sentences the checks and the judge accept; null if not judged. */
  citation_support?: number | null;
  /** Explain runs only: 1, 0.5 or 0 for the explanation against the strict answer; null if not compared. */
  win_rate?: number | null;
  /** The support judge's status: `complete`, `failed`, `skipped`; empty in a strict run. */
  judge_status?: string;
}

export interface EvalSummary {
  cases: number;
  scored: number;
  passed: number;
  pass_rate: number | null;
  grounded_rate: number | null;
  evidence_hit_rate: number | null;
  mean_fact_recall: number | null;
  answered_rate: number | null;
  abstain_correct_rate: number | null;
  errors: number;
  median_seconds: number | null;
  review: number;
  /** Explain runs only; absent or null in a strict run. */
  citation_support?: number | null;
  win_rate?: number | null;
  /** Cases behind `win_rate`, and support judgements that failed (left out of `citation_support`). */
  win_cases?: number;
  judge_failed?: number;
}

export interface EvalRun {
  run_id: string;
  paper_id: string;
  status: "queued" | "running" | "complete" | "failed" | "interrupted";
  message: string;
  created_at: string;
  finished_at: string | null;
  settings: Record<string, string>;
  compare_reranker: boolean;
  progress: { done: number; total: number };
  summary: Partial<Record<EvalResult["variant"], EvalSummary>>;
  results: EvalResult[];
}

/** An answer the reader rated, with what the answer recorded about itself; the new fields are absent on ratings saved before S15. */
export interface FeedbackItem {
  turn_id: string;
  question: string;
  rating: string;
  note: string;
  created_at: string;
  reasons?: FeedbackReason[];
  mode?: string;
  task?: string;
  prompt_version?: string;
  trace_id?: string;
  key_facts?: string[];
  source_sentence?: string;
}

/** A follow-up that read as a negative rating of the answer before it; never shown in chat. */
export interface ImplicitFeedback {
  turn_id: string;
  question: string;
  signal: "said_wrong" | "asked_again" | "repeated_question";
  follow_up: string;
  created_at: string;
}

/** How often the judge's verdict matched the reader's rating; `rate` is `agreed / judged`. */
export interface FeedbackAgreement {
  rated: number;
  judged: number;
  agreed: number;
  rate: number | null;
  /** True once 30 answers are rated (gate 4). */
  ready: boolean;
  /** Unrated answers a follow-up questioned; not part of the agreement. */
  implicit: number;
  by_task: { task: string; rated: number; judged: number; agreed: number; rate: number | null }[];
}

export interface PaperEvaluation {
  paper_id: string;
  cases: EvalCase[];
  runs: Omit<EvalRun, "results">[];
  latest: EvalRun | null;
  feedback: FeedbackItem[];
  implicit_feedback?: ImplicitFeedback[];
  /** Library-wide, not this paper's: gate 4 counts every rated answer. */
  agreement?: FeedbackAgreement;
}

export function getPaperEvaluation(paperId: string): Promise<PaperEvaluation> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/eval`);
}

export function startPaperEvaluation(
  paperId: string,
  options: { regenerate?: boolean; compare_reranker?: boolean } = {},
): Promise<EvalRun> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/eval/runs`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

export function getEvalRun(paperId: string, runId: string): Promise<EvalRun> {
  return requestJson(
    `/v1/papers/${encodeURIComponent(paperId)}/eval/runs/${encodeURIComponent(runId)}`,
  );
}

/** The last saved turns for a paper, oldest first (the server keeps 10 per paper). */
export async function getChat(paperId: string, signal?: AbortSignal): Promise<{ turns: SavedChatTurn[] }> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/chat`, signal ? { signal } : {});
}

/** Delete the saved chat of every paper; returns how many turns were removed. */
export function clearAllChats(): Promise<{ deleted: number }> {
  return requestJson("/v1/chat", { method: "DELETE" });
}

/** A term or an equation's symbol with its cited meaning. */
export interface GuideTerm {
  term: string;
  meaning: string;
}

/** The paper's reading guide (S14). Every text but questions, terms and equations has markers. */
export interface PaperGuide {
  tldr: string;
  problem: string;
  method_steps: string[];
  key_equations: { equation: string; meaning: string; symbols: GuideTerm[] }[];
  results: string[];
  limitations: string[];
  glossary: GuideTerm[];
  section_summaries: { section: string; summary: string }[];
  suggested_questions: string[];
}

/** `ready` carries the guide and one citation per marker; `pending` is still being written. */
export interface PaperGuideView {
  paper_id: string;
  status: "ready" | "pending" | "missing";
  error?: string;
  guide?: PaperGuide;
  citations?: unknown[];
}

export async function getGuide(paperId: string, signal?: AbortSignal): Promise<PaperGuideView> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/guide`, signal ? { signal } : {});
}

export async function getSkippedPapers(): Promise<{ items: SkippedPaper[] }> {
  return requestJson(ROUTES.skips);
}

export async function overridePaper(paperId: string): Promise<void> {
  await requestJson(`/v1/papers/${encodeURIComponent(paperId)}/override`, {
    method: "POST",
    body: JSON.stringify({
      size_limit_mb: 25,
      relevance_override: false,
      note: "Download Anyway from the skipped audit drawer",
    }),
  });
}

export interface SyncSettings {
  mode: "relative" | "year" | "month" | "since" | "range";
  initial_months: number;
  year?: number | null;
  month?: number | null;
  start_month?: string | null;
  end_month?: string | null;
  batch_size: number;
  max_papers: number;
}

export const DEFAULT_SYNC: SyncSettings = {
  mode: "relative", initial_months: 6, batch_size: 20, max_papers: 100,
};

export interface SyncStatus {
  job: SyncJob | null;
  initial_months: number;
  settings?: SyncSettings;
  sources?: string[];
}

export async function syncSource(
  source = "all", settings?: SyncSettings, resume_job_id?: string,
): Promise<SyncJob> {
  return requestJson(ROUTES.sync, {
    method: "POST",
    body: JSON.stringify({ source, settings, resume_job_id }),
  });
}

export async function getPluginSlots(): Promise<PluginResponse> {
  return requestJson(ROUTES.plugins);
}

export async function testPlugin(
  slot: string,
  plugin: string,
  config: Record<string, unknown> = {},
): Promise<{ ok: boolean }> {
  return requestJson(`/v1/plugins/${encodeURIComponent(slot)}/test`, {
    method: "POST",
    body: JSON.stringify({ plugin, config }),
  });
}

export async function setPlugin(
  slot: string,
  plugin: string,
  config: Record<string, unknown> = {},
): Promise<{ slot: string; plugin: string }> {
  return requestJson(`/v1/plugins/${encodeURIComponent(slot)}`, {
    method: "PUT",
    body: JSON.stringify({ plugin, config }),
  });
}

export interface SyncJob {
  job_id: string;
  status:
    | "queued"
    | "running"
    | "completed"
    | "completed_with_errors"
    | "failed"
    | "interrupted";
  source: string;
  kind?: "import" | "sync";
  paper_id?: string | null;
  sources: string[];
  since: string;
  until: string;
  phase: string;
  message: string;
  pause_requested?: boolean;
  limit_reached?: boolean;
  settings?: SyncSettings;
  current_paper: string | null;
  discovered: number;
  processed: number;
  indexed: number;
  skipped: number;
  duplicates: number;
  failed: number;
  source_errors: string[];
  error?: string;
}

export function getSyncStatus(): Promise<SyncStatus> {
  return requestJson("/v1/sync");
}

export function getExtractedPage(
  paperId: string,
  page: number,
): Promise<{ number: number; text: string }> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/pages/${page}`);
}

export function pauseSync(jobId: string): Promise<SyncJob> {
  return requestJson(`/v1/sync/${encodeURIComponent(jobId)}/pause`, {
    method: "POST",
  });
}

export function importArxivPaper(url: string): Promise<SyncJob> {
  return requestJson("/v1/imports", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

export function getImportStatus(jobId: string): Promise<SyncJob> {
  return requestJson(`/v1/imports/${encodeURIComponent(jobId)}`);
}

export function saveSyncSettings(settings: SyncSettings): Promise<{ settings: SyncSettings }> {
  return requestJson("/v1/sync/settings", { method: "PUT", body: JSON.stringify(settings) });
}

/** One answer-path model; `degraded` after repeated timeouts or failures, with the reason in `detail`. */
export interface AnswerModelHealth {
  plugin: string;
  status: string;
  detail: string;
  consecutive_failures?: number;
  last_error?: string;
  last_latency_seconds?: number | null;
}

/** The prompt-injection classifier that screens paper chunks at indexing (ADR 0014). */
export interface GuardHealth extends AnswerModelHealth {
  model?: string;
  threshold?: number;
  papers_scored?: number;
  papers_total?: number;
  untrusted_chunks?: number;
}

export interface StorageHealth {
  checked_at: string;
  database: {
    provider: string; status: string; size_bytes: number | null;
    papers: number | null; evidence_chunks: number | null;
  };
  file_storage: {
    provider: string; status: string; size_bytes: number | null; files: number | null;
    disk_free_bytes: number | null; disk_total_bytes: number | null;
  };
  vector_store: {
    provider: string; status: string; size_bytes: number | null; detail: string;
    embedder?: string; vectors?: number | null; papers_indexed?: number | null; papers_total?: number | null;
  };
  entailment?: AnswerModelHealth;
  reranker?: AnswerModelHealth;
  guard?: GuardHealth;
  active_jobs: SyncJob[];
}

export function getStorageHealth(): Promise<StorageHealth> {
  return requestJson("/v1/storage");
}

export function deletePaper(paperId: string, pdfOnly: boolean): Promise<void> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}?pdf_only=${pdfOnly}`, {
    method: "DELETE",
  });
}
