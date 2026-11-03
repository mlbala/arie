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
}

export interface BriefPart {
  label: string;
  title: string;
  content: string;
  citations?: CitationRange[] | undefined;
}

export interface Paper extends PaperSummary {
  outline: string[];
  brief: { parts: BriefPart[] };
  verified_5_part_brief: BriefPart[];
  pdf_url?: string | undefined;
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
  question: string;
  events: { event: string; data: Record<string, unknown> }[];
}

/** The last saved turns for a paper, oldest first (the server keeps 10 per paper). */
export async function getChat(paperId: string, signal?: AbortSignal): Promise<{ turns: SavedChatTurn[] }> {
  return requestJson(`/v1/papers/${encodeURIComponent(paperId)}/chat`, signal ? { signal } : {});
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
