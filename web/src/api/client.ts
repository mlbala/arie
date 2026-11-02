import type { paths } from './generated'

export type ApiPath = keyof paths

export interface PaperSummary {
  id: string
  title: string
  source: string
  status: string
  published_at: string
  abstract: string
  relevance_gate: boolean
  overridden_by_user: number
  affiliation?: string | undefined
  custom_metadata?: Record<string, string> | undefined
}

export interface CitationRange {
  page_start: number
  char_start: number
  char_end: number
  label?: string | undefined
}

export interface BriefPart {
  label: string
  title: string
  content: string
  citations?: CitationRange[] | undefined
}

export interface Paper extends PaperSummary {
  outline: string[]
  brief: { parts: BriefPart[] }
  verified_5_part_brief: BriefPart[]
  pdf_url?: string | undefined
}

export interface PaperPage {
  items: PaperSummary[]
  next_cursor: string | null
  has_more: boolean
}

export interface SkippedPaper {
  paper_id: string
  title: string
  source: string
  status: string
  reason: string
  overridden_by_user: number
}

export interface PluginSlot {
  slot: string
  configured: Record<string, unknown>
  available: string[]
}

export interface PluginResponse {
  plugins: PluginSlot[]
}

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

const ROUTES: {
  papers: ApiPath
  paper: ApiPath
  questions: ApiPath
  skips: ApiPath
  override: ApiPath
  sync: ApiPath
  plugins: ApiPath
  plugin: ApiPath
  testPlugin: ApiPath
} = {
  papers: '/v1/papers',
  paper: '/v1/papers/{paper_id}',
  questions: '/v1/papers/{paper_id}/questions',
  skips: '/v1/skips',
  override: '/v1/papers/{paper_id}/override',
  sync: '/v1/sync',
  plugins: '/v1/plugins',
  plugin: '/v1/plugins/{slot}',
  testPlugin: '/v1/plugins/{slot}/test',
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !headers.has('content-type')) {
    headers.set('content-type', 'application/json')
  }
  const response = await fetch(apiUrl(path), { ...init, headers })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body: unknown = await response.json()
      if (typeof body === 'object' && body !== null && 'detail' in body) {
        detail = String(body.detail)
      }
    } catch {
      detail = `${response.status} ${response.statusText}`
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export async function getPapers(
  cursor: string | null,
  signal?: AbortSignal,
): Promise<PaperPage> {
  const params = new URLSearchParams()
  if (cursor) params.set('cursor', cursor)
  const query = params.size > 0 ? `?${params.toString()}` : ''
  return requestJson<PaperPage>(`${ROUTES.papers}${query}`, signal ? { signal } : {})
}

export async function getPaper(paperId: string): Promise<Paper> {
  return requestJson<Paper>(`/v1/papers/${encodeURIComponent(paperId)}`)
}

export async function getSkippedPapers(): Promise<{ items: SkippedPaper[] }> {
  return requestJson(ROUTES.skips)
}

export async function overridePaper(paperId: string): Promise<void> {
  await requestJson(`/v1/papers/${encodeURIComponent(paperId)}/override`, {
    method: 'POST',
    body: JSON.stringify({
      size_limit_mb: 200,
      relevance_override: true,
      note: 'Download Anyway from the skipped audit drawer',
    }),
  })
}

export async function syncSource(source = 'arxiv'): Promise<{ job_id: string }> {
  return requestJson(ROUTES.sync, {
    method: 'POST',
    body: JSON.stringify({ source }),
  })
}

export async function getPluginSlots(): Promise<PluginResponse> {
  return requestJson(ROUTES.plugins)
}

export async function testPlugin(
  slot: string,
  plugin: string,
  config: Record<string, unknown> = {},
): Promise<{ ok: boolean }> {
  return requestJson(`/v1/plugins/${encodeURIComponent(slot)}/test`, {
    method: 'POST',
    body: JSON.stringify({ plugin, config }),
  })
}

export async function setPlugin(
  slot: string,
  plugin: string,
  config: Record<string, unknown> = {},
): Promise<{ slot: string; plugin: string }> {
  return requestJson(`/v1/plugins/${encodeURIComponent(slot)}`, {
    method: 'PUT',
    body: JSON.stringify({ plugin, config }),
  })
}
