import { useQuery } from '@tanstack/react-query'
import { ArrowUpRight, BadgeCheck, BookOpenText, FileText, Quote } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { getPaper, type BriefPart, type CitationRange } from '../api/client'
import { dispatchCitationHighlight } from '../pdf/citations'

interface BriefViewerProps {
  paperId: string | null
  citation: CitationRange | null
}

interface MarkdownWorker {
  postMessage(message: { id: string; content: string }): void
  terminate(): void
  onmessage: ((event: MessageEvent<{ id: string; html: string }>) => void) | null
  onerror: ((event: ErrorEvent) => void) | null
}

interface PdfWorker {
  postMessage(message: {
    type: 'highlight'
    pdfUrl: string
    citation: CitationRange
  }): void
  terminate(): void
  onmessage: ((event: MessageEvent<PdfWorkerResponse>) => void) | null
}

type PdfWorkerResponse =
  | { type: 'highlight'; page: number; before: string; highlighted: string; after: string }
  | { type: 'error'; message: string }

const EMPTY_PARTS: BriefPart[] = []

export function BriefViewer({ paperId, citation }: BriefViewerProps) {
  const paperQuery = useQuery({
    queryKey: ['paper', paperId],
    queryFn: () => getPaper(paperId ?? ''),
    enabled: paperId !== null,
  })
  const paper = paperQuery.data
  const parts = paper
    ? (paper.verified_5_part_brief ?? paper.brief.parts)
    : EMPTY_PARTS
  const [rendered, setRendered] = useState<Record<string, string>>({})
  const [renderError, setRenderError] = useState<string | null>(null)
  const [pdfResult, setPdfResult] = useState<Extract<PdfWorkerResponse, { type: 'highlight' }> | null>(
    null,
  )
  const [pdfError, setPdfError] = useState<string | null>(null)
  const markdownWorkerRef = useRef<MarkdownWorker | null>(null)
  const pdfWorkerRef = useRef<PdfWorker | null>(null)

  useEffect(() => {
    const worker = new Worker(new URL('../workers/markdown.worker.ts', import.meta.url), {
      type: 'module',
    }) as MarkdownWorker
    markdownWorkerRef.current = worker
    worker.onmessage = ({ data }) => {
      setRendered((current) => ({ ...current, [data.id]: data.html }))
    }
    worker.onerror = (event) => {
      setRenderError(event.message || 'Markdown rendering failed.')
    }
    return () => {
      worker.terminate()
      markdownWorkerRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!paper) return
    for (const [index, part] of parts.entries()) {
      markdownWorkerRef.current?.postMessage({
        id: `${paper.id}:${index}`,
        content: part.content,
      })
    }
  }, [paper, parts])

  useEffect(
    () => () => {
      pdfWorkerRef.current?.terminate()
      pdfWorkerRef.current = null
    },
    [paperId],
  )

  const pdfUrl = paper?.pdf_url
  const openCitation = useCallback((citationRange: CitationRange) => {
    if (!pdfUrl) return
    let worker = pdfWorkerRef.current
    if (!worker) {
      worker = new Worker(new URL('../workers/pdf.worker.ts', import.meta.url), {
        type: 'module',
      }) as PdfWorker
      worker.onmessage = ({ data }) => {
        if (data.type === 'highlight') setPdfResult(data)
        else setPdfError(data.message)
      }
      pdfWorkerRef.current = worker
    }
    dispatchCitationHighlight(worker, pdfUrl, citationRange)
  }, [pdfUrl])

  useEffect(() => {
    if (citation) openCitation(citation)
  }, [citation, openCitation])

  if (!paperId) {
    return (
      <section className="pane brief-pane" aria-label="Verified brief and PDF">
        <div className="loading-state">Select a paper to open its verified brief.</div>
      </section>
    )
  }

  if (paperQuery.isPending) {
    return (
      <section className="pane brief-pane" aria-label="Verified brief and PDF">
        <div className="brief-top"><span className="eyebrow">Paper intelligence</span></div>
        <div className="brief-scroll skeleton" aria-label="Loading paper">
          {[100, 82, 91, 67, 87, 72].map((width, index) => (
            <div className="skeleton-line" key={index} style={{ width: `${width}%` }} />
          ))}
        </div>
      </section>
    )
  }

  if (paperQuery.isError || !paper) {
    return (
      <section className="pane brief-pane" aria-label="Verified brief and PDF">
        <div className="error-state">{paperQuery.error?.message ?? 'Paper details are unavailable.'}</div>
      </section>
    )
  }

  return (
    <section className="pane brief-pane" aria-label="Verified brief and PDF">
      <header className="brief-top">
        <div>
          <p className="eyebrow">Paper intelligence</p>
          <div className="pane-title-row">
            <h1 className="pane-title">Verified brief</h1>
          </div>
        </div>
        <div className="brief-top-right">
          <span className="verified-pill"><BadgeCheck size={12} /> 5-part verified</span>
          <button
            aria-label="Open source PDF"
            className="icon-button"
            disabled={!paper.pdf_url}
            onClick={() => {
              const citation = parts.flatMap((part) => part.citations ?? [])[0]
              if (citation) openCitation(citation)
            }}
            title={paper.pdf_url ? 'Open source PDF' : 'PDF URL unavailable'}
            type="button"
          >
            <ArrowUpRight size={14} />
          </button>
        </div>
      </header>
      <div className="brief-scroll">
        <div className="selected-paper-heading">
          <h2>{paper.title}</h2>
          <p className="abstract">{paper.abstract}</p>
        </div>
        <div className="brief-divider" />
        {parts.map((part, index) => (
          <BriefSection
            index={index}
            key={`${paper.id}:${part.label}`}
            onCitation={openCitation}
            part={part}
            html={rendered[`${paper.id}:${index}`]}
            renderError={renderError}
          />
        ))}
        {parts.length === 0 && <p className="empty-state">No verified brief is available.</p>}
        <div className="pdf-card">
          <div className="pdf-card-header">
            <span className="chat-head"><FileText size={12} /> Source excerpt</span>
            <span>{pdfResult ? `Page ${pdfResult.page}` : 'Citation-linked'}</span>
          </div>
          {pdfResult ? (
            <div className="pdf-page">
              {pdfResult.before}<mark>{pdfResult.highlighted}</mark>{pdfResult.after}
            </div>
          ) : (
            <div className="pdf-placeholder">
              <BookOpenText size={15} />
              {pdfError ??
                (citation && !pdfUrl
                  ? 'A PDF URL is not available for this paper.'
                  : 'Click a citation badge to locate its exact source span.')}
            </div>
          )}
        </div>
      </div>
      <footer className="brief-footer">
        <span><BadgeCheck size={11} /> Every claim tied to verified evidence</span>
        <span><Quote size={11} /> Source: {paper.source}</span>
      </footer>
    </section>
  )
}

interface BriefSectionProps {
  index: number
  part: BriefPart
  html: string | undefined
  renderError: string | null
  onCitation: (citation: CitationRange) => void
}

function BriefSection({ index, part, html, renderError, onCitation }: BriefSectionProps) {
  return (
    <section className="brief-part">
      <div className="brief-heading">
        <span className="part-number">{String(index + 1).padStart(2, '0')}</span>
        <h3>{part.title}</h3>
      </div>
      <div className="brief-content" aria-live="polite">
        {html ? (
          <div dangerouslySetInnerHTML={{ __html: html }} />
        ) : renderError ? (
          <p>{part.content}</p>
        ) : (
          <div className="skeleton-line" style={{ width: '94%', minHeight: 8 }} />
        )}
      </div>
      {part.citations && part.citations.length > 0 && (
        <div className="citation-row" aria-label={`Citations for ${part.title}`}>
          {part.citations.map((citation, citationIndex) => (
            <button
              className="citation-button"
              key={`${citation.page_start}:${citation.char_start}:${citationIndex}`}
              onClick={() => onCitation(citation)}
              type="button"
            >
              <Quote size={10} />
              {citation.label ?? `p. ${citation.page_start}`}
            </button>
          ))}
        </div>
      )}
    </section>
  )
}
