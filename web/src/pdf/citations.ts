import type { CitationRange } from '../api/client'

/**
 * How the PDF worker opens a stored paper. A PDF is untrusted bytes (ADR 0013): pdf.js may not
 * compile code from it (font programs fall back to the interpreter) or render XFA forms.
 */
export const PDF_DOCUMENT_OPTIONS = { isEvalSupported: false, enableXfa: false } as const

export interface CitationHighlightMessage {
  type: 'highlight'
  pdfUrl: string
  citation: CitationRange
}

export interface CitationWorker {
  postMessage(message: CitationHighlightMessage): void
}

export function dispatchCitationHighlight(
  worker: CitationWorker,
  pdfUrl: string,
  citation: CitationRange,
): void {
  worker.postMessage({
    type: 'highlight',
    pdfUrl,
    citation: {
      page_start: citation.page_start,
      char_start: citation.char_start,
      char_end: citation.char_end,
    },
  })
}
