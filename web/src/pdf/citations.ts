import type { CitationRange } from '../api/client'

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
