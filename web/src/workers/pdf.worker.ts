import pdfWorkerUrl from 'pdfjs-dist/legacy/build/pdf.worker.mjs?url'
import type { PDFDocumentProxy, TextItem } from 'pdfjs-dist/types/src/display/api'
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs'
import type { CitationRange } from '../api/client'

interface HighlightRequest {
  type: 'highlight'
  pdfUrl: string
  citation: CitationRange
}

interface HighlightResult {
  type: 'highlight'
  page: number
  before: string
  highlighted: string
  after: string
}

interface WorkerError {
  type: 'error'
  message: string
}

GlobalWorkerOptions.workerSrc = pdfWorkerUrl

self.onmessage = async (event: MessageEvent<HighlightRequest>) => {
  try {
    const document: PDFDocumentProxy = await getDocument({ url: event.data.pdfUrl }).promise
    const pageNumber = event.data.citation.page_start
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > document.numPages) {
      throw new Error('Citation refers to a page outside the PDF.')
    }

    const page = await document.getPage(pageNumber)
    const textContent = await page.getTextContent()
    const text = textContent.items
      .filter((item): item is TextItem => 'str' in item)
      .map((item) => item.str)
      .join('')
    const { char_start: start, char_end: end } = event.data.citation
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > text.length) {
      throw new Error('Citation offsets are not valid for the selected PDF page.')
    }

    const result: HighlightResult = {
      type: 'highlight',
      page: pageNumber,
      before: text.slice(0, start),
      highlighted: text.slice(start, end),
      after: text.slice(end),
    }
    self.postMessage(result)
  } catch (error) {
    const result: WorkerError = {
      type: 'error',
      message: error instanceof Error ? error.message : 'Unable to load the PDF citation.',
    }
    self.postMessage(result)
  }
}
