import { dispatchCitationHighlight } from './citations'
import { describe, expect, it, vi } from 'vitest'

describe('dispatchCitationHighlight', () => {
  it('sends the exact page and character offsets to the PDF worker', () => {
    const worker = { postMessage: vi.fn() }
    const citation = { page_start: 4, char_start: 118, char_end: 164 }

    dispatchCitationHighlight(worker, '/papers/source.pdf', citation)

    expect(worker.postMessage).toHaveBeenCalledWith({
      type: 'highlight',
      pdfUrl: '/papers/source.pdf',
      citation: { page_start: 4, char_start: 118, char_end: 164 },
    })
  })
})
