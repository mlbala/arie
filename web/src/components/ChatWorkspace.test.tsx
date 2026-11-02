import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ChatWorkspace } from './ChatWorkspace'
import type { PaperSummary } from '../api/client'

const papers: PaperSummary[] = [
  {
    id: 'paper-1',
    title: 'Evidence and Verification',
    source: 'arxiv',
    status: 'INDEXED',
    published_at: '2025-01-12T00:00:00Z',
    abstract: 'A test paper.',
    relevance_gate: true,
    overridden_by_user: 0,
  },
  {
    id: 'paper-2',
    title: 'Retrieval Methods',
    source: 'openreview',
    status: 'INDEXED',
    published_at: '2025-01-10T00:00:00Z',
    abstract: 'Another test paper.',
    relevance_gate: true,
    overridden_by_user: 0,
  },
]

describe('ChatWorkspace', () => {
  it('retains the hidden Activity pane state when switching papers', () => {
    const onSelectPaper = vi.fn()
    const props = {
      papers,
      activePaperId: 'paper-1',
      onSelectPaper,
      onCitation: vi.fn(),
    }
    const { rerender } = render(<ChatWorkspace {...props} />)

    fireEvent.change(screen.getByLabelText('Ask about Evidence and Verification'), {
      target: { value: 'Keep this draft' },
    })
    rerender(<ChatWorkspace {...props} activePaperId="paper-2" />)
    rerender(<ChatWorkspace {...props} activePaperId="paper-1" />)

    expect(screen.getByLabelText('Ask about Evidence and Verification')).toHaveValue('Keep this draft')
  })

  it('aborts an in-flight question when switching papers', async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    const props = {
      papers,
      activePaperId: 'paper-1',
      onSelectPaper: vi.fn(),
      onCitation: vi.fn(),
    }
    const { rerender } = render(<ChatWorkspace {...props} />)

    fireEvent.change(screen.getByLabelText('Ask about Evidence and Verification'), {
      target: { value: 'What is the method?' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const signal = fetchMock.mock.calls[0]?.[1]?.signal

    rerender(<ChatWorkspace {...props} activePaperId="paper-2" />)

    await waitFor(() => expect(signal?.aborted).toBe(true))
  })
})
