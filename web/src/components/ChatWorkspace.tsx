import { Activity, useEffect, useRef, useState } from 'react'
import { CheckCircle2, CircleHelp, LoaderCircle, Send, ShieldCheck } from 'lucide-react'
import {
  apiUrl,
  type CitationRange,
  type PaperSummary,
} from '../api/client'

interface ChatWorkspaceProps {
  papers: PaperSummary[]
  activePaperId: string | null
  onSelectPaper: (paperId: string) => void
  onCitation: (citation: CitationRange) => void
}

interface VerifiedClaim {
  text: string
  quote: string
  citation?: CitationRange | undefined
}

interface ChatEntry {
  id: string
  question: string
  claims: VerifiedClaim[]
  error?: string | undefined
}

interface StreamClaim {
  paper_id?: unknown
  claim?: unknown
}

interface RequestCallbacks {
  onStart: (controller: AbortController) => void
  onFinish: (controller: AbortController) => void
}

export function ChatWorkspace({
  papers,
  activePaperId,
  onSelectPaper,
  onCitation,
}: ChatWorkspaceProps) {
  const recentPapers = papers.slice(0, 5)
  const activeRequest = useRef<AbortController | null>(null)
  const previousActiveId = useRef(activePaperId)

  useEffect(() => {
    if (previousActiveId.current !== activePaperId) {
      activeRequest.current?.abort()
      activeRequest.current = null
      previousActiveId.current = activePaperId
    }
  }, [activePaperId])

  const requestCallbacks: RequestCallbacks = {
    onStart: (controller) => {
      activeRequest.current = controller
    },
    onFinish: (controller) => {
      if (activeRequest.current === controller) activeRequest.current = null
    },
  }

  return (
    <section className="pane chat-pane-shell" aria-label="Grounded paper Q&A">
      <header className="pane-header chat-header">
        <div className="pane-title-row">
          <div>
            <p className="eyebrow">Grounded assistant</p>
            <h1 className="pane-title">Ask this paper</h1>
          </div>
          <span className="verified-pill"><ShieldCheck size={11} /> Verified only</span>
        </div>
        <div className="chat-paper-tabs" aria-label="Recent paper conversations">
          {recentPapers.map((paper) => (
            <button
              aria-current={paper.id === activePaperId ? 'true' : undefined}
              className={`chat-paper-tab${paper.id === activePaperId ? ' active' : ''}`}
              key={paper.id}
              onClick={() => onSelectPaper(paper.id)}
              title={paper.title}
              type="button"
            >
              {paper.title}
            </button>
          ))}
          {recentPapers.length === 0 && <span className="chat-tab-empty">Choose a paper to begin</span>}
        </div>
      </header>
      {recentPapers.length === 0 ? (
        <div className="empty-state">Your paper conversations will appear here.</div>
      ) : (
        recentPapers.map((paper) => (
          <Activity
            key={paper.id}
            mode={paper.id === activePaperId ? 'visible' : 'hidden'}
          >
            <ChatPanel
              callbacks={requestCallbacks}
              onCitation={onCitation}
              paper={paper}
            />
          </Activity>
        ))
      )}
      <footer className="status-strip">
        <span><span className="live-dot" /> Evidence verifier online</span>
        <span>max 3 tool calls</span>
      </footer>
    </section>
  )
}

interface ChatPanelProps {
  paper: PaperSummary
  callbacks: RequestCallbacks
  onCitation: (citation: CitationRange) => void
}

function ChatPanel({ paper, callbacks, onCitation }: ChatPanelProps) {
  const [question, setQuestion] = useState('')
  const [entries, setEntries] = useState<ChatEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const submitQuestion = async () => {
    const prompt = question.trim()
    if (!prompt || isLoading) return
    const controller = new AbortController()
    const entryId = `${Date.now()}-${Math.random().toString(36).slice(2)}`
    setQuestion('')
    setIsLoading(true)
    setEntries((current) => [...current, { id: entryId, question: prompt, claims: [] }])
    callbacks.onStart(controller)

    try {
      const response = await fetch(
        apiUrl(`/v1/papers/${encodeURIComponent(paper.id)}/questions`),
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ question: prompt }),
          signal: controller.signal,
        },
      )
      if (!response.ok) throw new Error(`Question failed (${response.status}).`)
      if (!response.body) throw new Error('The server did not return a response stream.')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const consumeEvents = (complete: boolean) => {
        const blocks = buffer.split('\n\n')
        buffer = complete ? '' : (blocks.pop() ?? '')
        for (const block of blocks) {
          const eventLine = block.split('\n').find((line) => line.startsWith('event:'))
          const dataLine = block.split('\n').find((line) => line.startsWith('data:'))
          if (eventLine?.trim() !== 'event: claim' || !dataLine) continue
          const parsed = JSON.parse(dataLine.slice(5).trim()) as StreamClaim
          const claim = normalizeClaim(parsed.claim)
          if (!claim) continue
          setEntries((current) =>
            current.map((entry) =>
              entry.id === entryId ? { ...entry, claims: [...entry.claims, claim] } : entry,
            ),
          )
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        consumeEvents(false)
      }
      buffer += decoder.decode()
      consumeEvents(true)
    } catch (error) {
      if (!controller.signal.aborted) {
        setEntries((current) =>
          current.map((entry) =>
            entry.id === entryId
              ? { ...entry, error: error instanceof Error ? error.message : 'Question failed.' }
              : entry,
          ),
        )
      }
    } finally {
      setIsLoading(false)
      callbacks.onFinish(controller)
    }
  }

  return (
    <div className="chat-pane">
      <div className="chat-intro">
        Answers stay inside <strong>{paper.title}</strong>. Each surfaced claim is checked against
        source evidence.
      </div>
      <div className="chat-log" aria-live="polite">
        {entries.length === 0 ? (
          <div className="chat-empty">
            <span className="chat-empty-icon"><CircleHelp size={15} /></span>
            <strong>Start with a focused question</strong>
            <p>Ask about the method, key findings, or limitations in this paper.</p>
          </div>
        ) : (
          entries.slice(-5).map((entry) => (
            <div key={entry.id}>
              <p className="question-bubble">{entry.question}</p>
              {entry.claims.map((claim, index) => (
                <article className="claim-card" key={`${entry.id}:${index}`}>
                  <p>{claim.text}</p>
                  <div className="claim-quote">“{claim.quote}”</div>
                  <div className="claim-status">
                    <CheckCircle2 size={10} /> Verified against source
                    {claim.citation && (
                      <button
                        className="citation-button"
                        onClick={() => onCitation(claim.citation!)}
                        type="button"
                      >
                        p. {claim.citation.page_start}
                      </button>
                    )}
                  </div>
                </article>
              ))}
              {entry.error && <p className="chat-error">{entry.error}</p>}
            </div>
          ))
        )}
        {isLoading && (
          <div className="claim-card response-reservation" aria-label="Verifying response">
            <div className="skeleton-line" style={{ width: '84%' }} />
            <div className="skeleton-line" style={{ width: '61%' }} />
            <span className="claim-status"><LoaderCircle className="spin" size={10} /> Verifying evidence</span>
          </div>
        )}
      </div>
      <form
        className="chat-composer"
        onSubmit={(event) => {
          event.preventDefault()
          void submitQuestion()
        }}
      >
        <textarea
          aria-label={`Ask about ${paper.title}`}
          className="chat-textarea"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              void submitQuestion()
            }
          }}
          placeholder="Ask a question about this paper..."
          value={question}
        />
        <button
          aria-label="Send question"
          className="chat-send"
          disabled={!question.trim() || isLoading}
          type="submit"
        >
          <Send size={12} />
        </button>
      </form>
      <div className="chat-footnote">Only source-verified claims are shown</div>
    </div>
  )
}

function normalizeClaim(value: unknown): VerifiedClaim | null {
  if (typeof value !== 'object' || value === null || !('text' in value) || !('quote' in value)) {
    return null
  }
  const { text, quote } = value
  if (typeof text !== 'string' || typeof quote !== 'string') return null
  const citationValue = 'citation' in value ? normalizeCitation(value.citation) : undefined
  return citationValue
    ? { text, quote, citation: citationValue }
    : { text, quote }
}

function normalizeCitation(value: unknown): CitationRange | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  if (!('page_start' in value) || !('char_start' in value) || !('char_end' in value)) {
    return undefined
  }
  const { page_start, char_start, char_end } = value
  if (
    typeof page_start !== 'number' ||
    typeof char_start !== 'number' ||
    typeof char_end !== 'number'
  ) {
    return undefined
  }
  return { page_start, char_start, char_end }
}
