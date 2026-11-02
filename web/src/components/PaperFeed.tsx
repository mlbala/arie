import { useInfiniteQuery, useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { BookOpen, CircleAlert, Search, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getPaper, getPapers, type PaperSummary } from '../api/client'

interface PaperFeedProps {
  selectedPaperId: string | null
  onSelectPaper: (paperId: string) => void
}

const ROW_HEIGHT = 98

export function PaperFeed({ selectedPaperId, onSelectPaper }: PaperFeedProps) {
  const queryClient = useQueryClient()
  const scrollRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'indexed' | 'skipped'>('all')
  const papersQuery = useInfiniteQuery({
    queryKey: ['papers'],
    queryFn: ({ pageParam, signal }) => getPapers(pageParam, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  })
  const papers = useMemo(
    () => papersQuery.data?.pages.flatMap((page) => page.items) ?? [],
    [papersQuery.data],
  )
  const filteredPapers = useMemo(() => {
    const query = search.trim().toLowerCase()
    return papers.filter((paper) => {
      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'indexed' && paper.status !== 'SKIPPED') ||
        (statusFilter === 'skipped' && paper.status === 'SKIPPED')
      return (
        matchesStatus &&
        (!query || `${paper.title} ${paper.abstract} ${paper.source}`.toLowerCase().includes(query))
      )
    })
  }, [papers, search, statusFilter])
  const virtualizer = useVirtualizer({
    count: filteredPapers.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 5,
  })

  const focusSearch = useCallback(() => searchRef.current?.focus(), [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target
      const isEditing =
        target instanceof HTMLElement &&
        (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        focusSearch()
        return
      }
      if (!isEditing && event.key === '/') {
        event.preventDefault()
        focusSearch()
        return
      }
      if (isEditing || !['j', 'k'].includes(event.key.toLowerCase())) return
      event.preventDefault()
      const currentIndex = filteredPapers.findIndex((paper) => paper.id === selectedPaperId)
      const nextIndex = Math.max(
        0,
        Math.min(filteredPapers.length - 1, currentIndex + (event.key.toLowerCase() === 'j' ? 1 : -1)),
      )
      const nextPaper = filteredPapers[nextIndex]
      if (nextPaper) {
        onSelectPaper(nextPaper.id)
        virtualizer.scrollToIndex(nextIndex, { align: 'auto' })
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [filteredPapers, focusSearch, onSelectPaper, selectedPaperId, virtualizer])

  const prefetchPaper = (paperId: string) => {
    void queryClient.prefetchQuery({
      queryKey: ['paper', paperId],
      queryFn: () => getPaper(paperId),
      staleTime: 60_000,
    })
  }

  const onScroll = () => {
    const element = scrollRef.current
    if (
      element &&
      element.scrollHeight - element.scrollTop - element.clientHeight < 120 &&
      papersQuery.hasNextPage &&
      !papersQuery.isFetchingNextPage
    ) {
      void papersQuery.fetchNextPage()
    }
  }

  return (
    <section className="pane feed-pane" aria-label="Paper feed">
      <header className="pane-header">
        <div className="pane-title-row">
          <div>
            <p className="eyebrow">Your library</p>
            <h1 className="pane-title">Research feed</h1>
          </div>
          <span className="count-pill">{papers.length} papers</span>
        </div>
        <label className="search-box">
          <Search size={13} aria-hidden="true" />
          <input
            ref={searchRef}
            aria-label="Search papers"
            placeholder="Search papers..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <kbd className="keycap">/</kbd>
        </label>
      </header>
      <div className="filter-row" role="group" aria-label="Filter papers">
        {(['all', 'indexed', 'skipped'] as const).map((filter) => (
          <button
            className={`chip-button${statusFilter === filter ? ' active' : ''}`}
            key={filter}
            onClick={() => setStatusFilter(filter)}
            type="button"
          >
            {filter === 'all' ? 'All papers' : filter === 'indexed' ? 'Indexed' : 'Skipped'}
          </button>
        ))}
      </div>
      <div className="feed-scroll" ref={scrollRef} onScroll={onScroll}>
        {papersQuery.isPending ? (
          <div className="loading-state">Loading your library…</div>
        ) : papersQuery.isError ? (
          <div className="error-state">
            <span>
              <CircleAlert size={18} />
              <br />
              {papersQuery.error.message}
              <br />
              <button
                className="text-button"
                onClick={() => void papersQuery.refetch()}
                type="button"
              >
                Try again
              </button>
            </span>
          </div>
        ) : filteredPapers.length === 0 ? (
          <div className="empty-state">No papers match this search.</div>
        ) : (
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              position: 'relative',
              width: '100%',
            }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const paper = filteredPapers[virtualRow.index]
              if (!paper) return null
              return (
                <PaperRow
                  index={virtualRow.index}
                  key={paper.id}
                  onPrefetch={prefetchPaper}
                  onSelect={onSelectPaper}
                  paper={paper}
                  selected={paper.id === selectedPaperId}
                  start={virtualRow.start}
                />
              )
            })}
          </div>
        )}
        {papersQuery.isFetchingNextPage && <div className="loading-state">Loading more…</div>}
      </div>
      <footer className="feed-hint">
        <span><kbd>j</kbd> <kbd>k</kbd> navigate</span>
        <span><Sparkles size={10} /> grounded library</span>
      </footer>
    </section>
  )
}

interface PaperRowProps {
  index: number
  paper: PaperSummary
  selected: boolean
  start: number
  onSelect: (paperId: string) => void
  onPrefetch: (paperId: string) => void
}

function PaperRow({ index, paper, selected, start, onSelect, onPrefetch }: PaperRowProps) {
  const metadata = Object.entries(paper.custom_metadata ?? {}).slice(0, 1)
  return (
    <button
      aria-current={selected ? 'true' : undefined}
      aria-label={`Paper ${index + 1}: ${paper.title}`}
      className={`feed-row${selected ? ' selected' : ''}`}
      onClick={() => onSelect(paper.id)}
      onMouseEnter={() => onPrefetch(paper.id)}
      onFocus={() => onPrefetch(paper.id)}
      style={{ height: `${ROW_HEIGHT}px`, transform: `translateY(${start}px)` }}
      type="button"
    >
      <div className="paper-meta">
        <span className="source-pill"><BookOpen size={10} />{paper.source}</span>
        <span>·</span>
        <span>{formatDate(paper.published_at)}</span>
        <span className={`status-pill${paper.status === 'SKIPPED' ? ' skipped' : ''}`}>
          {paper.status === 'SKIPPED' ? 'SKIPPED' : 'INDEXED'}
        </span>
      </div>
      <span className="feed-title">{paper.title}</span>
      <div className="paper-meta">
        {paper.affiliation && <span className="meta-badge">{paper.affiliation}</span>}
        {metadata.map(([key, value]) => (
          <span className="meta-badge" key={key} title={`${key}: ${value}`}>
            {key}: {value}
          </span>
        ))}
        {!paper.affiliation && metadata.length === 0 && (
          <span>{paper.overridden_by_user ? 'User reviewed' : 'Verified source'}</span>
        )}
      </div>
    </button>
  )
}

function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? 'recent'
    : new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(date)
}
