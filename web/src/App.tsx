import { useInfiniteQuery, useMutation, useQuery } from '@tanstack/react-query'
import {
  ArrowDownToLine,
  ChevronDown,
  CircleHelp,
  Command,
  Database,
  ExternalLink,
  FileStack,
  Layers3,
  RefreshCw,
  Settings2,
  Sparkles,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  getPapers,
  getPluginSlots,
  syncSource,
  type CitationRange,
  type PaperSummary,
} from './api/client'
import { BriefViewer } from './components/BriefViewer'
import { ChatWorkspace } from './components/ChatWorkspace'
import { PaperFeed } from './components/PaperFeed'
import { SettingsDialog } from './components/SettingsDialog'
import { SkippedDrawer } from './components/SkippedDrawer'

const SLOT_INFO = [
  { slot: 'llm', label: 'LLM', icon: Sparkles },
  { slot: 'vector_store', label: 'Vector', icon: Layers3 },
  { slot: 'database', label: 'DB', icon: Database },
]

export default function App() {
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null)
  const [recentPaperIds, setRecentPaperIds] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [skippedOpen, setSkippedOpen] = useState(false)
  const [syncMessage, setSyncMessage] = useState<string | null>(null)
  const [activeCitation, setActiveCitation] = useState<CitationRange | null>(null)
  const papersQuery = useInfiniteQuery({
    queryKey: ['papers'],
    queryFn: ({ pageParam, signal }) => getPapers(pageParam, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  })
  const pluginsQuery = useQuery({ queryKey: ['plugins'], queryFn: getPluginSlots })
  const papers = useMemo(
    () => papersQuery.data?.pages.flatMap((page) => page.items) ?? [],
    [papersQuery.data],
  )

  const activePaperId = selectedPaperId ?? papers[0]?.id ?? null

  const recentPapers = useMemo(() => {
    const ids = activePaperId
      ? [activePaperId, ...recentPaperIds.filter((id) => id !== activePaperId)]
      : recentPaperIds
    return ids
      .slice(0, 5)
      .map((id) => papers.find((paper) => paper.id === id))
      .filter((paper): paper is PaperSummary => paper !== undefined)
  }, [activePaperId, papers, recentPaperIds])

  const selectPaper = (paperId: string) => {
    if (paperId !== activePaperId) setActiveCitation(null)
    setSelectedPaperId(paperId)
    setRecentPaperIds((current) => [paperId, ...current.filter((id) => id !== paperId)].slice(0, 5))
  }

  const syncMutation = useMutation({
    mutationFn: (source: string) => syncSource(source),
    onSuccess: ({ job_id }) => {
      setSyncMessage(`Sync started · ${job_id}`)
      window.setTimeout(() => setSyncMessage(null), 4200)
    },
  })

  const sourceProvider = pluginsQuery.data?.plugins.find((plugin) => plugin.slot === 'source')
  const sourceName =
    sourceProvider?.configured.use && typeof sourceProvider.configured.use === 'string'
      ? sourceProvider.configured.use
      : 'arxiv'

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <FileStack size={17} strokeWidth={2.2} />
          </div>
          <div>
            <div className="brand-name">ARIE</div>
            <span className="brand-caption">RESEARCH INTELLIGENCE</span>
          </div>
        </div>
        <SlotBadges
          plugins={pluginsQuery.data?.plugins ?? []}
          isLoading={pluginsQuery.isPending}
        />
        <div className="top-actions">
          <button
            className="button button-primary sync-button"
            disabled={syncMutation.isPending}
            onClick={() => syncMutation.mutate(sourceName)}
            type="button"
          >
            <RefreshCw className={syncMutation.isPending ? 'spin' : ''} size={12} />
            <span className="button-label">{syncMutation.isPending ? 'Syncing…' : 'Sync now'}</span>
          </button>
          <button
            className="button skipped-button"
            onClick={() => setSkippedOpen(true)}
            type="button"
          >
            <ArrowDownToLine size={12} />
            <span className="button-label">Skipped audit</span>
          </button>
          <button
            aria-label="Settings"
            className="icon-button"
            onClick={() => setSettingsOpen(true)}
            type="button"
          >
            <Settings2 size={16} />
          </button>
          <button aria-label="Help" className="icon-button help-button" type="button">
            <CircleHelp size={15} />
          </button>
        </div>
      </header>
      {(syncMessage || syncMutation.error) && (
        <div className={`sync-toast${syncMutation.error ? ' error' : ''}`} role="status">
          {syncMutation.error?.message ?? syncMessage}
          <button
            aria-label="Dismiss sync message"
            className="toast-dismiss"
            onClick={() => {
              syncMutation.reset()
              setSyncMessage(null)
            }}
            type="button"
          >
            <ChevronDown size={12} />
          </button>
        </div>
      )}
      <div className="workspace">
        <PaperFeed selectedPaperId={activePaperId} onSelectPaper={selectPaper} />
        <BriefViewer citation={activeCitation} paperId={activePaperId} />
        <ChatWorkspace
          activePaperId={activePaperId}
          onCitation={setActiveCitation}
          onSelectPaper={selectPaper}
          papers={recentPapers}
        />
      </div>
      <footer className="app-footer">
        <span><span className="live-dot" /> Local-first workspace</span>
        <span className="footer-center"><Command size={10} /> K search · J/K navigate</span>
        <span>ARIE · 0.1.0</span>
      </footer>
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
      {skippedOpen && <SkippedDrawer onClose={() => setSkippedOpen(false)} />}
    </main>
  )
}

interface SlotBadgesProps {
  plugins: Array<{ slot: string; configured: Record<string, unknown> }>
  isLoading: boolean
}

function SlotBadges({ plugins, isLoading }: SlotBadgesProps) {
  return (
    <div className="slot-row" aria-label="Active plugin slots">
      {SLOT_INFO.map(({ slot, label, icon: Icon }) => {
        const plugin = plugins.find((item) => item.slot === slot)
        const provider = plugin?.configured.use
        return (
          <div className="slot-badge" key={slot} title={`${label}: ${String(provider ?? 'not configured')}`}>
            <span className="slot-dot" />
            <Icon size={11} />
            <span>{label}</span>
            <span className="slot-vendor">
              {isLoading ? '…' : typeof provider === 'string' ? <><ExternalLink size={9} /> {provider}</> : 'offline'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
