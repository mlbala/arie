import {
  useInfiniteQuery,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  Circle,
  FileStack,
  FolderOpen,
  History,
  LoaderCircle,
  Menu,
  Search,
  Settings2,
  ShieldCheck,
  PanelsTopLeft,
  Activity,
  SlidersHorizontal,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, useSyncExternalStore, type CSSProperties } from "react";
import {
  getPaper,
  getPapers,
  getPluginSlots,
  getSyncStatus,
  DOMAIN_LABELS,
  type CitationRange,
  type PaperSummary,
  type SyncJob,
} from "./api/client";
import { ChatWorkspace, type ChatProgress } from "./components/ChatWorkspace";
import { EvidenceDrawer } from "./components/EvidenceDrawer";
import { ImportPaper } from "./components/ImportPaper";
import { SettingsDialog } from "./components/SettingsDialog";
import { SkippedDrawer } from "./components/SkippedDrawer";
import { SyncControls } from "./components/SyncControls";
import { SelectedDocument, StorageHealth, SystemPane } from "./components/SystemPane";
import { PublicationWindowFields } from "./components/PublicationWindow";
import { RECENT_WINDOW, publicationLabel, publicationParams, type PublicationWindow } from "./publication";
import { loadSessions, MAX_HISTORY, paperSessionsReducer, saveSessions } from "./paperSessions";
import { ThemeControl } from "./components/ThemeControl";

type Layout = { library: "left" | "right"; import: "library" | "right"; libraryWidth: number; activityWidth: number; sizing: "auto" | "custom" };
function savedLayout(): Layout {
  try {
    const value = JSON.parse(localStorage.getItem("arie.ui.layout") ?? "null") as Layout | null;
    return {
      sizing: value?.sizing === "custom" ? "custom" : "auto",
      library: value?.library === "right" ? "right" : "left",
      import: value?.import === "library" ? "library" : "right",
      activityWidth: typeof value?.activityWidth === "number" ? Math.min(360, Math.max(260, value.activityWidth)) : 300,
      libraryWidth: typeof value?.libraryWidth === "number" ? Math.min(400, Math.max(240, value.libraryWidth)) : 280,
    };
  } catch { return { library: "left", import: "right", libraryWidth: 280, activityWidth: 300, sizing: "auto" }; }
}

/** Observe window resizing and browser zoom without storing a fixed screen size. */
function subscribeToViewport(onChange: () => void) {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
}
const viewportWidth = () => window.innerWidth;

const INGEST_STEPS = [
  ["discovery", "Discover papers"],
  ["download", "Validate & download"],
  ["extraction", "Extract PDF content"],
  ["indexing", "Index evidence"],
] as const;
const ANSWER_STEPS = [
  ["retrieval", "Find relevant passages"],
  ["reading", "Read source evidence"],
  ["verification", "Verify quotes & citations"],
] as const;

export default function App() {
  const queryClient = useQueryClient();
  const [sessions, dispatchSessions] = useReducer(paperSessionsReducer, undefined, loadSessions);
  // Reopen the same paper tabs after a reload; each tab then restores its saved chat.
  useEffect(() => saveSessions(sessions), [sessions]);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [dateWindow, setDateWindow] = useState<PublicationWindow>(RECENT_WINDOW);
  const [source, setSource] = useState("");
  const dateParams = publicationParams(dateWindow);
  const [domain, setDomain] = useState("");
  const [sort, setSort] = useState("newest");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const filtersRef = useRef<HTMLElement>(null);
  const filtersActive = publicationLabel(dateWindow) !== publicationLabel(RECENT_WINDOW) || Boolean(domain || source) || sort !== "newest";
  const [layout, setLayout] = useState<Layout>(savedLayout);
  const width = useSyncExternalStore(subscribeToViewport, viewportWidth);
  const activityBreakpoint = layout.sizing === "custom"
    ? Math.max(1180, layout.libraryWidth + layout.activityWidth + 620)
    : 1180;
  const docked = width >= activityBreakpoint;
  const compactLibrary = width <= 720;
  const [activityPanel, setActivityPanel] = useState({ docked, open: docked });
  const [libraryPanel, setLibraryPanel] = useState({ compact: compactLibrary, open: false });
  // Recompute visibility when crossing a layout boundary, preserving choices within it.
  if (activityPanel.docked !== docked) setActivityPanel({ docked, open: docked });
  if (libraryPanel.compact !== compactLibrary) setLibraryPanel({ compact: compactLibrary, open: false });
  const systemOpen = activityPanel.docked === docked ? activityPanel.open : docked;
  const libraryOpen = libraryPanel.compact === compactLibrary && libraryPanel.open;
  const setSystemOpen = useCallback((open: boolean) => setActivityPanel((current) => ({ ...current, open })), []);
  const setLibraryOpen = useCallback((open: boolean) => setLibraryPanel((current) => ({ ...current, open })), []);
  const [importRevision, setImportRevision] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const changeLayout = (update: Partial<Layout>) => {
    const next = { ...layout, ...update, ...(update.libraryWidth !== undefined || update.activityWidth !== undefined ? { sizing: "custom" as const } : {}) };
    setLayout(next);
    if (update.import === "right") setSystemOpen(true);
    try { localStorage.setItem("arie.ui.layout", JSON.stringify(next)); } catch { /* Session layout still applies. */ }
  };
  const searchInput = useRef<HTMLInputElement | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [auditOpen, setAuditOpen] = useState(false);
  const [sourceOpen, setSourceOpen] = useState(false);
  const closeSource = useCallback(() => setSourceOpen(false), []);
  const [citation, setCitation] = useState<CitationRange | null>(null);
  const [importJob, setImportJob] = useState<SyncJob | null>(null);
  const importProgress = useCallback(
    (job: SyncJob | null) => setImportJob(job),
    [],
  );
  const [progress, setProgress] = useState<Record<string, ChatProgress>>({});
  const papersQuery = useInfiniteQuery({
    queryKey: ["papers", { ...dateParams, domain, source, search: debouncedSearch, sort }],
    queryFn: ({ pageParam, signal }) => getPapers(pageParam, signal, {
      ...dateParams, domain, source, search: debouncedSearch, sort: sort === "newest" ? "" : sort,
    }),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const syncQuery = useQuery({
    queryKey: ["sync"],
    queryFn: getSyncStatus,
    refetchInterval: 1500,
  });
  const pluginsQuery = useQuery({
    queryKey: ["plugins"],
    queryFn: getPluginSlots,
  });
  const job = syncQuery.data?.job;
  const syncing = job?.status === "running" || job?.status === "queued";
  useEffect(() => {
    void queryClient.invalidateQueries({ queryKey: ["papers"] });
    void queryClient.invalidateQueries({ queryKey: ["skipped"] });
  }, [job?.indexed, job?.status, queryClient]);
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      const editing =
        event.target instanceof HTMLElement &&
        (event.target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName));
      if (
        (event.key === "/" && !editing) ||
        ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k")
      ) {
        event.preventDefault();
        setLibraryOpen(true);
        searchInput.current?.focus();
      }
      if (event.key === "Escape") {
        setFiltersOpen(false);
        setLibraryOpen(false);
        setSystemOpen(false);
      }
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, [setLibraryOpen, setSystemOpen]);
  useEffect(() => {
    if (!filtersOpen) return;
    // Close the filter popover on a click outside the library section.
    const outside = (event: PointerEvent) => {
      if (!filtersRef.current?.contains(event.target as Node)) setFiltersOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [filtersOpen]);
  const loadedPapers = useMemo(
    () => papersQuery.data?.pages.flatMap((page) => page.items) ?? [],
    [papersQuery.data],
  );
  // Pinned and history papers may fall outside the current filter, so fetch them individually.
  const sessionIds = [...sessions.pinned, ...sessions.history];
  const missingIds = sessionIds.filter((id) => !loadedPapers.some((paper) => paper.id === id));
  const sessionPapers = useQueries({
    queries: missingIds.map((id) => ({ queryKey: ["paper", id], queryFn: () => getPaper(id) })),
  });
  const papersById = useMemo(() => {
    const merged = new Map<string, PaperSummary>(loadedPapers.map((paper) => [paper.id, paper]));
    for (const query of sessionPapers) if (query.data) merged.set(query.data.id, query.data);
    return merged;
  }, [loadedPapers, sessionPapers]);
  const resolve = (ids: string[]) =>
    ids.flatMap((id) => papersById.get(id) ?? []);
  const activeId = sessions.activeId ?? (debouncedSearch
    ? null
    : sessions.pinned.at(-1) ?? (sessions.touched ? null : loadedPapers[0]?.id ?? null));
  const activePaper = activeId ? papersById.get(activeId) : undefined;
  // An unpinned default paper still gets a tab so it can be closed like any other.
  const pinnedPapers = resolve(
    activeId && !sessions.pinned.includes(activeId) ? [activeId] : sessions.pinned,
  );
  const historyPapers = resolve(sessions.history);
  const selectPaper = useCallback((id: string) => {
    dispatchSessions({ type: "open", id });
    setCitation(null);
    setSourceOpen(false);
    setLibraryOpen(false);
  }, [setLibraryOpen]);
  const filtered = loadedPapers;
  const chatProgress = activeId ? progress[activeId] : undefined;
  const chatRunning =
    chatProgress && !["idle", "complete", "error"].includes(chatProgress.stage);
  const importing =
    importJob?.status === "running" || importJob?.status === "queued";
  const showAnswer = Boolean(
    chatProgress && chatProgress.stage !== "idle" && !importing,
  );
  const steps = showAnswer ? ANSWER_STEPS : INGEST_STEPS;
  const pipelineJob = importing || (!syncing && importJob) ? importJob : job;
  const processing = importing || syncing;
  const phase = showAnswer ? chatProgress?.stage : pipelineJob?.phase;
  const completed = showAnswer
    ? phase === "complete"
    : pipelineJob?.status === "completed";
  const stepIndex = steps.findIndex(([key]) => key === phase);
  const llmSlot = pluginsQuery.data?.plugins.find((plugin) => plugin.slot === "llm");
  const llm = llmSlot?.configured.use;
  const activeProvider = llmSlot?.providers?.find((provider) => provider.name === llm);

  return (
    <main className={`research-shell library-${layout.library}`} style={{ "--library-width": layout.sizing === "auto" ? "clamp(240px, 18vw, 340px)" : `${layout.libraryWidth}px`, "--activity-width": layout.sizing === "auto" ? "clamp(260px, 20vw, 340px)" : `${layout.activityWidth}px` } as CSSProperties}>
      {libraryOpen && (
        <button
          className="library-scrim"
          aria-label="Close library"
          onClick={() => setLibraryOpen(false)}
        />
      )}
      <aside
        className={`library-sidebar${libraryOpen ? " is-open" : ""}`}
        aria-label="Research library and pipeline"
      >
        <a className="research-brand" href="/" aria-label="ARIE home">
          <span className="brand-mark">
            <FileStack size={20} />
          </span>
          <span>
            ARIE<small>Research, grounded.</small>
          </span>
        </a>
        <div className="sidebar-scroll">
        {layout.import === "library" && <ImportPaper key={importRevision} onImported={selectPaper} onProgress={importProgress} />}
        {historyPapers.length > 0 && (
          <section className="history-section" aria-label="Recently viewed papers">
            <div className="library-heading">
              <span>
                <History size={15} /> Recently viewed
              </span>
              <button className="text-button history-clear" onClick={() => dispatchSessions({ type: "clearHistory" })}>
                Clear history
              </button>
            </div>
            <ul className="history-list">
              {historyPapers.map((paper) => (
                <li key={paper.id}>
                  <button title={paper.title} onClick={() => selectPaper(paper.id)}>
                    <BookOpen size={12} />
                    <span>{paper.title}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="history-caption">{historyPapers.length} of {MAX_HISTORY} · reopen to pin again</p>
          </section>
        )}
        <section className="library-section" aria-label="Library papers" ref={filtersRef}>
        <div className="library-heading">
          <span>
            <FolderOpen size={15} /> Your library
            <span className="library-count">{loadedPapers.length}{papersQuery.hasNextPage ? "+" : ""}</span>
          </span>
          <button
            className={`icon-button library-filter-toggle${filtersActive ? " is-active" : ""}`}
            aria-label="Filter & sort"
            aria-expanded={filtersOpen}
            aria-controls="library-filter-panel"
            title={`Filter & sort · ${publicationLabel(dateWindow)}`}
            onClick={() => setFiltersOpen((open) => !open)}
          >
            <SlidersHorizontal size={14} />
          </button>
        </div>
        <label className="library-search">
          <Search size={14} />
          <input
            ref={searchInput}
            aria-label="Search library"
            placeholder="Title, topic, or a few words…"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              dispatchSessions({ type: "deselect" });
              setCitation(null);
              setSourceOpen(false);
            }}
          />
          <kbd>/</kbd>
        </label>
        {filtersOpen && (
        <div className="library-filters" id="library-filter-panel" role="group" aria-label="Filter and sort papers">
          <p className="library-filter-summary">{publicationLabel(dateWindow)}</p>
          <PublicationWindowFields value={dateWindow} allowAll onChange={setDateWindow} />
          <label>Source<select value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="">All sources</option>{[...new Set([...(syncQuery.data?.sources ?? []), ...loadedPapers.map((paper) => paper.source)])].sort().map((name) => <option key={name} value={name}>{name}</option>)}
          </select></label>
          <label>Research domain<select value={domain} onChange={(e) => setDomain(e.target.value)}>
            <option value="">All domains</option>
            {Object.entries(DOMAIN_LABELS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select></label>
          <label>Sort papers<select value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="domain">Research domain</option>
          </select></label>
          <button className="text-button" onClick={() => { setDateWindow(RECENT_WINDOW); setDomain(""); setSource(""); setSearch(""); setSort("newest"); }}>Reset filters</button>
          <button className="button filter-results-button" onClick={() => setFiltersOpen(false)}>Show results</button>
        </div>
        )}
        {debouncedSearch && <p className="search-help" role="status">Relevant papers · choose one to ask questions</p>}
        <div className="library-list">
          {papersQuery.isPending ? (
            <p className="sidebar-message">Loading your library…</p>
          ) : papersQuery.error ? (
            <div className="sidebar-message error" role="alert">
              {papersQuery.error.message}
              <button
                className="text-button"
                onClick={() => void papersQuery.refetch()}
              >
                Try again
              </button>
            </div>
          ) : filtered.length ? (
            filtered.map((paper) => (
              <button
                className={`library-paper${paper.id === activeId ? " selected" : ""}`}
                key={paper.id}
                onClick={() => selectPaper(paper.id)}
              >
                <BookOpen size={14} />
                <span>
                  <strong>{paper.title}</strong>
                  {debouncedSearch && <span className="search-result-detail"><em>{paper.match_reason ?? "Relevant paper"}</em>{paper.abstract.slice(0, 130)}{paper.abstract.length > 130 ? "…" : ""}</span>}
                  <small>
                    {paper.source} <span>·</span>{" "}
                    {formatDate(paper.published_at)}
                  </small>
                  <span className="topic-tags">{paper.tags?.slice(0, 3).map((tag) => <span key={tag.tag} title={`${tag.basis}: ${tag.quote ?? "Uncategorized"}`}>{tag.label}</span>)}</span>
                </span>
                {paper.id === activeId && <ChevronRight size={13} />}
              </button>
            ))
          ) : (
            <div className="library-empty">
              <BookOpen size={24} />
              <strong>
                {search || dateWindow.mode !== "all" || domain || source ? "No matching papers" : "Make room for new ideas."}
              </strong>
              <p>
                {search || dateWindow.mode !== "all" || domain || source
                  ? "Try another publication window, source, domain or search."
                  : "Sync your sources to build a library of extracted research."}
              </p>
            </div>
          )}
          {papersQuery.hasNextPage && (
            <button
              className="load-papers"
              disabled={papersQuery.isFetchingNextPage}
              onClick={() => void papersQuery.fetchNextPage()}
            >
              {papersQuery.isFetchingNextPage ? "Loading…" : "Load more papers"}
            </button>
          )}
        </div>
        </section>
        {activePaper && <SelectedDocument key={activePaper.id} paper={activePaper} onDeleted={(id, pdfOnly) => {
          setSourceOpen(false);
          setImportJob(null);
          setImportRevision((value) => value + 1);
          if (!pdfOnly) dispatchSessions({ type: "forget", id });
        }} />}
        </div>
        <div className="library-storage"><StorageHealth /></div>
        <div className="sidebar-appearance"><ThemeControl /></div>
        <div className="sidebar-footer">
          <button onClick={() => setAuditOpen(true)}>Sync audit</button>
          <button aria-label="Settings" onClick={() => setSettingsOpen(true)}>
            <Settings2 size={15} />
          </button>
        </div>
      </aside>
      <section className="conversation-main">
        <header className="conversation-topbar">
          <div>
            <button
              aria-label="Open library"
              className="icon-button mobile-library-toggle"
              onClick={() => setLibraryOpen(true)}
            >
              <Menu size={18} />
            </button>
            <BookOpen size={15} />
            <span className="workspace-title">Paper chat</span>
          </div>
          <div>
            <details className="layout-options">
              <summary><PanelsTopLeft size={13} /> Layout</summary>
              <div>
                <label>Library position<select value={layout.library} onChange={(e) => changeLayout({ library: e.target.value as Layout["library"] })}><option value="left">Left side</option><option value="right">Right side</option></select></label>
                <label>Panel sizing<select value={layout.sizing} onChange={(e) => changeLayout({ sizing: e.target.value as Layout["sizing"] })}><option value="auto">Automatic · fit the window</option><option value="custom">Custom widths</option></select></label>
                {layout.sizing === "custom" && <>
                <label>Library width · {layout.libraryWidth}px<input aria-label="Library width" type="range" min="240" max="400" step="10" value={layout.libraryWidth} onChange={(e) => changeLayout({ libraryWidth: Number(e.target.value) })} /></label>
                <label>Activity pane width · {layout.activityWidth}px<input aria-label="Activity pane width" type="range" min="260" max="360" step="10" value={layout.activityWidth} onChange={(e) => changeLayout({ activityWidth: Number(e.target.value) })} /></label>
                </>}
                <label>Import form position<select value={layout.import} onChange={(e) => changeLayout({ import: e.target.value as Layout["import"] })}><option value="right">Right pane</option><option value="library">In the library</option></select></label>
              </div>
            </details>
            <button className="button" aria-expanded={systemOpen} onClick={() => setSystemOpen(!systemOpen)}><Activity size={13} />Import & sync</button>
            <span className="local-label">
              <span className="live-dot" /> Local library
            </span>
            {activePaper && (
              <button
                className="button source-open"
                onClick={() => {
                  setCitation(null);
                  setSourceOpen(true);
                }}
              >
                <BookOpen size={13} />
                View paper
                <ArrowUpRight size={12} />
              </button>
            )}
          </div>
        </header>
        {activeProvider && !activeProvider.key_configured && <div className="model-config-notice" role="alert"><span>{activeProvider.label} API key is not configured. Source excerpts remain available.</span><button className="text-button" onClick={() => setSettingsOpen(true)}>Open Settings</button></div>}
        {syncing && (
          <div className="sync-banner" role="status">
            <LoaderCircle size={13} className="spin" />
            <span>
              {job.message}
              {job.current_paper && <small> · {job.current_paper}</small>}
            </span>
            <strong>{job.indexed} ready</strong>
          </div>
        )}
        {debouncedSearch && !activeId && <div className="search-choice-notice" role="status">Choose a paper from the library results to start asking questions.</div>}
        <ChatWorkspace
          papers={activeId ? pinnedPapers : []}
          activePaperId={activeId}
          onSelectPaper={selectPaper}
          onClosePaper={(id) => {
            if (id === activeId) {
              setCitation(null);
              setSourceOpen(false);
            }
            dispatchSessions({ type: "close", id });
          }}
          onCitation={(range) => {
            setCitation(range);
            setSourceOpen(true);
          }}
          onProgress={(id, update) =>
            setProgress((current) => ({ ...current, [id]: update }))
          }
          modes={sessions.modes}
          onModeChange={(id, mode) => dispatchSessions({ type: "setMode", id, mode })}
        />
        <footer className="research-footer">
          <span>
            <ShieldCheck size={12} /> Every answer anchored to its source
          </span>
          <span>
            {typeof llm === "string" ? llm : "Research assistant"}
            <span className="footer-separator">·</span> ARIE 0.1.0
          </span>
        </footer>
      </section>
      <SystemPane docked={docked} open={systemOpen} onClose={() => setSystemOpen(false)}>
        <SyncControls status={syncQuery.data} error={syncQuery.error} />
        {layout.import === "right" && <ImportPaper key={importRevision} onImported={selectPaper} onProgress={importProgress} />}
        <section
          className="pipeline-card"
          aria-label={showAnswer ? "Answer pipeline" : "Sync pipeline"}
        >
          <div className="pipeline-title">
            <span>{showAnswer ? "Answer pipeline" : "Ingestion pipeline"}</span>
            <span
              className={`pipeline-state${chatRunning || (!showAnswer && processing) ? " active" : ""}`}
            >
              {chatRunning || (!showAnswer && processing)
                ? "Live"
                : completed
                  ? "Complete"
                  : phase === "error"
                    ? "Stopped"
                    : "Ready"}
            </span>
          </div>
          <ol className="pipeline-steps">
            {steps.map(([key, label], index) => {
              const done = completed || (stepIndex >= 0 && index < stepIndex);
              const current =
                key === phase && (showAnswer ? chatRunning : processing);
              return (
                <li
                  key={key}
                  className={done ? "done" : current ? "current" : ""}
                >
                  <span className="pipeline-node">
                    {done ? (
                      <Check size={11} />
                    ) : current ? (
                      <LoaderCircle size={12} className="spin" />
                    ) : (
                      <Circle size={10} />
                    )}
                  </span>
                  <span>{label}</span>
                </li>
              );
            })}
          </ol>
          <p className="pipeline-caption" aria-live="polite">
            {showAnswer
              ? chatProgress?.message
              : processing
                ? (pipelineJob?.current_paper ?? pipelineJob?.message)
                : "PDF pages → searchable evidence"}
          </p>
          {showAnswer && (
            <button
              className="text-button"
              onClick={() =>
                activeId &&
                setProgress((current) => ({
                  ...current,
                  [activeId]: { stage: "idle", message: "" },
                }))
              }
            >
              View sync pipeline <ArrowUpRight size={10} />
            </button>
          )}
        </section>
      </SystemPane>
      {sourceOpen && activeId && (
        <EvidenceDrawer
          key={`${activeId}:${activePaper?.revision}:${citation?.page_start}:${citation?.char_start}`}
          paperId={activeId}
          citation={citation}
          onClose={closeSource}
        />
      )}
      {settingsOpen && (
        <SettingsDialog onClose={() => setSettingsOpen(false)} />
      )}
      {auditOpen && <SkippedDrawer onClose={() => setAuditOpen(false)} />}
    </main>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Undated"
    : new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      }).format(date);
}
