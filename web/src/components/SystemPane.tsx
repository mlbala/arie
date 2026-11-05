import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ChevronRight, Database, HardDrive, Layers3, ListOrdered, ShieldCheck, Trash2, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { deletePaper, getStorageHealth, type AnswerModelHealth, type PaperSummary } from "../api/client";

const SOURCE_TOOLS_KEY = "arie.ui.sourceTools";

function readToolsOpen(): boolean {
  try {
    return localStorage.getItem(SOURCE_TOOLS_KEY) === "open";
  } catch {
    return false;
  }
}

/** Sync and import controls in one section, closed until opened; the choice is remembered. A running sync shows on the closed row. */
export function SourceTools({ label, syncing, children }: { label: string; syncing: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(readToolsOpen);
  const toggle = (next: boolean) => {
    setOpen(next);
    try {
      localStorage.setItem(SOURCE_TOOLS_KEY, next ? "open" : "closed");
    } catch {
      // Private windows and blocked storage just forget the choice.
    }
  };
  return <details className="source-tools" open={open} onToggle={(event) => toggle(event.currentTarget.open)}>
    <summary><ChevronRight size={12} aria-hidden /><span>{label}</span>
      {syncing && <span className="source-tools-state"><span className="live-dot" />Syncing</span>}
    </summary>
    {children}
  </details>;
}

function useStorageHealth() {
  return useQuery({ queryKey: ["storage"], queryFn: getStorageHealth, refetchInterval: 15000 });
}

/** Keep source controls in a bounded, independently scrollable activity pane. */
export function SystemPane({ open, docked, onClose, children }: { open: boolean; docked: boolean; onClose: () => void; children?: ReactNode }) {
  const health = useStorageHealth();
  return <>
    {open && !docked && <button className="system-scrim" aria-label="Close system pane" onClick={onClose} />}
    <aside className={`system-pane ${docked ? "is-docked" : "is-overlay"}${open ? " is-open" : ""}`} aria-label="Import and sync" inert={!open}>
      <header><span><Activity size={15} /> Sources & activity</span><button className="icon-button" aria-label="Close activity pane" onClick={onClose}><X size={15} /></button></header>
      <div className="system-scroll">{children}
        <section className="system-section"><h2>Extraction activity</h2>
          {health.data?.active_jobs.length ? health.data.active_jobs.map((job) => <div className="active-document" key={job.job_id}>
            <span className="health-pill running">{job.phase}</span><strong>{job.current_paper ?? "Discovering papers"}</strong>
            <p>{job.message}</p><small>{job.indexed} ready · {job.processed} examined · {job.failed} failed</small>
          </div>) : <p>{health.isPending ? "Checking activity…" : health.error ? "Activity unavailable" : "No extraction running"}</p>}
        </section>
      </div>
    </aside>
  </>;
}

/** Show measured storage next to the library; no estimated file or index sizes. */
export function StorageHealth() {
  const health = useStorageHealth();
  const data = health.data;
  const modelDegraded = data?.entailment?.status === "degraded" || data?.reranker?.status === "degraded";
  const allHealthy = data?.database.status === "healthy" && data.file_storage.status === "healthy" && !modelDegraded;
  return (
          <details className="system-section storage-section">
            <summary><HardDrive size={14} /><span>Storage & health</span><span className={`storage-status ${allHealthy ? "healthy" : ""}`}>{health.error ? "Unavailable" : data ? allHealthy ? "Healthy" : modelDegraded ? "Model degraded" : "Check status" : "Checking"}</span></summary>
            <div className="storage-details">
            {health.error && <p className="sync-error" role="alert">{health.error.message}</p>}
            <article><div><Database size={14} /><strong>Database</strong><span className={`health-pill ${data?.database.status ?? "unknown"}`}>{data?.database.status ?? "Checking"}</span></div>
              <p>{data?.database.provider ?? "—"} · {bytes(data?.database.size_bytes)}</p>
              <small>{data?.database.papers ?? "—"} papers · {data?.database.evidence_chunks ?? "—"} evidence chunks</small>
            </article>
            <article><div><Layers3 size={14} /><strong>Vector index</strong><span className={`health-pill ${data?.vector_store.status ?? "unknown"}`}>{data?.vector_store.status ?? "Checking"}</span></div>
              <p>{data?.vector_store.provider ?? "—"} · {data?.vector_store.vectors ?? "—"} vectors · {bytes(data?.vector_store.size_bytes)}</p>
              <small>{data?.vector_store.detail ?? "Waiting for provider status."}</small>
            </article>
            {data?.entailment && <AnswerModelCard icon={<ShieldCheck size={14} />} title="Claim checker" purpose="Tier 3 entailment" model={data.entailment} />}
            {data?.reranker && <AnswerModelCard icon={<ListOrdered size={14} />} title="Passage reranker" purpose="question relevance" model={data.reranker} />}
            <article><div><HardDrive size={14} /><strong>PDF storage</strong><span className={`health-pill ${data?.file_storage.status ?? "unknown"}`}>{data?.file_storage.status ?? "Checking"}</span></div>
              <p>{data?.file_storage.provider ?? "—"} · {bytes(data?.file_storage.size_bytes)}{data?.file_storage.status === "partial" ? " (partial)" : ""}</p>
              <small>{data?.file_storage.files ?? "—"} PDF files · {bytes(data?.file_storage.disk_free_bytes)} free</small>
              {data?.file_storage.disk_total_bytes && data.file_storage.disk_free_bytes !== null ? <meter min={0} max={data.file_storage.disk_total_bytes} value={data.file_storage.disk_total_bytes - data.file_storage.disk_free_bytes} aria-label="Disk space used" /> : null}
            </article>
            {data && <p className="probe-time">Storage checked {new Date(data.checked_at).toLocaleTimeString()}. Refreshes every 15 seconds.</p>}
            </div>
          </details>

  );
}

/** One answer-path model; when degraded, `detail` carries the reason and `last call` shows how long it ran. */
function AnswerModelCard({ icon, title, purpose, model }: { icon: ReactNode; title: string; purpose: string; model: AnswerModelHealth }) {
  const latency = model.last_latency_seconds;
  return (
    <article><div>{icon}<strong>{title}</strong><span className={`health-pill ${model.status}`}>{model.status}</span></div>
      <p>{model.plugin} · {purpose}{latency == null ? "" : ` · last call ${latency.toFixed(1)} s`}</p>
      <small>{model.detail}</small>
    </article>
  );
}

/** Confirm deletion against the displayed paper and block removal during ingestion. */
export function SelectedDocument({ paper, onDeleted }: { paper: PaperSummary; onDeleted: (id: string, pdfOnly: boolean) => void }) {
  const client = useQueryClient();
  const health = useStorageHealth();
  const busy = !health.data || Boolean(health.data.active_jobs.length);
  const [deleteMode, setDeleteMode] = useState<"pdf" | "paper" | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const remove = useMutation({
    mutationFn: ({ id, pdfOnly }: { id: string; pdfOnly: boolean }) => deletePaper(id, pdfOnly),
    onSuccess: (_, { id, pdfOnly }) => {
      setDeleteMode(null);
      client.removeQueries({ queryKey: ["paper", id] });
      onDeleted(id, pdfOnly);
      void client.invalidateQueries({ queryKey: ["papers"] });
      void client.invalidateQueries({ queryKey: ["paper"] });
      void client.invalidateQueries({ queryKey: ["storage"] });
    },
  });
  return (
    <section className="system-section selected-document">
            <h2>Selected document</h2><strong title={paper.title}>{paper.title}</strong>
            <p>{paper.status} · {paper.page_count ?? 0} pages · {paper.chunk_count ?? "—"} evidence chunks</p>
            <div className="topic-tags">{paper.tags?.slice(0, 3).map((tag) => <span key={tag.tag} title={`${tag.basis}${tag.page ? ` ${tag.page}` : ""}${tag.quote ? `: “${tag.quote}”` : ""}`}>{tag.label}</span>)}</div>
            {paper.pdf_deleted && <p className="retained-evidence-note">PDF removed · Extracted text retained</p>}
            <div className="delete-actions">
              <button disabled={busy || remove.isPending || paper.pdf_deleted} onClick={() => { setDeleteMode("pdf"); setDeletingId(paper.id); remove.reset(); }}><Trash2 size={11} /> Delete PDF</button>
              <button disabled={busy || remove.isPending} onClick={() => { setDeleteMode("paper"); setDeletingId(paper.id); remove.reset(); }}>Delete paper</button>
            </div>
            {busy && <small>Pause sync and let imports finish to delete files.</small>}
            {deleteMode && deletingId === paper.id && <div className="delete-confirm" role="group" aria-label="Confirm deletion">
              <p>{deleteMode === "pdf" ? "Delete the original PDF? Extracted text stays available for chat." : "Delete this paper, its PDF, metadata and extracted evidence? This cannot be undone."}</p>
              <button className="button danger-button" disabled={busy || remove.isPending} onClick={() => remove.mutate({ id: paper.id, pdfOnly: deleteMode === "pdf" })}>{remove.isPending ? "Deleting…" : "Confirm delete"}</button>
              <button className="text-button" onClick={() => setDeleteMode(null)}>Cancel</button>
            </div>}
            {remove.error && <p className="sync-error" role="alert">{remove.error.message}</p>}
    </section>

  );
}

function bytes(value: number | null | undefined): string {
  if (value == null) return "Size unavailable";
  if (value < 1024) return `${value} B`;
  const exponent = Math.min(3, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** exponent).toFixed(1)} ${["B", "KB", "MB", "GB"][exponent]}`;
}
