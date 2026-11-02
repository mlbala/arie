import { useMutation, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, SlidersHorizontal } from "lucide-react";
import { useRef, useState } from "react";
import { PublicationWindowFields } from "./PublicationWindow";
import {
  DEFAULT_SYNC, pauseSync, saveSyncSettings, syncSource,
  type SyncSettings, type SyncStatus,
} from "../api/client";

export function SyncControls({ status, error }: { status: SyncStatus | undefined; error: Error | null }) {
  const client = useQueryClient();
  const form = useRef<HTMLFormElement>(null);
  const [draft, setDraft] = useState<SyncSettings | null>(null);
  const [source, setSource] = useState("all");
  const settings = draft ?? status?.settings ?? DEFAULT_SYNC;
  const job = status?.job;
  const busy = job?.status === "running" || job?.status === "queued";
  const start = useMutation({
    mutationFn: (resume: boolean) => syncSource(source, settings, resume ? job?.job_id : undefined),
    onSuccess: (result) => client.setQueryData<SyncStatus>(["sync"], (old) => ({
      ...old, initial_months: old?.initial_months ?? 6, job: result,
    })),
  });
  const pause = useMutation({
    mutationFn: () => pauseSync(job!.job_id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["sync"] }),
  });
  const save = useMutation({
    mutationFn: () => saveSyncSettings(settings),
    onSuccess: () => client.invalidateQueries({ queryKey: ["sync"] }),
  });
  const update = (value: Partial<SyncSettings>) => {
    setDraft({ ...settings, ...value });
    save.reset();
    start.reset();
  };
  const feedback = start.error?.message ?? pause.error?.message ?? save.error?.message ?? error?.message;
  return (
    <section className="sync-card" aria-label="Source sync">
      <div><span className="live-dot" /> Research sources</div>
      <p>{busy ? `${job.processed} examined · ${job.indexed} ready` : windowLabel(settings)}</p>
      <form ref={form} onSubmit={(event) => { event.preventDefault(); start.mutate(false); }}>
        <details className="sync-options">
          <summary><SlidersHorizontal size={12} /> Sync options</summary>
          <label>Sources<select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="all">All enabled sources</option>
            {(status?.sources ?? []).map((name) => <option key={name} value={name}>{name}</option>)}
          </select></label>
          <PublicationWindowFields value={settings} onChange={(window) => {
            if (window.mode === "all") return;
            setDraft({ ...window, mode: window.mode, batch_size: settings.batch_size, max_papers: settings.max_papers });
            save.reset(); start.reset();
          }} />
          <label>Paper limit per run<input type="number" min="1" max="10000" required value={settings.max_papers} onChange={(e) => update({ max_papers: Number(e.target.value) })} /></label>
          <label>Metadata page size<input type="number" min="1" max="100" required value={settings.batch_size} onChange={(e) => update({ batch_size: Number(e.target.value) })} /></label>
          <p>Months include their last day (UTC). The paper limit counts examined records, including skips. Resume continues from the saved position.</p>
          <button className="button" type="button" disabled={save.isPending} onClick={() => { if (form.current?.reportValidity()) save.mutate(); }}>Save defaults to arie.yaml</button>
          {save.isSuccess && <p role="status">Defaults saved to arie.yaml.</p>}
        </details>
        <button className="button button-primary" disabled={busy || start.isPending || !status} type="submit">
          <RefreshCw size={13} className={busy ? "spin" : ""} />{busy ? "Sync in progress" : "Sync papers"}
        </button>
      </form>
      {job?.status === "interrupted" && <button className="button resume-sync" disabled={start.isPending} onClick={() => start.mutate(true)}>Resume saved run</button>}
      {busy && <button className="pause-sync text-button" disabled={pause.isPending || Boolean(job.pause_requested)} onClick={() => pause.mutate()}>{job.pause_requested ? "Pausing after this paper…" : "Pause sync"}</button>}
      {feedback && <p className="sync-error" role="alert">{feedback}</p>}
      {job && !busy && <details className="sync-details">
        <summary>{job.limit_reached ? "Paper limit reached" : "Last sync details"}</summary>
        <p>{job.message}</p><p>{job.since.slice(0, 10)} → {job.until.slice(0, 10)}</p>
        {job.source_errors.map((message) => <p className="sync-error" key={message}>{message}</p>)}
        {job.error && <p className="sync-error">{job.error}</p>}
      </details>}
    </section>
  );
}

function windowLabel(settings: SyncSettings): string {
  if (settings.mode === "relative") return `Last ${settings.initial_months} months · ${settings.max_papers} papers per run`;
  if (settings.mode === "year") return `Year ${settings.year}`;
  if (settings.mode === "month") return `${settings.year}-${String(settings.month).padStart(2, "0")}`;
  if (settings.mode === "since") return `From ${settings.start_month} onward`;
  return `${settings.start_month} through ${settings.end_month}`;
}
